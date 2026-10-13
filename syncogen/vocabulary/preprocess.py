from pathlib import Path
import argparse
import json
import time
import csv
from typing import Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit import RDLogger
from tqdm import tqdm
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Disable RDKit warnings globally for this script
RDLogger.DisableLog("rdApp.*")


def lazy_diverse_pick(
    bbs: list[str],
    n_pick: int,
    n_threads: int = 8,
) -> list[str]:
    """
    Use MaxMinPicker to select a diverse subset of building blocks.

    Args:
        bbs: List of building block SMILES strings
        n_pick: Number of building blocks to select
        n_threads: Number of threads for fingerprint generation

    Returns:
        List of selected SMILES strings
    """
    if n_pick >= len(bbs):
        return bbs

    # Generate fingerprints
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    fps = []
    for smiles in tqdm(bbs, desc="Generating fingerprints"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            fp = fpg.GetFingerprint(mol)
            fps.append(fp)
        except Exception as e:
            print(f"Error generating fingerprint for {smiles}: {e}")
            fps.append(None)

    # Filter out None fingerprints and track indices
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    valid_fps = [fps[i] for i in valid_indices]

    if n_pick >= len(valid_fps):
        return [bbs[i] for i in valid_indices]

    # Perform lazy diverse picking
    mmp = rdSimDivPickers.MaxMinPicker()
    picked_indices = mmp.LazyBitVectorPick(valid_fps, len(valid_fps), n_pick)

    # Map back to original building block indices
    selected_bbs = [bbs[valid_indices[i]] for i in picked_indices]

    return selected_bbs


def validate_reaction_templates(templates):
    """Return {smarts: reason} for every template RDKit cannot actually execute.

    `ReactionFromSmarts` happily parses a template whose atom-map numbers are
    duplicated (e.g. `[S:10](=[O:15])=[O:15]`, both sulfonyl oxygens mapped :15),
    and `IsInitialized()` is lazy so it reports False for valid templates too --
    neither catches it. The failure only shows up later as
    `ChemicalParserException: initMatchers() must be called before runReactants()`,
    at which point the template silently yields no products for anything that tries
    to run it (retrosynthesis checks, eval, validity scoring) while the assembly,
    which never calls RunReactants, carries on using it.

    `Validate()` is the reliable test, so use it, plus an explicit duplicate-map
    check to give an actionable message.
    """
    import re
    from collections import Counter

    problems = {}
    for smarts in templates:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        if rxn is None:
            problems[smarts] = "does not parse"
            continue
        dups = []
        for side in smarts.split(">>"):
            counts = Counter(int(m) for m in re.findall(r":(\d+)\]", side))
            dups.extend(sorted(k for k, v in counts.items() if v > 1))
        try:
            n_warn, n_err = rxn.Validate(silent=True)
        except Exception as exc:
            problems[smarts] = f"Validate() raised: {exc}"
            continue
        if n_err:
            msg = f"{n_err} validation error(s)"
            if dups:
                msg += f"; atom map number(s) {sorted(set(dups))} used more than once"
            problems[smarts] = msg
            continue
        if dups:
            problems[smarts] = (
                f"atom map number(s) {sorted(set(dups))} used more than once"
            )
            continue

        # RDKitMoleculeAssembly copies each fragment's internal bonds verbatim and
        # only ADDS the new inter-fragment bond -- it cannot change the order of an
        # existing bond. A template that saturates an alkene (Michael addition,
        # C=C -> C-C) is therefore inexpressible: the assembly keeps the double bond,
        # giving an over-valent carbon that fails sanitisation.
        def _mapped_bonds(templates):
            out = {}
            for t in templates:
                idx2map = {a.GetIdx(): a.GetAtomMapNum() for a in t.GetAtoms()}
                for b in t.GetBonds():
                    i = idx2map[b.GetBeginAtomIdx()]
                    j = idx2map[b.GetEndAtomIdx()]
                    if i > 0 and j > 0:
                        out[frozenset((i, j))] = str(b.GetBondType())
            return out

        r_bonds = _mapped_bonds(
            [rxn.GetReactantTemplate(i) for i in range(rxn.GetNumReactantTemplates())]
        )
        p_bonds = _mapped_bonds(
            [rxn.GetProductTemplate(i) for i in range(rxn.GetNumProductTemplates())]
        )
        changed = [
            f"{sorted(k)}: {v} -> {p_bonds[k]}"
            for k, v in r_bonds.items()
            if k in p_bonds and p_bonds[k] != v
        ]
        if changed:
            problems[smarts] = (
                "changes the order of a bond inside a fragment "
                f"({'; '.join(changed)}); the assembly can only add the new "
                "inter-fragment bond, so this reaction is inexpressible"
            )
    return problems


def get_smarts_reaction_sites(smarts: str):
    """Per reactant, return {"attach": idx, "leaving": [idx, ...]} in template-local
    atom indices.

    The previous implementation returned one (index, drop_flag) per reactant and
    asserted exactly one reaction centre, which forbade reactions that remove
    more than one atom. The CENTRE is now the ATTACHMENT atom
    (the atom that forms the new inter-fragment bond and survives into the product)
    rather than the leaving atom, which makes the attachment explicit instead of
    recovering it at assembly time via `next(neighbors(...))` -- ambiguous as soon as
    a leaving atom has degree > 1, which is exactly what boron's B(OH)2 introduces.
    """
    rxn = rdChemReactions.ReactionFromSmarts(smarts)
    if rxn is None:
        raise ValueError(f"unparseable SMARTS: {smarts}")
    rtemps = [rxn.GetReactantTemplate(i) for i in range(rxn.GetNumReactantTemplates())]
    ptemps = [rxn.GetProductTemplate(i) for i in range(rxn.GetNumProductTemplates())]

    prod_map = {}
    for ptemp in ptemps:
        for atom in ptemp.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                prod_map[atom.GetAtomMapNum()] = atom

    sites = []
    for rtemp in rtemps:
        leaving, degree_changed = [], []
        for atom in rtemp.GetAtoms():
            amap, idx = atom.GetAtomMapNum(), atom.GetIdx()
            if amap == 0 or amap not in prod_map:
                leaving.append(idx)
            elif atom.GetDegree() != prod_map[amap].GetDegree():
                degree_changed.append(idx)
        leaving_set = set(leaving)

        adjacent = []
        for li in leaving:
            for nb in rtemp.GetAtomWithIdx(li).GetNeighbors():
                j = nb.GetIdx()
                if j not in leaving_set and j not in adjacent:
                    adjacent.append(j)

        if len(degree_changed) == 1:
            attach = degree_changed[0]
        elif len(degree_changed) == 0 and len(adjacent) == 1:
            attach = adjacent[0]
        elif len(degree_changed) > 1:
            inter = [d for d in degree_changed if d in adjacent]
            if len(inter) != 1:
                raise ValueError(
                    f"ambiguous attachment: degree-changed={degree_changed} "
                    f"adjacent-to-leaving={adjacent} in {smarts}"
                )
            attach = inter[0]
        elif len(adjacent) > 1:
            raise ValueError(f"ambiguous attachment: adjacent={adjacent} in {smarts}")
        else:
            raise ValueError(f"no attachment atom found in a reactant of {smarts}")

        if attach in leaving_set:
            raise ValueError(f"attachment {attach} is also leaving in {smarts}")

        kept = [a.GetIdx() for a in rtemp.GetAtoms() if a.GetIdx() not in leaving_set]
        if kept:
            seen, stack = {kept[0]}, [kept[0]]
            while stack:
                cur = stack.pop()
                for nb in rtemp.GetAtomWithIdx(cur).GetNeighbors():
                    j = nb.GetIdx()
                    if j not in leaving_set and j not in seen:
                        seen.add(j)
                        stack.append(j)
            if len(seen) != len(kept):
                raise ValueError(
                    f"removing the leaving set disconnects a reactant of {smarts}"
                )
        sites.append({"attach": attach, "leaving": sorted(leaving_set)})
    return sites


def bb_leaving_group(mol, rtemp, match, site):
    """Map a template match onto a building block; return (attach_idx, leaving_idxs).

    The template names only its own atoms, so `[c:2]-[B]` marks B as leaving but says
    nothing about the two hydroxyls of a real B(OH)2. Everything reachable from a
    matched leaving atom without passing through the attachment atom leaves with it.
    """
    attach_bb = match[site["attach"]]
    seeds = [match[i] for i in site["leaving"] if i < len(match)]
    leaving = set(seeds)
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j != attach_bb and j not in leaving:
                leaving.add(j)
                stack.append(j)

    # A leaving group must be PENDANT: exactly one bond may cross between the kept
    # part and the leaving set. The template-level connectivity check is not enough,
    # because a template like `[c:2]-[B]` has boron terminal while a real
    # benzoxaborole has it inside a ring -- the flood-fill then travels around the
    # ring and swallows the molecule (18 of 19 atoms, keeping one carbon). Returning
    # None marks the (bb, reaction, centre) slot as not actually applicable.
    if leaving:
        crossing = 0
        for a in leaving:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() not in leaving:
                    crossing += 1
        if crossing != 1:
            return attach_bb, None
    return attach_bb, sorted(leaving)


def load_templates(path: Path, smarts_column: Optional[str] = None):
    """Load reaction templates (SMARTS) from a text file or CSV.

    - If ``path`` is a CSV, ``smarts_column`` must be provided and will be
      used to extract the SMARTS strings.
    - Otherwise, each non-empty line is treated as a template.
    """
    if path.suffix.lower() == ".csv":
        if not smarts_column:
            raise ValueError(
                f"smarts_column must be provided when loading templates from CSV: {path}"
            )
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if smarts_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{smarts_column}' not found in CSV file {path}. "
                    f"Available columns: {reader.fieldnames}"
                )
            return [
                row[smarts_column].strip()
                for row in reader
                if row.get(smarts_column, "").strip()
            ]

    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def load_bbs(path: Path, smiles_column: Optional[str] = None):
    """Load building block SMILES from a text file or CSV.

    - If ``path`` is a CSV, ``smiles_column`` must be provided and will be
      used to extract the SMILES strings.
    - Otherwise, each non-empty line is treated as a SMILES string.
    """
    if path.suffix.lower() == ".csv":
        if not smiles_column:
            raise ValueError(
                f"smiles_column must be provided when loading building blocks from CSV: {path}"
            )
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if smiles_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{smiles_column}' not found in CSV file {path}. "
                    f"Available columns: {reader.fieldnames}"
                )
            return [
                row[smiles_column].strip()
                for row in reader
                if row.get(smiles_column, "").strip()
            ]

    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def build_compatibility_and_bbs(
    bbs: list[str],
    templates: list[str],
    annotated_smarts: dict,
    max_centers: int,
):
    """Build compatibility tensor and building-block center annotations."""
    num_bbs = len(bbs)
    num_rxns = len(templates)
    compatibility = torch.zeros((num_bbs, num_rxns, max_centers), dtype=torch.uint8)

    annotated_bbs: dict[str, list[int]] = {}
    reaction_usage = {smarts: 0 for smarts in templates}

    # Parse each reaction template ONCE. This used to sit inside the per-building-block
    # loop, i.e. ReactionFromSmarts was called len(bbs) * len(templates) times
    # (~4.8M times for a 64k x 75 vocabulary) when len(templates) parses suffice.
    parsed_templates = []
    for smarts in templates:
        rxn = AllChem.ReactionFromSmarts(smarts)
        reactants = rxn.GetReactants()
        parsed_templates.append(
            (
                reactants[0] if len(reactants) > 0 else None,
                reactants[1] if len(reactants) > 1 else None,
            )
        )

    # int32 bitmask over building-block atom slots: bit a set => atom a leaves.
    # Computed here because it needs the same substructure matches as compatibility.
    leaving_mask = torch.zeros((num_bbs, num_rxns, max_centers), dtype=torch.int32)
    non_pendant = 0
    role_conflicts = []

    for i, smiles in enumerate(tqdm(bbs, desc="Building BB compatibilities")):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        centers: list[int] = []

        for j, smarts in enumerate(templates):
            r1, r2 = parsed_templates[j]

            any_comp_for_rxn = False

            if r1 is not None:
                matches1 = mol.GetSubstructMatches(r1)
                if matches1:
                    site1 = annotated_smarts[smarts][0]
                    rc_idx_r1 = site1["attach"]
                    for match in matches1:
                        atom_idx = match[rc_idx_r1]
                        if atom_idx not in centers:
                            centers.append(atom_idx)
                        center_idx = centers.index(atom_idx)
                        if center_idx < max_centers:
                            _, leaving = bb_leaving_group(mol, r1, match, site1)
                            if leaving is None:
                                # non-pendant leaving group: template does not apply
                                non_pendant += 1
                                continue
                            bits = 0
                            for a in leaving:
                                bits |= 1 << a
                            prev = int(leaving_mask[i, j, center_idx])
                            if prev and prev != bits:
                                role_conflicts.append((smiles, j, center_idx))
                            leaving_mask[i, j, center_idx] = bits
                            compatibility[i, j, center_idx] |= 1
                            any_comp_for_rxn = True

            if r2 is not None:
                matches2 = mol.GetSubstructMatches(r2)
                if matches2:
                    site2 = annotated_smarts[smarts][1]
                    rc_idx_r2 = site2["attach"]
                    for match in matches2:
                        atom_idx = match[rc_idx_r2]
                        if atom_idx not in centers:
                            centers.append(atom_idx)
                        center_idx = centers.index(atom_idx)
                        if center_idx < max_centers:
                            _, leaving = bb_leaving_group(mol, r2, match, site2)
                            if leaving is None:
                                # non-pendant leaving group: template does not apply
                                non_pendant += 1
                                continue
                            bits = 0
                            for a in leaving:
                                bits |= 1 << a
                            prev = int(leaving_mask[i, j, center_idx])
                            if prev and prev != bits:
                                role_conflicts.append((smiles, j, center_idx))
                            leaving_mask[i, j, center_idx] = bits
                            compatibility[i, j, center_idx] |= 2
                            any_comp_for_rxn = True

            if any_comp_for_rxn:
                reaction_usage[smarts] += 1

        if centers:
            # Truncate to max_centers so the stored centre list, the compatibility
            # tensor and leaving.pt all have the same addressable range. Previously
            # the full list was stored while the tensors had only max_centers slots,
            # so RDKitMoleculeAssembly.all_open_reaction_centers() could yield a
            # centre index that then raised IndexError against COMPATIBILITY.
            if len(centers) > max_centers:
                print(
                    f"Warning: building block {smiles} has {len(centers)} reaction centers, "
                    f"exceeding max_centers_per_bb={max_centers}; extra centers are dropped."
                )
                centers = centers[:max_centers]
            annotated_bbs[smiles] = centers

    if non_pendant:
        print(
            f"Skipped {non_pendant} (bb, reaction, centre) slots whose leaving group is "
            f"not pendant (leaving atom shares a ring with the attachment, so the "
            f"template does not genuinely apply)."
        )

    if role_conflicts:
        raise RuntimeError(
            f"{len(role_conflicts)} (bb, reaction, centre) slots have different leaving "
            f"groups depending on whether the BB acts as reactant 1 or 2 "
            f"(e.g. {role_conflicts[:3]}). leaving_mask must become role-aware "
            f"(shape [n_bb, n_rxn, n_centers, 2]) before this vocabulary can be used."
        )

    return compatibility, annotated_bbs, reaction_usage, leaving_mask


def filter_bbs(
    bbs: list[str],
    compatibility: torch.Tensor,
    annotated_bbs: dict[str, list[int]],
    keep_incompatible: bool,
    leaving_mask: torch.Tensor = None,
):
    """Filter building blocks without reaction centers if requested.

    leaving_mask is row-indexed identically to compatibility and must be sliced in
    lockstep, or atom bookkeeping silently decouples from the vocabulary indices.
    """
    if keep_incompatible:
        return bbs, compatibility, annotated_bbs, leaving_mask

    kept_bbs = list(annotated_bbs.keys())
    keep_indices = [i for i, smiles in enumerate(bbs) if smiles in annotated_bbs]
    compatibility = compatibility[keep_indices]
    if leaving_mask is not None:
        leaving_mask = leaving_mask[keep_indices]
    return kept_bbs, compatibility, annotated_bbs, leaving_mask


def filter_reactions(
    templates: list[str],
    annotated_smarts: dict,
    compatibility: torch.Tensor,
    reaction_usage: dict[str, int],
    keep_incompatible: bool,
    leaving_mask: torch.Tensor = None,
):
    """Filter reactions incompatible with all BBs if requested."""
    if keep_incompatible:
        return templates, annotated_smarts, compatibility, leaving_mask

    keep_rxn_indices = [
        j for j, smarts in enumerate(templates) if reaction_usage.get(smarts, 0) > 0
    ]
    filtered_templates = [templates[j] for j in keep_rxn_indices]
    filtered_annotations = {
        smarts: annotated_smarts[smarts] for smarts in filtered_templates
    }
    compatibility = compatibility[:, keep_rxn_indices, :]
    if leaving_mask is not None:
        leaving_mask = leaving_mask[:, keep_rxn_indices, :]
    return filtered_templates, filtered_annotations, compatibility, leaving_mask


def filter_bbs_by_properties(
    bbs: list[str],
    max_atoms_per_bb: int,
    allowed_atom_types: set[str],
) -> list[str]:
    """
    Property-level filtering of building blocks, independent of compatibility.

    Steps:
      - Drop SMILES containing '.' (disjoint fragments)
      - Parse without sanitization, then explicitly sanitize; drop failures
      - Canonicalize SMILES and deduplicate
      - Enforce max_atoms_per_bb
      - Enforce allowed_atom_types
    """
    filtered_bbs: list[str] = []
    seen_smiles: set[str] = set()

    for smiles in tqdm(bbs, desc="Property-filtering BBs"):
        # Drop disjoint molecules upfront
        if "." in smiles:
            continue

        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # Invalid / unsanitizable molecule – drop it.
            continue

        # Canonicalize SMILES in a kekulized form so that downstream code
        # consistently sees explicit double/single bonds rather than
        # aromatic "c" notation.
        can_smi = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)
        if can_smi in seen_smiles:
            continue
        seen_smiles.add(can_smi)

        # Atom-count and atom-type filters
        if mol.GetNumAtoms() > max_atoms_per_bb:
            continue
        if any(atom.GetSymbol() not in allowed_atom_types for atom in mol.GetAtoms()):
            continue

        filtered_bbs.append(can_smi)

    return filtered_bbs


def filter_bbs_unwanted_substructures(bbs: list[str]):
    """Filter building blocks with unwanted substructures."""
    filtered_bbs: list[str] = []
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
    catalog_all = FilterCatalog(params)
    for smiles in tqdm(bbs, desc="Filtering BBs with unwanted substructures"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        if catalog_all.HasMatch(mol):
            continue
        filtered_bbs.append(smiles)
    return filtered_bbs


def write_vocab_jsons(
    out_dir: Path,
    bbs: list[str],
    annotated_bbs: dict[str, list[int]],
    annotated_smarts: dict,
):
    """Write building_blocks.json and reactions.json."""
    building_blocks_out = {smiles: annotated_bbs.get(smiles, []) for smiles in bbs}
    (out_dir / "building_blocks.json").write_text(
        json.dumps(building_blocks_out, indent=2)
    )

    reactions_out = {}
    for idx, (smarts, sites) in enumerate(annotated_smarts.items()):
        # drop_bools is retained so older vocabularies/readers keep working, but it is
        # now derived from the leaving sets and is no longer the source of truth --
        # leaving.pt is (it carries per-BB leaving groups, which drop_bools cannot).
        drop_bools = [len(site["leaving"]) > 0 for site in sites]
        n_leaving = [len(site["leaving"]) for site in sites]
        attach = [site["attach"] for site in sites]
        while len(drop_bools) < 2:
            drop_bools.append(False)
            n_leaving.append(0)
            attach.append(-1)
        reactions_out[smarts] = {
            "drop_bools": drop_bools,
            "n_leaving_template": n_leaving,
            "attach_template": attach,
            "index": idx,
        }

    (out_dir / "reactions.json").write_text(json.dumps(reactions_out, indent=2))
    return building_blocks_out


def compute_fragment_features(
    bbs: list[str],
    building_blocks_out: dict[str, list[int]],
    compatibility: torch.Tensor,
):
    """Compute fragment-level features and metadata."""
    n_bbs_final, n_rxns_final, n_centers_final = compatibility.shape

    mols = []
    max_atoms_per_bb = 0
    atom_type_set: set[str] = set()
    for smiles in tqdm(bbs, desc="Loading BB molecules"):
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)
        if mol is not None:
            max_atoms_per_bb = max(max_atoms_per_bb, mol.GetNumAtoms())
            for atom in mol.GetAtoms():
                atom_type_set.add(atom.GetSymbol())

    # Dynamically determine atom types from the vocabulary instead of using a
    # fixed list. This keeps atom features in sync with whatever elements are
    # actually present in the building blocks.
    atom_types = sorted(atom_type_set)
    n_atom_features = len(atom_types) + 3
    n_bond_features = 5

    fragment_maccs = torch.zeros((n_bbs_final + 1, 167), dtype=torch.float32)
    fragment_atomfeats = torch.zeros(
        (n_bbs_final + 1, max_atoms_per_bb, n_atom_features), dtype=torch.float32
    )
    fragment_bondfeats = torch.zeros(
        (n_bbs_final + 1, max_atoms_per_bb, max_atoms_per_bb, n_bond_features),
        dtype=torch.float32,
    )
    fragment_atomadj = torch.zeros(
        (n_bbs_final + 1, max_atoms_per_bb, max_atoms_per_bb), dtype=torch.float32
    )

    for i, (smiles, mol) in enumerate(
        tqdm(list(zip(bbs, mols)), desc="Computing fragment features")
    ):
        if mol is None:
            continue

        fp = AllChem.GetMACCSKeysFingerprint(mol)
        arr = np.zeros((167,), dtype=float)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fragment_maccs[i] = torch.tensor(arr, dtype=torch.float32)

        centers = set(building_blocks_out.get(smiles, []))
        for j, atom in enumerate(mol.GetAtoms()):
            feats = []
            atom_type = atom.GetSymbol()
            feats.extend(1.0 if atom_type == t else 0.0 for t in atom_types)
            feats.append(int(atom.IsInRing()))
            feats.append(1.0 if j in centers else 0.0)
            feats.append(0.0)
            fragment_atomfeats[i, j] = torch.tensor(feats, dtype=torch.float32)

        bond_feats = torch.zeros(
            (max_atoms_per_bb, max_atoms_per_bb, n_bond_features), dtype=torch.float32
        )
        atom_adj = torch.zeros(
            (max_atoms_per_bb, max_atoms_per_bb), dtype=torch.float32
        )
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            atom_adj[a1, a2] = 1.0
            atom_adj[a2, a1] = 1.0
            if bond_type == 1.0:
                bond_feats[a1, a2, 0] = 1.0
                bond_feats[a2, a1, 0] = 1.0
            elif bond_type == 2.0:
                bond_feats[a1, a2, 1] = 1.0
                bond_feats[a2, a1, 1] = 1.0
            elif bond_type == 3.0:
                bond_feats[a1, a2, 2] = 1.0
                bond_feats[a2, a1, 2] = 1.0
            elif bond_type == 1.5:
                bond_feats[a1, a2, 3] = 1.0
                bond_feats[a2, a1, 3] = 1.0

        fragment_bondfeats[i] = bond_feats
        fragment_atomadj[i] = atom_adj

    fragment_atomfeats[-1, :, -1] = 1.0
    fragment_bondfeats[-1, :, :, -1] = 1.0

    fragment_features = {
        "fragment_maccs": fragment_maccs,
        "fragment_atomfeats": fragment_atomfeats,
        "fragment_bondfeats": fragment_bondfeats,
        "fragment_atomadj": fragment_atomadj,
    }

    meta = {
        "n_building_blocks": int(n_bbs_final),
        "n_reactions": int(n_rxns_final),
        "n_centers": int(n_centers_final),
        "max_atoms_per_bb": int(max_atoms_per_bb),
        "n_atom_features": int(n_atom_features),
        "n_bond_features": int(n_bond_features),
        "atom_types": atom_types,
    }

    return fragment_features, meta


def main():
    """Run vocabulary preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess vocabulary: annotate and build compatibility tensor"
    )
    parser.add_argument(
        "--bb_input",
        type=str,
        required=True,
        help="Path to building blocks file (.txt or .smi)",
    )
    parser.add_argument(
        "--rxn_input",
        type=str,
        required=True,
        help="Path to reaction templates file (.txt)",
    )
    parser.add_argument(
        "--bb_smiles_column",
        type=str,
        default=None,
        help="Column name in CSV building-block file containing SMILES (required if --bb_input ends with .csv)",
    )
    parser.add_argument(
        "--rxn_smarts_column",
        type=str,
        default=None,
        help="Column name in CSV reaction file containing SMARTS (required if --rxn_input ends with .csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--keep_incompatible",
        action="store_true",
        help="Keep BBs/reactions without compatible reaction centers",
    )
    parser.add_argument(
        "--max_centers_per_bb",
        type=int,
        default=3,
        help="Maximum number of reaction centers per building block stored in compatibility tensor",
    )
    parser.add_argument(
        "--diverse_subset",
        type=int,
        default=None,
        help=(
            "If specified, use MaxMinPicker to select this many diverse building blocks "
            "from the property-filtered BBs before compatibility computation"
        ),
    )
    parser.add_argument(
        "--max_atoms_per_bb_filter",
        type=int,
        default=16,
        help="Maximum number of atoms allowed per building block before filtering (applied on initial BB set)",
    )
    parser.add_argument(
        "--allowed_atom_types",
        type=str,
        default="C,N,O,B,F,Cl,Br,S",
        help=(
            "Comma-separated list of allowed atom symbols for building blocks "
            "(e.g. 'C,N,O,B,F,Cl,Br,S'); BBs containing other atom types are dropped"
        ),
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=8,
        help="Number of threads for fingerprint generation in diverse picking",
    )
    parser.add_argument(
        "--no_property_filtering",
        action="store_true",
        help="If set, skip property-based filtering of building blocks.",
    )
    parser.add_argument(
        "--allow_invalid_reactions",
        action="store_true",
        help=(
            "Drop reaction templates that RDKit cannot execute instead of failing. "
            "By default preprocessing aborts and lists them."
        ),
    )
    parser.add_argument(
        "--no_substructure_filtering",
        action="store_true",
        help="If set, skip unwanted substructure filtering (PAINS, Brenk, etc.) of building blocks.",
    )
    args = parser.parse_args()

    bb_path = Path(args.bb_input)
    rxn_path = Path(args.rxn_input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Annotating reactions...")
    # If reactions are provided as CSV, require rxn_smarts_column.
    templates = load_templates(rxn_path, smarts_column=args.rxn_smarts_column)
    invalid = validate_reaction_templates(templates)
    if invalid:
        print(f"{len(invalid)} of {len(templates)} reaction templates are invalid:")
        for smarts, reason in invalid.items():
            print(f"  - {reason}\n      {smarts[:110]}")
        if not args.allow_invalid_reactions:
            raise SystemExit(
                "Refusing to build a vocabulary from templates RDKit cannot execute. "
                "Repair them, or pass --allow_invalid_reactions to drop them."
            )
        templates = [t for t in templates if t not in invalid]
        print(f"Dropped them; {len(templates)} templates remain.")

    annotated_smarts = {
        smarts: get_smarts_reaction_sites(smarts) for smarts in templates
    }
    (out_dir / "templates_annotated.json").write_text(
        json.dumps(annotated_smarts, indent=2)
    )
    print(f"Annotated {len(annotated_smarts)} reactions")

    print("Loading building blocks...")
    bbs = load_bbs(bb_path, smiles_column=args.bb_smiles_column)
    print(f"Loaded {len(bbs)} building blocks")

    # Property-level filtering (independent of compatibility):
    #   - canonicalization & sanitization
    #   - disjoint-fragment removal
    #   - atom-count and atom-type filters
    if not args.no_property_filtering:
        allowed_atom_types = {t for t in args.allowed_atom_types.split(",") if t}
        max_atoms_filter = int(args.max_atoms_per_bb_filter)
        bbs = filter_bbs_by_properties(
            bbs=bbs,
            max_atoms_per_bb=max_atoms_filter,
            allowed_atom_types=allowed_atom_types,
        )
        print(f"Remaining building blocks after property filtering: {len(bbs)}")
    else:
        print(
            "Skipping property-based filtering of building blocks (--no_property_filtering set)"
        )

    if not args.no_substructure_filtering:
        bbs = filter_bbs_unwanted_substructures(bbs)
        print(
            f"Remaining building blocks after unwanted substructure filtering: {len(bbs)}"
        )
    else:
        print(
            "Skipping unwanted substructure filtering (--no_substructure_filtering set)"
        )

    # Optionally apply diverse subset selection after filtering but before
    # compatibility computations.
    if args.diverse_subset is not None and args.diverse_subset < len(bbs):
        print(
            f"Selecting {args.diverse_subset} diverse building blocks from {len(bbs)} filtered BBs..."
        )
        bbs = lazy_diverse_pick(bbs, args.diverse_subset, args.n_threads)
        print(f"Selected {len(bbs)} diverse building blocks")

    print("Annotating building blocks and constructing compatibility tensor...")
    max_centers = int(args.max_centers_per_bb)
    compatibility, annotated_bbs, reaction_usage, leaving_mask = (
        build_compatibility_and_bbs(
            bbs=bbs,
            templates=templates,
            annotated_smarts=annotated_smarts,
            max_centers=max_centers,
        )
    )

    print("Filtering building blocks...")
    bbs, compatibility, annotated_bbs, leaving_mask = filter_bbs(
        bbs=bbs,
        compatibility=compatibility,
        annotated_bbs=annotated_bbs,
        keep_incompatible=args.keep_incompatible,
        leaving_mask=leaving_mask,
    )
    print(f"Remaining building blocks after filtering: {len(bbs)}")

    (out_dir / "smiles_annotated.json").write_text(json.dumps(annotated_bbs, indent=2))
    torch.save(compatibility, out_dir / "compatibility.pt")

    templates, annotated_smarts, compatibility, leaving_mask = filter_reactions(
        templates=templates,
        annotated_smarts=annotated_smarts,
        compatibility=compatibility,
        reaction_usage=reaction_usage,
        keep_incompatible=args.keep_incompatible,
        leaving_mask=leaving_mask,
    )
    torch.save(compatibility, out_dir / "compatibility.pt")
    torch.save(leaving_mask, out_dir / "leaving.pt")
    (out_dir / "templates_annotated.json").write_text(
        json.dumps(annotated_smarts, indent=2)
    )

    print("Writing building_blocks.json and reactions.json...")
    building_blocks_out = write_vocab_jsons(
        out_dir=out_dir,
        bbs=bbs,
        annotated_bbs=annotated_bbs,
        annotated_smarts=annotated_smarts,
    )

    print("Computing fragment features and metadata...")
    fragment_features, meta = compute_fragment_features(
        bbs=bbs,
        building_blocks_out=building_blocks_out,
        compatibility=compatibility,
    )
    torch.save(fragment_features, out_dir / "fragment_features.pt")
    meta["max_centers_per_bb"] = max_centers
    # Presence of these fields is how constants.py feature-detects the new
    # attachment-centred convention; without them it falls back to the legacy
    # drop_bools path so existing vocabularies and checkpoints keep working.
    meta["centre_convention"] = "attachment"
    meta["has_leaving_masks"] = True
    nz = leaving_mask != 0
    meta["max_leaving_atoms"] = int(
        max(
            (int(v).bit_count() for v in leaving_mask[nz].flatten().tolist()), default=0
        )
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    num_bbs = len(bbs)
    print("Done.")
    print(f"  Building blocks annotated: {len(annotated_bbs)} / {num_bbs}")
    if not args.keep_incompatible:
        print(f"  Dropped BBs with no reaction centers: {num_bbs - len(annotated_bbs)}")
    print(
        f"  Compatibility tensor: {tuple(compatibility.shape)} saved to {out_dir/'compatibility.pt'}"
    )
    print(
        f"  Reactions annotated: {len(annotated_smarts)} / {len(annotated_smarts)} → {out_dir/'templates_annotated.json'}"
    )
    if not args.keep_incompatible:
        print(
            f"  Dropped reactions incompatible with all remaining BBs: {len(templates) - len(annotated_smarts)}"
        )
    print(f"  BB annotations saved to: {out_dir/'smiles_annotated.json'}")
    print(f"  Building blocks JSON: {out_dir/'building_blocks.json'}")
    print(f"  Reactions JSON: {out_dir/'reactions.json'}")
    print(f"  Fragment features: {out_dir/'fragment_features.pt'}")
    print(f"  Metadata: {out_dir/'meta.json'}")


if __name__ == "__main__":
    main()
