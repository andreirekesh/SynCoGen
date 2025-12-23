import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from typing import List
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Lipinski
import numpy as np
from rdkit import RDLogger


# Add syncogen to path for testing
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from syncogen.api.rdkit.assembly import RDKitMoleculeAssembly
from syncogen.utils.rdkit import is_valid_smiles
from syncogen.api.graph.graph import BBRxnGraph
from syncogen.constants.constants import (
    BUILDING_BLOCKS,
    BUILDING_BLOCKS_SMI_TO_IDX,
    N_BUILDING_BLOCKS,
    N_REACTIONS,
    N_CENTERS,
)
from syncogen.utils.rdkit import get_lipinski_descriptors


def lipinski_mask(
    descriptors_frag: List[dict],
    descriptor_current: dict,
    max_mol_weight: float = 600.0,
    max_mol_hbd: int = 5,
    max_mol_hba: int = 10,
) -> List[bool]:
    """Check if adding each fragment would violate Lipinski rules."""

    descriptors_frag_array = np.array(
        [list(descriptor.values()) for descriptor in descriptors_frag]
    )
    molwt_frag = descriptors_frag_array[:, 0]
    descriptors_current_array = np.array(list(descriptor_current.values()))
    descriptors_total = descriptors_frag_array + descriptors_current_array

    # Check violations: MW, LogP (not checked), HBD, HBA
    valid_mask = (
        (descriptors_total[:, 0] <= max_mol_weight)  # MW
        & (descriptors_total[:, 1] <= max_mol_hbd)  # HBD
        & (descriptors_total[:, 2] <= max_mol_hba)  # HBA
    )

    return valid_mask.tolist()


def enumerate_possible_actions(
    molecule: RDKitMoleculeAssembly,
    comp_r1: torch.Tensor,
    comp_r2: torch.Tensor,
    bb_descriptors: List[dict],
    max_mol_weight: float = 600.0,
    max_mol_hbd: int = 5,
    max_mol_hba: int = 10,
    sample_by_inverse_molwt: bool = False,
):
    if molecule.num_fragments() == 0:
        # Initial BB sampling - apply Lipinski mask and sample by molwt if requested
        lipinski_valid = lipinski_mask(
            bb_descriptors,
            {"MW": 0.0, "HBD": 0, "HBA": 0},
            max_mol_weight,
            max_mol_hbd,
            max_mol_hba,
        )
        valid_indices = [i for i in range(N_BUILDING_BLOCKS) if lipinski_valid[i]]

        if sample_by_inverse_molwt:
            return [(valid_indices, [bb_descriptors[i]["MW"] for i in valid_indices])]
        else:
            return [valid_indices]

    results = []
    n_bbs, n_rxns, n_centers = comp_r1.shape

    frags_and_availability = [
        (
            frag_order,
            molecule.fragment_graph.nodes[frag_order]["node"].smiles,
            molecule.fragment_graph.nodes[frag_order]["rxn_center_available"],
        )
        for frag_order in molecule.fragment_graph.nodes
    ]

    descriptors_frag = get_lipinski_descriptors(
        [Chem.MolFromSmiles(frag_smiles) for _, frag_smiles, _ in frags_and_availability]
    )

    try:
        descriptors_current = get_lipinski_descriptors(molecule.to_mol(sanitize=True))
    except:
        print("Molecule could not be sanitized")
        return []

    # Apply Lipinski mask
    lipinski_valid = lipinski_mask(
        descriptors_frag, descriptors_current, max_mol_weight, max_mol_hbd, max_mol_hba
    )
    frags_and_availability = [
        frags_and_availability[i] for i in range(len(frags_and_availability)) if lipinski_valid[i]
    ]
    descriptors_frag = [
        descriptors_frag[i] for i in range(len(descriptors_frag)) if lipinski_valid[i]
    ]

    # Build results with molecular weights for later sampling
    for idx, (frag_order, frag_smiles, center_availability) in enumerate(frags_and_availability):
        frag_idx = BUILDING_BLOCKS_SMI_TO_IDX[frag_smiles]["index"]
        frag_molwt = descriptors_frag[idx]["MW"]

        for cidx, is_open in enumerate(center_availability):
            if not is_open:
                continue
            if cidx >= n_centers:
                continue

            # All reactions for which this fragment is compatible as reactant 1 at center cidx
            rxn_mask = comp_r1[frag_idx, :, cidx]  # (n_rxns,)
            rxn_indices = rxn_mask.nonzero(as_tuple=True)[0]

            for rxn_idx in rxn_indices.tolist():
                # All (other_bb_idx, other_center_idx) compatible as reactant 2 for this reaction
                partners_mask = comp_r2[:, rxn_idx, :]  # (n_bbs, n_centers)
                other_idxs, oc_idxs = partners_mask.nonzero(as_tuple=True)

                for other_idx, oc in zip(other_idxs.tolist(), oc_idxs.tolist()):
                    action = [
                        other_idx,
                        rxn_idx,
                        frag_order,
                        cidx,
                        oc,
                    ]
                    if sample_by_inverse_molwt:
                        # Store action with its molecular weight for weighted sampling
                        results.append((action, frag_molwt))
                    else:
                        results.append(action)

    return results


def sample_random_molecules(
    n: int,
    length: List[int],
    comp_r1: torch.Tensor,
    comp_r2: torch.Tensor,
    seed: int = None,
    sample_by_inverse_molwt: bool = False,
    molwt_temperature: float = 1.0,
) -> List[Tuple[str, nx.Graph]]:
    """
    Sample random molecules by building the fragment graph using depth-first search.
    If `length` is a list, a random length is chosen for each molecule.
    """
    if seed is not None:
        random.seed(seed)

    # Pre-compute BB descriptors once
    bb_descriptors = get_lipinski_descriptors(
        [Chem.MolFromSmiles(bb_smiles) for bb_smiles in BUILDING_BLOCKS]
    )

    def dfs(
        mfg: RDKitMoleculeAssembly, target_length: int, actions_taken=None
    ) -> Tuple[List, RDKitMoleculeAssembly]:
        if actions_taken is None:
            actions_taken = []

        if mfg.num_fragments() == target_length:
            return actions_taken, mfg

        possible_actions = enumerate_possible_actions(
            mfg, comp_r1, comp_r2, bb_descriptors, sample_by_inverse_molwt=sample_by_inverse_molwt
        )

        # Handle weighted sampling by inverse molecular weight
        if sample_by_inverse_molwt and possible_actions:
            if mfg.num_fragments() == 0:
                # Initial BB sampling: possible_actions is a tuple (valid_indices, molwts)
                valid_indices, molwts = possible_actions[0]
                actions = [[i] for i in valid_indices]
            else:
                # Subsequent actions: possible_actions is a list of (action, molwt) tuples
                actions = [item[0] for item in possible_actions]
                molwts = [item[1] for item in possible_actions]

            # Apply inverse molecular weight sampling with temperature-based softmax
            molwts = np.array(molwts)
            # Inverse weights (lower molwt = higher weight)
            inv_weights = 1.0 / (molwts + 1e-6)
            # Apply temperature scaling and softmax
            scaled_weights = inv_weights / molwt_temperature
            exp_weights = np.exp(
                scaled_weights - np.max(scaled_weights)
            )  # subtract max for numerical stability
            weights = exp_weights / exp_weights.sum()
            sampled_indices = np.random.choice(
                len(actions), size=len(actions), replace=False, p=weights
            )
            possible_actions = [actions[i] for i in sampled_indices]
        else:
            # Uniform random sampling
            if mfg.num_fragments() == 0:
                valid_indices = possible_actions[0]
                possible_actions = [[i] for i in valid_indices]
            random.shuffle(possible_actions)

        for action in possible_actions:
            new_mfg = mfg.copy()
            new_mfg.add_fragment(*action)

            final_actions, final_mfg = dfs(new_mfg, target_length, actions_taken + [action])
            if final_mfg is not None:
                return final_actions, final_mfg

        return None, None

    graph_data = []
    labeled_molecules = []
    seen_smiles = set()
    pbar = tqdm(total=n)

    while len(graph_data) < n:
        # Pick molecule length
        if isinstance(length, list):
            target_length = random.choice(length)
        else:
            target_length = length

        frag_indices = []
        final_actions, mfg = dfs(RDKitMoleculeAssembly(), target_length)

        if mfg:
            reaction_info = []
            for idx, action in enumerate(final_actions):
                frag_indices.append(action[0])
                if idx > 0:
                    reaction_info.append((action[1], action[2], idx, action[3], action[4]))

            adj_matrix = torch.tensor(nx.to_numpy_array(mfg.fragment_graph))
            adj_matrix = torch.maximum(adj_matrix, adj_matrix.T)

            graph = BBRxnGraph.from_tuple(torch.tensor(frag_indices), torch.tensor(reaction_info))
            X = graph.bb_onehot
            E = graph.rxn_onehot
            edge_index = adj_matrix.nonzero().t()
            edge_attr = E[edge_index[0], edge_index[1]]

            smiles, mol = mfg.to_smiles(), mfg.to_mol()
            if smiles not in seen_smiles and is_valid_smiles(mol) and not "." in smiles:
                seen_smiles.add(smiles)
                sample_graph = Data(
                    x=X,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    data_index=len(graph_data),
                    smiles=smiles,
                )
                graph_data.append(sample_graph)
                labeled_molecules.append(mol)
                pbar.update(1)
            else:
                print(f"Failed to reconstruct smiles or duplicate for {smiles}, {X.argmax(dim=-1)}")

    pbar.close()
    if args.save_conformers:
        xtb_utils.process_molecules(
            labeled_molecules,
            do_xtb=args.xtb,
            out_path=args.conformer_dir,
            num_conformers=args.num_conformers,
            rmsd_threshold=args.rmsd_threshold,
            energy_cutoff=args.energy_cutoff,
        )
    return graph_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample random molecules by building fragment graphs."
    )
    parser.add_argument(
        "-n", "--num_samples", type=int, default=100, help="Number of molecules to sample"
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        nargs="+",
        default=[2],
        help="Length(s) of molecules in fragments (default: 2). "
        "If multiple are provided, a random one will be chosen for each molecule.",
    )
    parser.add_argument("--save_graphs", action="store_true", help="Save output as numpy arrays")
    parser.add_argument(
        "--output_dir", type=str, default="data/molecule_graphs", help="Directory to save molecules"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--sample_by_inverse_molwt",
        action="store_true",
        help="Sample building blocks inversely proportional to their molecular weight",
    )
    parser.add_argument(
        "--molwt_temperature",
        type=float,
        default=1.0,
        help="Temperature for molecular weight sampling (higher = more uniform, lower = more peaked)",
    )

    # Conformer options
    parser.add_argument("--save_conformers", action="store_true", help="Save conformers")
    parser.add_argument(
        "--num_conformers", type=int, default=50, help="Number of conformers to generate"
    )
    parser.add_argument(
        "--energy_cutoff", type=float, default=10.0, help="Energy cutoff (kcal/mol)"
    )
    parser.add_argument(
        "--rmsd_threshold", type=float, default=1.5, help="RMSD threshold for clustering"
    )
    parser.add_argument(
        "--conformer_dir", type=str, default="data/conformers", help="Directory to save conformers"
    )
    parser.add_argument(
        "--xtb", action="store_true", help="Run XTB optimization on molecules and save conformers"
    )

    # Path to compatibility tensor
    parser.add_argument(
        "--compat_path",
        type=str,
        default="vocabulary/compatibility.pt",
        help="Path to compact (n_bbs x n_rxns x n_centers) compatibility tensor",
    )

    args = parser.parse_args()

    # Make output dirs if needed
    if args.save_graphs:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.save_conformers:
        os.makedirs(args.conformer_dir, exist_ok=True)

    # Load compatibilities and decode roles
    compatibility = torch.load(args.compat_path)
    compatibility = compatibility[:N_BUILDING_BLOCKS, :N_REACTIONS, :N_CENTERS].to(torch.int64)
    comp_r1 = (compatibility & 1) != 0  # (n_bbs, n_rxns, n_centers)
    comp_r2 = (compatibility & 2) != 0  # (n_bbs, n_rxns, n_centers)

    # Generate molecules
    molecules = sample_random_molecules(
        n=args.num_samples,
        length=args.length,
        comp_r1=comp_r1,
        comp_r2=comp_r2,
        seed=args.seed,
        sample_by_inverse_molwt=args.sample_by_inverse_molwt,
        molwt_temperature=args.molwt_temperature,
    )

    used_bbs = set()
    # Track Lipinski's rules and QED scores
    lipinski_stats = {"mw": [], "logp": [], "hbd": [], "hba": [], "qed": []}

    for molecule in molecules:
        bbs_mol = list(torch.argmax(molecule.x, dim=-1).numpy())
        print(molecule.smiles)
        used_bbs.update(bbs_mol)

        # Calculate molecular properties
        mol = Chem.MolFromSmiles(molecule.smiles)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            qed_score = QED.qed(mol)

            lipinski_stats["mw"].append(mw)
            lipinski_stats["logp"].append(logp)
            lipinski_stats["hbd"].append(hbd)
            lipinski_stats["hba"].append(hba)
            lipinski_stats["qed"].append(qed_score)

    # Print averaged statistics
    print("\n=== Averaged Molecular Property Statistics ===")
    print(f"Total molecules: {len(molecules)}")
    if lipinski_stats["mw"]:
        print(
            f"Average Molecular Weight: {sum(lipinski_stats['mw'])/len(lipinski_stats['mw']):.2f}"
        )
        print(f"Average LogP: {sum(lipinski_stats['logp'])/len(lipinski_stats['logp']):.2f}")
        print(f"Average H-Bond Donors: {sum(lipinski_stats['hbd'])/len(lipinski_stats['hbd']):.2f}")
        print(
            f"Average H-Bond Acceptors: {sum(lipinski_stats['hba'])/len(lipinski_stats['hba']):.2f}"
        )
        print(f"Average QED Score: {sum(lipinski_stats['qed'])/len(lipinski_stats['qed']):.4f}")
    #     print(torch.argmax(molecule.x, dim=-1))
    # print(len(used_bbs))
    # print("UNUSED: ")
    # print([bb_idx for bb_idx in range(N_BUILDING_BLOCKS) if bb_idx not in used_bbs])

    if args.save_graphs:
        out_path = os.path.join(args.output_dir, "dataset_list_full.pt")
        torch.save(molecules, out_path)
        print(f"Saved {len(molecules)} molecules to {out_path}")
    else:
        print(f"Generated {len(molecules)} molecules")
