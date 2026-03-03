from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import networkx as nx
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Lipinski
import numpy as np
from rdkit import RDLogger
from multiprocessing import Process, Queue, Event
import queue


# Add syncogen to path for testing
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syncogen.constants.constants import load_vocabulary


def precompute_compatibility_lookups(
    comp_r1: torch.Tensor,
    comp_r2: torch.Tensor,
) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[int, Tuple[List[int], List[int]]]]:
    """Precompute nonzero lookups for compatibility tensors."""
    n_bbs, n_rxns, n_centers = comp_r1.shape

    # comp_r1[frag_idx, :, cidx].nonzero() -> reaction indices
    r1_lookup = {}
    for frag_idx in range(n_bbs):
        for cidx in range(n_centers):
            rxn_indices = comp_r1[frag_idx, :, cidx].nonzero(as_tuple=True)[0].tolist()
            if rxn_indices:
                r1_lookup[(frag_idx, cidx)] = rxn_indices

    # comp_r2[:, rxn_idx, :].nonzero() -> (bb_indices, center_indices)
    r2_lookup = {}
    for rxn_idx in range(n_rxns):
        other_idxs, oc_idxs = comp_r2[:, rxn_idx, :].nonzero(as_tuple=True)
        if len(other_idxs) > 0:
            r2_lookup[rxn_idx] = (other_idxs.tolist(), oc_idxs.tolist())

    return r1_lookup, r2_lookup


def lipinski_mask(
    descriptors_frag: List[dict],
    descriptor_current: dict,
    max_mol_weight: float = 550.0,
    max_mol_hbd: int = 5,
    max_mol_hba: int = 10,
) -> List[bool]:
    """Check if adding each fragment would violate Lipinski rules."""

    descriptors_frag_array = np.array(
        [list(descriptor.values()) for descriptor in descriptors_frag]
    )
    descriptors_current_array = np.array(list(descriptor_current.values()))
    descriptors_total = descriptors_frag_array + descriptors_current_array

    valid_mask = (
        (descriptors_total[:, 0] <= max_mol_weight)  # MW
        & (descriptors_total[:, 1] <= max_mol_hbd)  # HBD
        & (descriptors_total[:, 2] <= max_mol_hba)  # HBA
    )

    return valid_mask.tolist()


def enumerate_possible_actions(
    molecule: "RDKitMoleculeAssembly",
    r1_lookup: Dict[Tuple[int, int], List[int]],
    r2_lookup: Dict[int, Tuple[List[int], List[int]]],
    bb_descriptors: List[dict],
    n_building_blocks: int,
    max_mol_weight: float = 650.0,
    max_mol_hbd: int = 5,
    max_mol_hba: int = 10,
    sample_by_inverse_molwt: bool = False,
):
    if molecule.num_fragments() == 0:
        lipinski_valid = lipinski_mask(
            bb_descriptors,
            {"MW": 0.0, "HBD": 0, "HBA": 0},
            max_mol_weight,
            max_mol_hbd,
            max_mol_hba,
        )
        valid_indices = [i for i in range(n_building_blocks) if lipinski_valid[i]]

        if sample_by_inverse_molwt:
            return [(valid_indices, [bb_descriptors[i]["MW"] for i in valid_indices])]
        else:
            return [valid_indices]

    results = []

    frags_and_availability = [
        (
            frag_order,
            molecule.fragment_graph.nodes[frag_order]["node"].smiles,
            molecule.fragment_graph.nodes[frag_order]["rxn_center_available"],
        )
        for frag_order in molecule.fragment_graph.nodes
    ]

    descriptors_frag = get_lipinski_descriptors(
        [
            Chem.MolFromSmiles(frag_smiles)
            for _, frag_smiles, _ in frags_and_availability
        ]
    )

    try:
        descriptors_current = get_lipinski_descriptors(molecule.to_mol(sanitize=True))
    except Exception as e:
        print("Molecule could not be sanitized")
        return []

    lipinski_valid = lipinski_mask(
        descriptors_frag, descriptors_current, max_mol_weight, max_mol_hbd, max_mol_hba
    )
    frags_and_availability = [
        frags_and_availability[i]
        for i in range(len(frags_and_availability))
        if lipinski_valid[i]
    ]
    descriptors_frag = [
        descriptors_frag[i] for i in range(len(descriptors_frag)) if lipinski_valid[i]
    ]

    for idx, (frag_order, frag_smiles, center_availability) in enumerate(
        frags_and_availability
    ):
        frag_idx = constants.BUILDING_BLOCKS_SMI_TO_IDX[frag_smiles]["index"]
        frag_molwt = descriptors_frag[idx]["MW"]

        for cidx, is_open in enumerate(center_availability):
            if not is_open:
                continue

            rxn_indices = r1_lookup.get((frag_idx, cidx))
            if not rxn_indices:
                continue

            for rxn_idx in rxn_indices:
                partners = r2_lookup.get(rxn_idx)
                if not partners:
                    continue
                other_idxs, oc_idxs = partners

                for other_idx, oc in zip(other_idxs, oc_idxs):
                    action = [
                        other_idx,
                        rxn_idx,
                        frag_order,
                        cidx,
                        oc,
                    ]
                    if sample_by_inverse_molwt:
                        results.append((action, frag_molwt))
                    else:
                        results.append(action)

    return results


def _generate_one_molecule(
    length: List[int],
    r1_lookup: Dict[Tuple[int, int], List[int]],
    r2_lookup: Dict[int, Tuple[List[int], List[int]]],
    bb_descriptors: List[dict],
    n_building_blocks: int,
    sample_by_inverse_molwt: bool = False,
    molwt_temperature: float = 1.0,
    timeout: float = None,
    worker_seed: int = None,
) -> Optional[Data]:
    """Generate a single molecule. Returns a Data object if successful, None otherwise."""
    import syncogen.constants.constants as constants
    from syncogen.api.rdkit.assembly import RDKitMoleculeAssembly
    from syncogen.utils.rdkit import is_valid_smiles
    from syncogen.api.graph.graph import BBRxnGraph

    if worker_seed is not None:
        random.seed(worker_seed)

    def dfs(
        mfg: "RDKitMoleculeAssembly",
        target_length: int,
        actions_taken=None,
        dfs_start_time: float = None,
        dfs_timeout: float = None,
    ) -> Tuple[List, "RDKitMoleculeAssembly"]:
        if actions_taken is None:
            actions_taken = []

        if dfs_timeout is not None and dfs_start_time is not None:
            if (time.time() - dfs_start_time) > dfs_timeout:
                return None, None

        if mfg.num_fragments() == target_length:
            return actions_taken, mfg

        possible_actions = enumerate_possible_actions(
            mfg,
            r1_lookup,
            r2_lookup,
            bb_descriptors,
            n_building_blocks,
            sample_by_inverse_molwt=sample_by_inverse_molwt,
        )

        if dfs_timeout is not None and dfs_start_time is not None:
            if (time.time() - dfs_start_time) > dfs_timeout:
                return None, None

        if not possible_actions:
            return None, None

        if sample_by_inverse_molwt and possible_actions:
            if mfg.num_fragments() == 0:
                valid_indices, molwts = possible_actions[0]
                actions = [[i] for i in valid_indices]
            else:
                actions = [item[0] for item in possible_actions]
                molwts = [item[1] for item in possible_actions]

            molwts = np.array(molwts)
            inv_weights = 1.0 / (molwts + 1e-6)
            scaled_weights = inv_weights / molwt_temperature
            exp_weights = np.exp(scaled_weights - np.max(scaled_weights))
            weights = exp_weights / exp_weights.sum()
            sampled_indices = np.random.choice(
                len(actions), size=len(actions), replace=False, p=weights
            )
            possible_actions = [actions[i] for i in sampled_indices]
        else:
            if mfg.num_fragments() == 0:
                valid_indices = possible_actions[0]
                possible_actions = [[i] for i in valid_indices]
            random.shuffle(possible_actions)

        for action in possible_actions:
            if dfs_timeout is not None and dfs_start_time is not None:
                if (time.time() - dfs_start_time) > dfs_timeout:
                    return None, None

            new_mfg = mfg.copy()
            new_mfg.add_fragment(*action)

            final_actions, final_mfg = dfs(
                new_mfg,
                target_length,
                actions_taken + [action],
                dfs_start_time=dfs_start_time,
                dfs_timeout=dfs_timeout,
            )
            if final_mfg is not None:
                return final_actions, final_mfg

        return None, None

    if isinstance(length, list):
        target_length = random.choice(length)
    else:
        target_length = length

    start_time = time.time()
    final_actions, mfg = dfs(
        RDKitMoleculeAssembly(),
        target_length,
        dfs_start_time=start_time,
        dfs_timeout=timeout,
    )

    if mfg is None:
        return None

    reaction_info = []
    frag_indices = []
    for idx, action in enumerate(final_actions):
        frag_indices.append(action[0])
        if idx > 0:
            reaction_info.append((action[1], action[2], idx, action[3], action[4]))

    adj_matrix = torch.tensor(nx.to_numpy_array(mfg.fragment_graph))
    adj_matrix = torch.maximum(adj_matrix, adj_matrix.T)

    graph = BBRxnGraph.from_tuple(
        torch.tensor(frag_indices), torch.tensor(reaction_info)
    )
    X = graph.bb_indices.long()
    edge_index = adj_matrix.nonzero().t()

    rxn_indices_matrix = graph.rxn_indices
    edge_attr = rxn_indices_matrix[edge_index[0], edge_index[1]].long()

    smiles, mol = mfg.to_smiles(), mfg.to_mol()
    if is_valid_smiles(mol) and not "." in smiles:
        return Data(
            x=X.tolist(),
            edge_index=edge_index.tolist(),
            edge_attr=edge_attr.tolist(),
            smiles=smiles,
        )
    return None


def _molecule_worker(
    q: Queue,
    stop_event: Event,
    length: List[int],
    r1_lookup: Dict[Tuple[int, int], List[int]],
    r2_lookup: Dict[int, Tuple[List[int], List[int]]],
    bb_descriptors: List[dict],
    n_building_blocks: int,
    sample_by_inverse_molwt: bool,
    molwt_temperature: float,
    timeout: float,
    worker_seed: int,
):
    """Worker process that generates molecules and pushes results to a queue."""
    import syncogen.constants.constants as constants
    from syncogen.api.rdkit.assembly import RDKitMoleculeAssembly
    from syncogen.utils.rdkit import is_valid_smiles
    from syncogen.api.graph.graph import BBRxnGraph

    torch.set_num_threads(1)
    seed_counter = 0
    while not stop_event.is_set():
        result = _generate_one_molecule(
            length=length,
            r1_lookup=r1_lookup,
            r2_lookup=r2_lookup,
            bb_descriptors=bb_descriptors,
            n_building_blocks=n_building_blocks,
            sample_by_inverse_molwt=sample_by_inverse_molwt,
            molwt_temperature=molwt_temperature,
            timeout=timeout,
            worker_seed=worker_seed + seed_counter,
        )
        seed_counter += 1
        if result is not None:
            q.put(result)
    q.put(None)


def sample_random_molecules(
    n: int,
    length: List[int],
    comp_r1: torch.Tensor,
    comp_r2: torch.Tensor,
    seed: int = None,
    sample_by_inverse_molwt: bool = False,
    molwt_temperature: float = 1.0,
    timeout: float = None,
    num_workers: int = None,
) -> List[Data]:
    """Sample unique random molecules using multiprocessing."""
    if num_workers is None:
        num_workers = os.cpu_count()

    base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    seed_stride = 10_000_000

    import syncogen.constants.constants as constants
    from syncogen.utils.rdkit import get_lipinski_descriptors

    bb_descriptors = get_lipinski_descriptors(
        [Chem.MolFromSmiles(bb_smiles) for bb_smiles in constants.BUILDING_BLOCKS]
    )
    n_building_blocks = constants.N_BUILDING_BLOCKS

    print("Precomputing compatibility lookups...", flush=True)
    r1_lookup, r2_lookup = precompute_compatibility_lookups(comp_r1, comp_r2)
    print(
        f"Done. r1_lookup: {len(r1_lookup)} entries, r2_lookup: {len(r2_lookup)} entries",
        flush=True,
    )

    q = Queue()
    stop_event = Event()

    workers = []
    for i in range(num_workers):
        p = Process(
            target=_molecule_worker,
            args=(
                q,
                stop_event,
                length,
                r1_lookup,
                r2_lookup,
                bb_descriptors,
                n_building_blocks,
                sample_by_inverse_molwt,
                molwt_temperature,
                timeout,
                base_seed + i * seed_stride,
            ),
        )
        p.start()
        workers.append(p)

    # Consumer: deduplicate and collect in main process
    graph_data = []
    seen_smiles = set()
    pbar = tqdm(total=n)

    sentinels_received = 0
    while len(graph_data) < n:
        item = q.get()
        if item is None:
            sentinels_received += 1
            if sentinels_received == num_workers:
                break
            continue
        smiles = item.smiles
        if smiles not in seen_smiles:
            seen_smiles.add(smiles)
            data = Data(
                x=torch.tensor(item.x),
                edge_index=torch.tensor(item.edge_index),
                edge_attr=torch.tensor(item.edge_attr),
                smiles=smiles,
            )
            data.data_index = len(graph_data)
            graph_data.append(data)
            pbar.update(1)

    stop_event.set()
    pbar.close()

    # Kill straggling workers
    deadline = time.time() + 10
    while sentinels_received < num_workers and time.time() < deadline:
        try:
            item = q.get(timeout=1)
            if item is None:
                sentinels_received += 1
        except queue.Empty:
            continue

    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    return graph_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample random molecules by building fragment graphs."
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=100,
        help="Number of molecules to sample",
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
    parser.add_argument(
        "--save_graphs", action="store_true", help="Save output as numpy arrays"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/molecule_graphs/dataset_list_full.pt",
        help="Path to save the output .pt file",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
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
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds for generating a single molecule. If exceeded, the attempt is restarted. (default: None, no timeout)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: os.cpu_count())",
    )

    # Conformer options
    parser.add_argument(
        "--save_conformers", action="store_true", help="Save conformers"
    )
    parser.add_argument(
        "--num_conformers",
        type=int,
        default=50,
        help="Number of conformers to generate",
    )
    parser.add_argument(
        "--energy_cutoff", type=float, default=10.0, help="Energy cutoff (kcal/mol)"
    )
    parser.add_argument(
        "--rmsd_threshold",
        type=float,
        default=1.5,
        help="RMSD threshold for clustering",
    )
    parser.add_argument(
        "--conformer_dir",
        type=str,
        default="data/conformers",
        help="Directory to save conformers",
    )
    parser.add_argument(
        "--xtb",
        action="store_true",
        help="Run XTB optimization on molecules and save conformers",
    )

    # Path to vocabulary directory
    parser.add_argument(
        "--vocab_dir",
        type=str,
        required=True,
        help="Path to vocabulary directory containing building_blocks.json, reactions.json, compatibility.pt, etc.",
    )

    args = parser.parse_args()

    if args.save_graphs:
        output_path = Path(args.output_path)
        if output_path.suffix != ".pt":
            raise ValueError("output_path must be a .pt file")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_conformers:
        os.makedirs(args.conformer_dir, exist_ok=True)

    # Load vocabulary and update module-level constants
    load_vocabulary(Path(args.vocab_dir))
    import syncogen.constants.constants as constants
    from syncogen.api.rdkit.assembly import RDKitMoleculeAssembly
    from syncogen.utils.rdkit import is_valid_smiles
    from syncogen.api.graph.graph import BBRxnGraph
    from syncogen.utils.rdkit import get_lipinski_descriptors

    # Extract compatibility tensor and decode roles
    compatibility = constants.COMPATIBILITY.to(torch.int64)
    comp_r1 = (compatibility & 1) != 0
    comp_r2 = (compatibility & 2) != 0

    # Generate molecules
    molecules = sample_random_molecules(
        n=args.num_samples,
        length=args.length,
        comp_r1=comp_r1,
        comp_r2=comp_r2,
        seed=args.seed,
        sample_by_inverse_molwt=args.sample_by_inverse_molwt,
        molwt_temperature=args.molwt_temperature,
        timeout=args.timeout,
        num_workers=args.num_workers,
    )

    used_bbs = set()
    lipinski_stats = {"mw": [], "logp": [], "hbd": [], "hba": [], "qed": []}

    for molecule in molecules:
        bbs_mol = list(molecule.x.numpy())
        used_bbs.update(bbs_mol)

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

    print("\n=== Averaged Molecular Property Statistics ===")
    print(f"Total molecules: {len(molecules)}")
    if lipinski_stats["mw"]:
        print(
            f"Average Molecular Weight: {sum(lipinski_stats['mw'])/len(lipinski_stats['mw']):.2f}"
        )
        print(
            f"Average LogP: {sum(lipinski_stats['logp'])/len(lipinski_stats['logp']):.2f}"
        )
        print(
            f"Average H-Bond Donors: {sum(lipinski_stats['hbd'])/len(lipinski_stats['hbd']):.2f}"
        )
        print(
            f"Average H-Bond Acceptors: {sum(lipinski_stats['hba'])/len(lipinski_stats['hba']):.2f}"
        )
        print(
            f"Average QED Score: {sum(lipinski_stats['qed'])/len(lipinski_stats['qed']):.4f}"
        )

    if args.save_graphs:
        torch.save(molecules, args.output_path)
        print(f"Saved {len(molecules)} molecules to {args.output_path}")
    else:
        print(f"Generated {len(molecules)} molecules")
