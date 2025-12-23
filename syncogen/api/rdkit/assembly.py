from dataclasses import dataclass
from typing import List, Optional

import networkx as nx
from rdkit import Chem

from syncogen.constants.constants import (
    BUILDING_BLOCKS_SMI_TO_IDX,
    BUILDING_BLOCKS_IDX_TO_SMI,
    REACTIONS,
)
import copy

BOND_TYPE_MAP = {
    1.0: Chem.BondType.SINGLE,
    2.0: Chem.BondType.DOUBLE,
    3.0: Chem.BondType.TRIPLE,
    1.5: Chem.BondType.AROMATIC,
}


@dataclass
class FragmentInstance:
    """
    A class to represent an instance of a fragment in our fragment set.
    """

    def __init__(self, smiles: str, frag_order: int, global_frag_id: int):
        self.smiles = smiles
        self.frag_order = frag_order
        self.global_frag_id = global_frag_id
        self.atom_graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        mol = Chem.MolFromSmiles(self.smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {self.smiles}")

        # Ensure stereochemistry is properly assigned
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

        # Add atoms with stereochemistry
        for atom in mol.GetAtoms():
            unique_id = (atom.GetIdx(), self.frag_order)
            self.atom_graph.add_node(
                unique_id,
                frag_order=self.frag_order,
                global_frag_id=self.global_frag_id,
                symbol=atom.GetSymbol(),
                formal_charge=atom.GetFormalCharge(),
                is_aromatic=atom.GetIsAromatic(),
                num_explicit_hs=atom.GetNumExplicitHs(),
                chirality=atom.GetChiralTag().name,
                is_chiral_center=atom.HasProp("_ChiralityPossible"),
            )

        # Add bonds with stereochemistry
        for bond in mol.GetBonds():
            start_id = (bond.GetBeginAtomIdx(), self.frag_order)
            end_id = (bond.GetEndAtomIdx(), self.frag_order)
            self.atom_graph.add_edge(
                start_id,
                end_id,
                bond_order=bond.GetBondTypeAsDouble(),
                is_aromatic=bond.GetIsAromatic(),
            )


@dataclass
class RDKitMoleculeAssembly:
    """
    A class to represent a molecule assembly. Has both an underlying fragment graph and an underlying atom graph.
    Fragment node ids are just the order in which they are added.
    Each fragment knows about its reaction center occupation statuses.
    """

    def __init__(self, fragment_graph: nx.Graph = None, atom_graph: nx.Graph = None):
        self.fragment_graph = fragment_graph if fragment_graph else nx.Graph()
        self.atom_graph = atom_graph if atom_graph else nx.Graph()

    def num_fragments(self):
        return len(self.fragment_graph.nodes)

    def frag_idx_to_atom_graph(self, frag_order: int):
        return self.fragment_graph.nodes[frag_order]["node"]

    def add_node(
        self,
        node: FragmentInstance,
        global_frag_id: int,
        rxn_center_available: List[bool],
    ):
        node_id = self.num_fragments()
        self.fragment_graph.add_node(
            node_id,
            node=node,
            global_frag_id=global_frag_id,
            rxn_center_available=rxn_center_available,
        )

    def add_edge(self, node_a: FragmentInstance, node_b: FragmentInstance):
        node_a_id = node_a.frag_order
        node_b_id = node_b.frag_order
        self.fragment_graph.add_edge(node_a_id, node_b_id)

    def add_fragment(
        self,
        new_frag_global_id: int,
        reaction_id: Optional[int] = None,
        self_frag_order: Optional[int] = None,
        self_reaction_center_idx: Optional[int] = None,
        other_reaction_center_idx: Optional[int] = None,
    ):
        """
        Add a new fragment to the molecule fragment graph.

        Args:
            new_frag_global_id (int): The global id of the new fragment as indexed by our vocabulary.
            reaction_id (int): The reaction id of the reaction used to add the new fragment.
            self_frag_order (int): The fragment order in order of addition (0, 1, 2, 3) of the fragment in our graph that the new fragment is being added to.
            self_reaction_center_idx (int): The index of the reaction center we want to occupy on the fragment in our graph.
            other_reaction_center_idx (int): The index of the reaction center we want to occupy on the other fragment.

        Returns:
            The updated fragment and atom graphs.
        """
        smiles = BUILDING_BLOCKS_IDX_TO_SMI[new_frag_global_id]["smiles"]
        new_frag_order = self.num_fragments()
        new_fragment_instance = FragmentInstance(smiles, new_frag_order, new_frag_global_id)
        new_rxn_center_available = [True] * len(BUILDING_BLOCKS_SMI_TO_IDX[smiles]["centers"])

        # If this is the first fragment, we're done
        if self.num_fragments() == 0:
            assert reaction_id is None
            self.add_node(
                node=new_fragment_instance,
                global_frag_id=new_frag_global_id,
                rxn_center_available=new_rxn_center_available,
            )
            self.atom_graph = new_fragment_instance.atom_graph
            return self.fragment_graph, self.atom_graph

        # Build idx->SMARTS mapping from constants (once per process)
        if not hasattr(RDKitMoleculeAssembly, "_REACTION_IDX_TO_SMARTS"):
            RDKitMoleculeAssembly._REACTION_IDX_TO_SMARTS = {
                v["index"]: k for k, v in REACTIONS.items()
            }
        reaction = RDKitMoleculeAssembly._REACTION_IDX_TO_SMARTS[reaction_id]

        # Check if the reaction center we picked is already used
        if (
            self.fragment_graph.nodes[self_frag_order]["rxn_center_available"][
                self_reaction_center_idx
            ]
            == False
        ):
            raise ValueError("Reaction center already used")

        # Update the reaction center availability
        self.fragment_graph.nodes[self_frag_order]["rxn_center_available"][
            self_reaction_center_idx
        ] = False
        new_rxn_center_available[other_reaction_center_idx] = False

        # Add the new fragment to the fragment graph
        self.add_node(
            node=new_fragment_instance,
            global_frag_id=new_frag_global_id,
            rxn_center_available=new_rxn_center_available,
        )

        # Add an edge on the fragment graph level
        self.fragment_graph.add_edge(self_frag_order, new_frag_order)

        # Update the atom graph as well
        self.update_atom_graph(
            new_fragment_instance,
            reaction,
            self_frag_order,
            new_frag_order,
            self_reaction_center_idx,
            other_reaction_center_idx,
        )

        return self.fragment_graph, self.atom_graph

    def update_atom_graph(
        self,
        new_fragment_instance: FragmentInstance,
        reaction: str,
        self_frag_order: int,
        new_frag_order: int,
        self_reaction_center_idx: int,
        other_reaction_center_idx: int,
    ):
        """
        Update the atom graph to add a new fragment.
        """
        # What are the atom indices we're connecting with our reaction?
        self_fragment_instance = self.frag_idx_to_atom_graph(self_frag_order)

        # get attachment point atom indices
        rc_self = BUILDING_BLOCKS_SMI_TO_IDX[self_fragment_instance.smiles]["centers"][
            self_reaction_center_idx
        ]
        rc_other = BUILDING_BLOCKS_SMI_TO_IDX[new_fragment_instance.smiles]["centers"][
            other_reaction_center_idx
        ]

        # convert attachment point atom indices to node ids
        attachpt_a = (rc_self, self_frag_order)
        attachpt_b = (rc_other, new_frag_order)

        # do we drop an atom on either side?
        leaving_group_self, leaving_group_other = REACTIONS[reaction]["drop_bools"]

        # add the new fragment atoms to the atom graph
        self.atom_graph = nx.compose(self.atom_graph, new_fragment_instance.atom_graph)

        if leaving_group_self:
            old_attachpt_a = attachpt_a
            attachpt_a = next(self.atom_graph.neighbors(attachpt_a))
            self.atom_graph.remove_node(old_attachpt_a)

        if leaving_group_other:
            old_attachpt_b = attachpt_b
            attachpt_b = next(self.atom_graph.neighbors(attachpt_b))
            self.atom_graph.remove_node(old_attachpt_b)

        self.atom_graph.add_edge(attachpt_a, attachpt_b, bond_order=1.0)

    def to_smiles(self):
        """Converts the atom graph representation back to a SMILES string."""
        mol = self.to_mol()
        smiles = Chem.MolToSmiles(mol)
        return smiles

    def to_mol(self, sanitize: bool = True):
        """Converts the atom graph representation back to an RDKit molecule.
        Reconstructs an RDKit molecule by mapping graph nodes to atoms while preserving
        stereochemistry and formal charges."""

        # Create empty RDKit molecule and map graph nodes to atom indices
        mol = Chem.RWMol()
        node_to_idx = {}
        for node, data in self.atom_graph.nodes(data=True):
            atom_idx = mol.AddAtom(Chem.Atom(data["symbol"]))
            atom = mol.GetAtomWithIdx(atom_idx)
            atom.SetIsAromatic(data.get("is_aromatic", False))
            atom.SetIntProp("frag_order", data["frag_order"])
            atom.SetIntProp("global_frag_id", data["global_frag_id"])
            atom.SetFormalCharge(data.get("formal_charge"))
            atom.SetNumExplicitHs(data.get("num_explicit_hs", 0))
            # Set chirality from graph data
            if data.get("is_chiral_center"):
                atom.SetChiralTag(getattr(Chem.ChiralType, data["chirality"]))
            node_to_idx[node] = atom_idx

        # Add bonds between atoms, preserving bond orders and stereochemistry
        for u, v, data in self.atom_graph.edges(data=True):
            bond_type = data.get("bond_order")
            mol.AddBond(node_to_idx[u], node_to_idx[v], BOND_TYPE_MAP[bond_type])
            bond = mol.GetBondBetweenAtoms(node_to_idx[u], node_to_idx[v])

            # Set bond stereochemistry from graph data
            if data.get("bond_stereo"):
                bond.SetStereo(getattr(Chem.BondStereo, data["bond_stereo"]))

        # Validate molecular structure while preserving aromaticity
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except:
                print("Warning: Molecule could not be sanitized")

        return mol

    def all_open_reaction_centers(self):
        """
        Returns a list of all open reaction centers in the molecule fragment graph.
        """
        return [
            (frag_id, idx)
            for frag_id in self.fragment_graph.nodes
            for idx, available in enumerate(
                self.fragment_graph.nodes[frag_id]["rxn_center_available"]
            )
            if available
        ]

    def copy(self):
        """
        Return a deep copy of the molecule fragment graph, including both
        the fragment graph and atom graph.
        """
        return RDKitMoleculeAssembly(
            fragment_graph=copy.deepcopy(self.fragment_graph),
            atom_graph=copy.deepcopy(self.atom_graph),
        )
