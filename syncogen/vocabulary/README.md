# Vocabulary Preprocessing

The `preprocess.py` script processes building blocks and reaction templates to create a vocabulary for SynCoGen. It performs annotation, compatibility checking, filtering, and feature computation.

## Usage

```bash
python preprocess.py --bb_input <path> --rxn_input <path> --output_dir <path> [options]
```

## Required Arguments

### `--bb_input`
- **Type**: `str`
- **Required**: Yes
- **Description**: Path to the building blocks input file
- **Formats**: 
  - Text file (`.txt` or `.smi`): One SMILES string per line
  - CSV file (`.csv`): Requires `--bb_smiles_column` to specify the column containing SMILES
- **Example**: `--bb_input building_blocks.txt` or `--bb_input building_blocks.csv`

### `--rxn_input`
- **Type**: `str`
- **Required**: Yes
- **Description**: Path to the reaction templates input file
- **Formats**:
  - Text file (`.txt`): One SMARTS reaction template per line
  - CSV file (`.csv`): Requires `--rxn_smarts_column` to specify the column containing SMARTS
- **Example**: `--rxn_input reactions.txt` or `--rxn_input reactions.csv`

### `--output_dir`
- **Type**: `str`
- **Required**: Yes
- **Description**: Directory where all output files will be written
- **Note**: The directory will be created if it doesn't exist
- **Example**: `--output_dir ./vocab_output`

## Optional Arguments

### `--bb_smiles_column`
- **Type**: `str`
- **Default**: `None`
- **Description**: Column name in CSV building-block file containing SMILES strings
- **Required when**: `--bb_input` is a CSV file
- **Example**: `--bb_smiles_column smiles` or `--bb_smiles_column SMILES`

### `--rxn_smarts_column`
- **Type**: `str`
- **Default**: `None`
- **Description**: Column name in CSV reaction file containing SMARTS reaction templates
- **Required when**: `--rxn_input` is a CSV file
- **Example**: `--rxn_smarts_column reaction_smarts` or `--rxn_smarts_column SMARTS`

### `--keep_incompatible`
- **Type**: `flag` (no value)
- **Default**: `False`
- **Description**: If set, keeps building blocks and reactions that have no compatible reaction centers. By default, incompatible items are filtered out.
- **Example**: `--keep_incompatible`

### `--max_centers_per_bb`
- **Type**: `int`
- **Default**: `3`
- **Description**: Maximum number of reaction centers per building block stored in the compatibility tensor. Building blocks with more centers will have the extra centers ignored (with a warning).
- **Example**: `--max_centers_per_bb 5`

### `--diverse_subset`
- **Type**: `int`
- **Default**: `None`
- **Description**: If specified, uses MaxMinPicker to select this many diverse building blocks from the property-filtered BBs before compatibility computation. This helps reduce the vocabulary size while maintaining chemical diversity.
- **Note**: Only applied if the value is less than the number of filtered building blocks
- **Example**: `--diverse_subset 1000`

### `--max_atoms_per_bb_filter`
- **Type**: `int`
- **Default**: `16`
- **Description**: Maximum number of atoms allowed per building block. Building blocks exceeding this limit are filtered out during property-based filtering.
- **Note**: Only applied if `--no_property_filtering` is not set
- **Example**: `--max_atoms_per_bb_filter 20`

### `--allowed_atom_types`
- **Type**: `str`
- **Default**: `"C,N,O,B,F,Cl,Br,S"`
- **Description**: Comma-separated list of allowed atom symbols for building blocks. Building blocks containing other atom types are dropped.
- **Note**: Only applied if `--no_property_filtering` is not set
- **Example**: `--allowed_atom_types "C,N,O,F,Cl"`

### `--n_threads`
- **Type**: `int`
- **Default**: `8`
- **Description**: Number of threads to use for fingerprint generation during diverse picking (when `--diverse_subset` is specified).
- **Example**: `--n_threads 16`

### `--no_property_filtering`
- **Type**: `flag` (no value)
- **Default**: `False`
- **Description**: If set, skips property-based filtering of building blocks. Property filtering includes:
  - Removal of disjoint fragments (SMILES containing '.')
  - Sanitization and canonicalization
  - Deduplication
  - Atom count filtering (`--max_atoms_per_bb_filter`)
  - Atom type filtering (`--allowed_atom_types`)
- **Example**: `--no_property_filtering`

### `--no_substructure_filtering`
- **Type**: `flag` (no value)
- **Default**: `False`
- **Description**: If set, skips unwanted substructure filtering. By default, building blocks containing PAINS (A, B, C), Brenk, or NIH unwanted substructures are filtered out.
- **Example**: `--no_substructure_filtering`

## Processing Pipeline

The script performs the following steps in order:

1. **Reaction Annotation**: Annotates reaction templates with reaction center information
2. **Building Block Loading**: Loads building blocks from the input file
3. **Property Filtering** (optional): Filters BBs by properties (atom count, atom types, sanitization)
4. **Substructure Filtering** (optional): Removes BBs with unwanted substructures (PAINS, Brenk, NIH)
5. **Diverse Subset Selection** (optional): Selects a diverse subset of BBs using MaxMinPicker
6. **Compatibility Computation**: Builds compatibility tensor between BBs and reactions
7. **BB Filtering**: Removes BBs without reaction centers (unless `--keep_incompatible`)
8. **Reaction Filtering**: Removes reactions incompatible with all BBs (unless `--keep_incompatible`)
9. **Feature Computation**: Computes fragment-level features (MACCS fingerprints, atom features, bond features)
10. **Output Writing**: Writes all output files

## Output Files

The script generates the following files in `--output_dir`:

- **`building_blocks.json`**: Mapping of SMILES to reaction center indices
- **`reactions.json`**: Mapping of SMARTS to reaction metadata (drop_bools, index)
- **`compatibility.pt`**: PyTorch tensor of shape `(n_bbs, n_rxns, max_centers)` indicating compatibility
- **`fragment_features.pt`**: PyTorch tensor containing fragment-level features (MACCS, atom features, bond features, adjacency)
- **`smiles_annotated.json`**: Building block annotations (reaction center indices)
- **`templates_annotated.json`**: Reaction template annotations (reaction center information)
- **`meta.json`**: Metadata including counts, dimensions, atom types, etc.

## Example

```bash
python preprocess.py \
  --bb_input data/building_blocks.csv \
  --bb_smiles_column SMILES \
  --rxn_input data/reactions.txt \
  --output_dir vocab_output \
  --max_centers_per_bb 5 \
  --max_atoms_per_bb_filter 20 \
  --allowed_atom_types "C,N,O,F,Cl,Br" \
  --diverse_subset 5000
```

