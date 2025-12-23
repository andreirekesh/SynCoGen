<p align="center">
  <img src="syncogen.gif" width="400">
</p>

## SynCoGen
**Syn**thesizabile **CoGen**eration (**SynCoGen**) is a generative small molecule design framework comprised of simultaneous absorbing state discrete diffusion model and continuous flow matching. The discrete diffusion model is based on [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/pdf/2406.07524) for synthesizable molecular generation, based on [this repo](https://github.com/kuleshov-group/mdlm) and using the SUBS parametrization. The continuous flow matching model is a straightforward implementation of conditional flow matching over molecular coordinates. We provide a simple graph transformer as a backbone.

<p align="center">
  <img src="molmdgm.png" width="400">
</p>

Rather than operate on sequences, SynCoGen's discrete component operates on graphs. Masking is performed on both node (B x N x D) and edge (B x N x N x D) matrices.

### Chemistry
The purpose of SynCoGen is to generate molecules with valid synthesis paths. To this end, we define a "vocabulary" by a compatible set of building blocks and reaction templates from which to generate synthesizable molecules via combinatorial synthesis using RDKit. Conformers are then generated for these molecules using GFN-xTB. Upon validation end, we evaluate all sampled molecules by their validity according to RGFN's reaction rules.

### Graph Parametrization
Node and edge identities are encoded as onehot vectors. The dimensionality of the node onehot vector is simply the number of fragment types. The dimensionality of the edge onehot vector is (R x C^2) where R is the number of reaction types and C is the number of centers in the reaction. Both reaction type and center indices are encoded to define generated molecules are strictly as possible, such that a graph sampled by SynCoGen corresponds to a valid molecule with no structural ambiguity.

### Coordinate Parametrization
We use continuous flow matching to predict the coordinates of the atoms in the molecule. Ground-truth and predicted coordinates are given by a (B x N x MAX_ATOMS x 3) tensor, where MAX_ATOMS is the maximum number of atoms in any of the fragments in the vocabulary. During loss calculation, we only consider the atoms in the molecule, and ignore the dummy atoms used to pad the fragments as well as the atoms that are dropped during reactions. During sampling, we first reassemble the fragments into an RDKit molecule using the predicted graph, and then use the number of atoms attributed to each fragment to determine a final molecule mask. The remaining coordinates are assigned to the molecule and an SDF file is generated.

### Data Generation
See vocabulary/README.md.

The full SynSpace dataset is available [here](https://tyers.s3.us-west-1.amazonaws.com/all_steps_clean.tgz). For SynSpace, download all_steps_clean.tgz. For pharmacophores, download pharmacophores.lmdb. An LMDB version of SynSpace conformers will be uploaded soon.

### Directory Structure
```
rewrite/
├─ train.py
├─ configs/
├─ syncogen/
│  ├─ api/
│  │  ├─ atomics/
│  │  ├─ graph/
│  │  ├─ ops/
│  │  ├─ rdkit/
│  │  └─ (molecule.py, pharmacophores.py)
│  ├─ constants/
│  ├─ data/
│  ├─ diffusion/
│  │  ├─ interpolation/
│  │  ├─ loss/
│  │  ├─ noise/
│  │  ├─ sampling/
│  │  └─ training/
│  ├─ logging/
│  │  ├─ loggers/
│  │  └─ metrics/
│  ├─ models/
│  └─ utils/
└─ vocabulary/
```

SynCoGen uses Gin configs, housed in `configs/`. Defaults can be found in `.gin` files corresponding to `@gin.configurable`-decorated classes. 

### Getting Started

Create the environment with conda:
```bash
module load cuda/12.4.0   # on cluster
conda env create -f requirements.yaml
conda activate syncogen
```

### Training
To start a training run on SynSpace with pharmacophore conditioning, run:
```bash
python train.py --config configs/experiments/pharmacophore.gin --vocab_dir vocabulary/synspace
```

