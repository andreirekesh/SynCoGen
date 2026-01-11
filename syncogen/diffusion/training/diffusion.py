from typing import Optional, Sequence, Dict

import itertools

import gin
import lightning as L
import torch

from syncogen.api.atomics.coordinates import Coordinates
from syncogen.api.atomics.pharmacophores import ShepherdPharmacophores
from syncogen.diffusion.training.optimizers.optimizer import Optimizer
from syncogen.diffusion.training.optimizers.lr_scheduler import LRScheduler
from syncogen.api.graph.graph import BBRxnGraph
from syncogen.constants.constants import COORDS_STD, N_REACTIONS, N_CENTERS
from syncogen.data.dataloader import (
    SyncogenDataManager,
    RandomFaultTolerantSampler,
    FaultTolerantDistributedSampler,
)
from torch_geometric.loader import DataLoader
from syncogen.data.utils import (
    to_dense_graph,
    to_dense_coords,
    to_dense_pharmacophores,
    to_dense_bonds,
)
from syncogen.diffusion.interpolation import InterpolatorBase, LinearInterpolator
from syncogen.diffusion.loss import LossBase, LossList
from syncogen.diffusion.noise import LogLinearNoise, NoiseBase
from syncogen.diffusion.sampling.discrete_strategies import DiscreteStrategyBase, MDLM, PathPlanning
from syncogen.diffusion.sampling.integrators import IntegratorBase, EulerIntegrator
from syncogen.logging.metrics import (
    MetricsBase,
    MetricsList,
    log_molecule_images,
    log_fragment_histogram,
)
from syncogen.utils.rdkit import build_molecules_from_graphs
from syncogen.models.semla_pharm import SemlaPharmGenerator
from syncogen.models.semla import SemlaGenerator
import torch_geometric


def get_backbone(backbone: str, self_conditioning: bool, pharm_subset: int):
    if backbone == "semla":
        return SemlaGenerator(self_conditioning=self_conditioning)
    elif backbone == "semla_pharm":
        return SemlaPharmGenerator(
            self_conditioning=self_conditioning,
            pharmacophore_subset=pharm_subset,
        )
    raise ValueError(f"Unknown backbone: {backbone}")


@gin.configurable
class Diffusion(L.LightningModule):
    def __init__(
        self,
        data_manager: SyncogenDataManager,
        losses: Sequence[LossBase] = (),
        metrics: Sequence[MetricsBase] = (),
        device: Optional[torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        augmentations: Optional[list[str]] = ["center", "normalize", "random_rotate"],
        normalization_scale: Optional[float] = 1.0 / COORDS_STD,
        discrete_noise: Optional[NoiseBase] = LogLinearNoise(),
        interpolator: Optional[InterpolatorBase] = LinearInterpolator(),
        discrete_strategy: Optional[DiscreteStrategyBase] = MDLM(),
        integrator: Optional[IntegratorBase] = EulerIntegrator(),
        train_rot_align: bool = False,
        self_conditioning: bool = False,
        time_conditioning: bool = True,
        backbone: str = "semla_pharm",
        use_compat: bool = True,
        sampling_eps: float = 1e-3,
        importance_sampling: bool = False,
        antithetic_sampling: bool = True,
        sampling_noise_removal: bool = True,
        generate_eval_samples: bool = True,
        sample_every_n_epochs: int = 1,
        num_sample_steps: int = 100,
        optimizer: Optimizer = None,
        lr_scheduler: LRScheduler = None,
        ema_decay: float = 0.0,
        pharm_subset: int = 7,
        scale_noise: bool = False,
        scale_noise_factor: float = 0.2,
        num_fragments_probs: Optional[Dict[int, float]] = None,
    ):
        super().__init__()
        self.default_device = device
        self.data_manager = data_manager
        self.augmentations = augmentations
        self.normalization_scale = normalization_scale
        self.discrete_noise = discrete_noise
        self.interpolator = interpolator
        self.discrete_strategy = discrete_strategy
        self.integrator = integrator
        self.train_rot_align = train_rot_align
        self.self_conditioning = self_conditioning
        self.time_conditioning = time_conditioning
        self.backbone = get_backbone(backbone, self_conditioning, pharm_subset)
        self.losses = LossList(losses)
        self.metrics = MetricsList(metrics)
        self.neg_infinity = -25000.0
        self.use_compat = use_compat
        self.sampling_eps = sampling_eps
        self.importance_sampling = importance_sampling
        self.antithetic_sampling = antithetic_sampling
        self.sampling_noise_removal = sampling_noise_removal
        self.generate_eval_samples = generate_eval_samples
        self.sample_every_n_epochs = sample_every_n_epochs
        self.num_sample_steps = num_sample_steps
        self.scale_noise = scale_noise
        self.scale_noise_factor = scale_noise_factor

        self._optimizer_config = optimizer
        self._lr_scheduler_config = lr_scheduler

        # Store pharm_subset for use in training
        self.pharm_subset = pharm_subset

        # Override fragment number probabilities if provided
        if num_fragments_probs is not None:
            fragments = list(num_fragments_probs.keys())
            probs = list(num_fragments_probs.values())

            # Validate probabilities are between 0 and 1
            if not all(0 <= p <= 1 for p in probs):
                raise ValueError("All probabilities in num_fragments_probs must be between 0 and 1")

            # Normalize probabilities to sum to 1
            probs_tensor = torch.tensor(probs, dtype=torch.float)
            if probs_tensor.sum() == 0:
                raise ValueError("All probabilities in num_fragments_probs are zero")
            probs_tensor = probs_tensor / probs_tensor.sum()

            # Set dataloader probabilities directly
            data_manager.train_length_values = torch.tensor(fragments, dtype=torch.long)
            data_manager.train_length_probs = probs_tensor

        # Set the discrete noise for the discrete strategy
        self.discrete_strategy.discrete_noise = self.discrete_noise

        # EMA setup
        self.ema = None
        if ema_decay > 0:
            from syncogen.models.ema import ExponentialMovingAverage

            self.ema = ExponentialMovingAverage(
                itertools.chain(
                    self.backbone.parameters(),
                    self.discrete_noise.parameters(),
                ),
                decay=ema_decay,
            )

        # Fault-tolerant checkpoint resumption
        self.fast_forward_epochs = None
        self.fast_forward_batches = None

        # Edge feature dimension (reactions * centers * centers + 2 for no-edge and mask tokens)
        self.n_edge_features = N_REACTIONS * N_CENTERS * N_CENTERS + 2

    @property
    def batch_size(self) -> int:
        """Training batch size."""
        return self.data_manager.batch_size

    @property
    def eval_batch_size(self) -> int:
        """Evaluation/validation batch size."""
        return self.data_manager.eval_batch_size

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def _apply_scale_noise(self, coords: Coordinates) -> Coordinates:
        """Scale coordinates by log(num_atoms) * scale_noise_factor."""
        atom_mask = coords.atom_mask.tensor
        coords_tensor = coords.atom_coords
        mask_flat = (
            atom_mask.reshape(atom_mask.shape[0], -1)
            if coords.is_batched
            else atom_mask.reshape(1, -1)
        )
        num_atoms = mask_flat.sum(dim=-1)
        scale = torch.log(num_atoms + 1e-8).view(-1, 1, 1) * self.scale_noise_factor
        coords.set_coordinates(coords_tensor * scale)
        return coords

    def _compute_step_loss(self, batch, prefix: str = "train"):
        """Helper function to compute loss for both training and validation steps.

        Args:
            batch: Data batch from dataloader
            prefix: Either "train" or "val" - used to prefix metric names for WandB grouping
        """
        # STEP 1: Assemble dense data from dataloader batch
        # 1.1: Dense graph data
        X_dense, E_dense, node_padding_mask, edge_padding_mask = to_dense_graph(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
            max_num_nodes=self.data_manager.max_bbs,
        )

        # 1.2: Dense coords data
        C_dense, coords_mask_dense = to_dense_coords(
            coordinates=batch.coordinates,
            coords_mask=batch.coords_mask,
            batch=batch.batch,
            max_num_nodes=self.data_manager.max_bbs,
        )

        # 1.3 Build graph object
        ground_truth_graph = BBRxnGraph.from_onehot(X_dense, E_dense)
        ground_truth_graph.apply_edge_givens()  # Enforce: diagonals=NO-EDGE, padding=zeros
        ground_truth_graph_mask = ground_truth_graph.ground_truth_atom_mask.tensor

        # 1.4: Build coords object
        ground_truth_coords = Coordinates(coordinates=C_dense, atom_mask=ground_truth_graph_mask)
        ground_truth_coords = ground_truth_coords.reshape_to_atoms()

        # 1.5: Append dense bonds data
        if self.data_manager.load_bonds:
            bonds_dense, bonds_padding_mask = to_dense_bonds(
                bonds=batch.bonds,
                bonds_len=batch.bonds_len,
                max_num_bonds=C_dense.shape[1] * C_dense.shape[2],
            )
        else:
            bonds_dense = bonds_padding_mask = None

        # 1.6: Append dense pharmacophores data + ShepherdPharmacophores attach logic
        if self.data_manager.load_pharmacophores:
            from syncogen.constants.constants import MAX_PHARM

            pharm_types_dense, pharm_pos_dense, pharm_mask_dense = to_dense_pharmacophores(
                pharm_types=batch.pharm_types,
                pharm_pos=batch.pharm_pos,
                pharm_len=batch.pharm_len,
                max_num_pharm=MAX_PHARM,
            )
            pharms = ShepherdPharmacophores(
                pharm_coords=pharm_pos_dense,
                pharm_types=pharm_types_dense,
                padding_mask=pharm_mask_dense,
                n_subset=self.pharm_subset,
            )
            ground_truth_coords.attach_pharmacophores(
                pharm_coords=pharms.coords,
                pharm_padding_mask=pharms.padding_mask.to(dtype=ground_truth_coords.tensor.dtype),
            )
        else:
            pharm_types_dense = pharm_pos_dense = pharm_mask_dense = None
            pharms = None

        # STEP 2: Apply augmentations to the training batch
        for augmentation in self.augmentations:
            if augmentation == "center":
                ground_truth_coords.center()
            elif augmentation == "normalize":
                ground_truth_coords.scale(self.normalization_scale)
            elif augmentation == "random_rotate":
                ground_truth_coords.random_rotate()
            elif augmentation == "translate":
                ground_truth_coords.random_translate()

        # STEP 3: Noise the training batch
        # 3.1: Sample a noise level t
        current_batch_size = batch.num_graphs
        t = self._sample_t(current_batch_size, self.default_device)

        # 3.2: Noise the training batch
        discrete_sigma, discrete_dsigma = self.discrete_noise(t)
        graph_noised = self._noise_discrete(ground_truth_graph, discrete_sigma)
        graph_noised.apply_edge_givens()  # Enforce: diagonals=NO-EDGE, padding=zeros
        partially_noised_graph_mask = graph_noised.partial_atom_mask.tensor
        atom_mask = graph_noised.partial_atom_mask.tensor

        # 3.3 Noise the coordinates
        C0_shifted, coords_noised = self._noise_continuous(
            ground_truth_coords,
            ground_truth_graph_mask,
            partially_noised_graph_mask,
            t,
            prefix=prefix,
        )
        if bonds_dense is not None:
            C0_shifted.attach_bonds(
                bonds=bonds_dense,
                bonds_mask=bonds_padding_mask.to(dtype=C0_shifted.atom_coords.dtype),
            )

        # STEP 4: Forward pass through the backbone
        Xt, Et, Ct = graph_noised.bb_onehot, graph_noised.rxn_onehot, coords_noised.atom_coords
        cond = (
            (pharms.types_onehot, C0_shifted.pharmacophores, pharms.padding_mask)
            if pharms is not None and C0_shifted.has_pharmacophores
            else None
        )
        logits_X, logits_E, C0_hat = self.forward(Xt, Et, Ct, atom_mask, cond=cond)

        # STEP 5: SUBS Parameterization
        logits_X_subs, logits_E_subs = self._subs_parameterization(graph_noised, logits_X, logits_E)

        # STEP 6: Compute the loss
        # Prepare log probs for graph losses
        gt_bb_indices = ground_truth_graph.bb_indices
        log_p_theta_X = torch.gather(
            logits_X_subs, dim=-1, index=gt_bb_indices[:, :, None]
        ).squeeze(-1)

        gt_rxn_indices = (
            ground_truth_graph.rxn_indices
        )  # Diagonals already NO-EDGE from apply_edge_givens
        log_p_theta_E = torch.gather(
            logits_E_subs, dim=-1, index=gt_rxn_indices[:, :, :, None]
        ).squeeze(-1)

        sigma_factor = discrete_dsigma / torch.expm1(discrete_sigma)

        # Compute losses via LossList
        coords_pred = Coordinates(coordinates=C0_hat, atom_mask=ground_truth_graph_mask)
        coords_pred.center()

        graph_losses = self.losses.compute_graph(
            log_p_theta_X, log_p_theta_E, ground_truth_graph.node_mask, sigma_factor
        )
        coords_losses = self.losses.compute_coords(coords_pred, C0_shifted, t)

        # Combine and compute total
        raw_losses = {**graph_losses, **coords_losses}
        total_loss = graph_losses["graph_total"] + coords_losses["coords_total"]
        raw_losses["total"] = total_loss

        # Add prefix for WandB grouping (train/ or val/)
        all_losses = {f"{prefix}/{k}": v for k, v in raw_losses.items()}

        print(
            f'MSE: {raw_losses.get("MSELoss", float("nan")):.4f} | '
            f'Bond: {raw_losses.get("BondLengthLoss", float("nan")):.4f} | '
            f'Pairwise: {raw_losses.get("PairwiseDistanceLoss", float("nan")):.4f} | '
            f'LDDT: {raw_losses.get("SmoothLDDTLoss", float("nan")):.4f}'
        )
        return all_losses, total_loss

    def training_step(self, batch, batch_idx):
        all_losses, total_loss = self._compute_step_loss(batch, prefix="train")
        self.log_dict(all_losses, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        all_losses, total_loss = self._compute_step_loss(batch, prefix="val")
        self.log_dict(all_losses, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def get_pharm_cond(self, batch):
        """Extract and augment pharmacophore conditioning from a batch."""
        from syncogen.constants.constants import MAX_PHARM

        # Densify coordinates and pharmacophores
        C_dense, coords_mask_dense = to_dense_coords(
            coordinates=batch.coordinates,
            coords_mask=batch.coords_mask,
            batch=batch.batch,
            max_num_nodes=self.data_manager.max_bbs,
        )
        pharm_types_dense, pharm_pos_dense, pharm_mask_dense = to_dense_pharmacophores(
            pharm_types=batch.pharm_types,
            pharm_pos=batch.pharm_pos,
            pharm_len=batch.pharm_len,
            max_num_pharm=MAX_PHARM,
        )

        # Subset pharmacophores to 7
        pharms = ShepherdPharmacophores(
            pharm_coords=pharm_pos_dense,
            pharm_types=pharm_types_dense,
            padding_mask=pharm_mask_dense,
            n_subset=self.pharm_subset,
        )

        # Create coords object with pharmacophores attached for joint augmentation
        coords = Coordinates(
            coordinates=C_dense.reshape(C_dense.shape[0], -1, 3),
            atom_mask=coords_mask_dense.reshape(coords_mask_dense.shape[0], -1),
        )
        coords.attach_pharmacophores(
            pharm_coords=pharms.coords, pharm_padding_mask=pharms.padding_mask
        )

        # Apply augmentations (center, normalize)
        for aug in self.augmentations:
            if aug == "center":
                coords.center()
            elif aug == "normalize":
                coords.scale(self.normalization_scale)

        return (
            pharms.types_onehot.to(self.device),
            coords.pharmacophores.to(self.device),
            pharms.padding_mask.to(self.device),
        )

    def _self_conditioning_train(self, Xt, Et, Ct, atom_mask, cond):
        """Returns the self-conditioned graph, edge, and coordinate tensors.
        Half the time, the self-conditioned tensors are all zeros."""
        Xt_self_cond = torch.zeros_like(Xt)
        Et_self_cond = torch.zeros_like(Et)
        Ct_self_cond = torch.zeros_like(Ct)

        if torch.rand(1).item() < 0.5:
            if cond is not None:  # if pharmacophores are present, use them for self-conditioning
                pharm_types, pharm_pos, pharm_padding_mask = cond
                Xt_self_cond, Et_self_cond, Ct_self_cond = self.backbone(
                    torch.cat([Xt, Xt_self_cond], dim=-1),
                    torch.cat([Et, Et_self_cond], dim=-1),
                    torch.cat([Ct, Ct_self_cond], dim=-1),
                    atom_mask=atom_mask,
                    pharm_types=pharm_types,
                    pharm_pos=pharm_pos,
                    pharm_padding_mask=pharm_padding_mask,
                )
            else:  # use the regular backbone for self-conditioning
                Xt_self_cond, Et_self_cond, Ct_self_cond = self.backbone(
                    torch.cat([Xt, Xt_self_cond], dim=-1),
                    torch.cat([Et, Et_self_cond], dim=-1),
                    torch.cat([Ct, Ct_self_cond], dim=-1),
                    atom_mask=atom_mask,
                )

            logits_X_subs, logits_E_subs = self._subs_parameterization(
                BBRxnGraph.from_onehot(Xt, Et), Xt_self_cond, Et_self_cond
            )
            Xt_self_cond = logits_X_subs.exp()
            Et_self_cond = logits_E_subs.exp()

        Xt = torch.cat([Xt, Xt_self_cond], dim=-1)
        Et = torch.cat([Et, Et_self_cond], dim=-1)
        Ct = torch.cat([Ct, Ct_self_cond], dim=-1)

        return Xt, Et, Ct

    def _process_sigma(self, sigma):
        if sigma is None:
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(self, Xt, Et, Ct, atom_mask, cond=None, sampling=False):
        """Returns log score."""

        # If sampling, self-conditioning is handled sample()
        if not sampling and self.self_conditioning:
            Xt, Et, Ct = self._self_conditioning_train(Xt, Et, Ct, atom_mask, cond)

        if cond is not None:
            pharm_types, pharm_pos, pharm_padding_mask = cond
            logits_X, logits_E, C0_pred = self.backbone(
                Xt,
                Et,
                Ct,
                atom_mask=atom_mask,
                pharm_types=pharm_types,
                pharm_pos=pharm_pos,
                pharm_padding_mask=pharm_padding_mask,
            )
        else:
            logits_X, logits_E, C0_pred = self.backbone(Xt, Et, Ct, atom_mask=atom_mask)

        return logits_X, logits_E, C0_pred

    def _subs_parameterization(self, graph: BBRxnGraph, logits_X, logits_E):
        """SUBS parameterization using BBRxnGraph to access masks and compatibilities."""
        Xt = graph.bb_onehot
        Et = graph.rxn_onehot
        # Node and edge padding masks
        node_padding_mask = graph.node_mask
        edge_padding_mask = node_padding_mask.unsqueeze(1) & node_padding_mask.unsqueeze(2)

        # Edge connectivity status
        n_edge_features = Et.shape[-1]
        active_edges = Et[..., :-2].sum(dim=-1)  # Sum over all features except last 2
        num_active_edges = (active_edges == 1).sum(dim=(1, 2))  # Sum over nodes for each batch item
        n_nodes = node_padding_mask.sum(dim=1)

        # If all the edges are denoised, only allow no-edge predictions
        fully_connected = (num_active_edges // 2) >= (n_nodes - 1)
        logits_E[fully_connected] = torch.full_like(logits_E[fully_connected], self.neg_infinity)
        logits_E[fully_connected, :, :, -2] = 0  # Allow only no-edge token

        # log prob at the mask index = - infinity
        logits_X[:, :, -1] += self.neg_infinity
        logits_E[:, :, :, -1] += self.neg_infinity

        # Optional compatibility masks from the graph

        if self.use_compat:
            compatibility_mask_X, compatibility_mask_E = graph.compatibility_masks
            logits_X[~compatibility_mask_X] += self.neg_infinity
            logits_E[~compatibility_mask_E] += self.neg_infinity

        # Normalize logits to log-probabilities
        logits_X = logits_X - torch.logsumexp(logits_X, dim=-1, keepdim=True)
        logits_E = logits_E - torch.logsumexp(logits_E, dim=-1, keepdim=True)

        # Freeze logits for already-unmasked tokens to their current class
        unmasked_nodes = graph.unmasked_bbs  # (B,N)
        unmasked_edges = graph.unmasked_rxns  # (B,N,N)
        unmasked_indices_X = unmasked_nodes.unsqueeze(-1).expand_as(logits_X)
        unmasked_indices_E = unmasked_edges.unsqueeze(-1).expand_as(logits_E)

        logits_X[unmasked_indices_X] = self.neg_infinity
        logits_X[(Xt == 1) & unmasked_indices_X] = 0

        logits_E[unmasked_indices_E] = self.neg_infinity
        logits_E[(Et == 1) & unmasked_indices_E] = 0

        # Hard-code the logits for the diagonals, respecting node_padding_mask
        batch_size, n, _, _ = logits_E.shape
        diag_indices = torch.arange(n, device=logits_E.device)
        diag_values = torch.full((logits_E.shape[-1],), self.neg_infinity, device=logits_E.device)
        diag_values[-2] = 0
        diag_values = diag_values.expand(batch_size, n, -1)
        logits_E[:, diag_indices, diag_indices, :] = diag_values

        # Validate sums over non-padded tokens
        tolerance = 1e-4
        sum_exp_X = torch.exp(logits_X).sum(dim=-1)
        sum_exp_E = torch.exp(logits_E).sum(dim=-1)

        assert torch.all(
            logits_X[node_padding_mask] <= 0
        ), "Found log probabilities > 0 in non-padded logits_X"
        assert torch.all(
            torch.abs(sum_exp_X[node_padding_mask] - 1) < tolerance
        ), f"Sum of exp(logits_X) is not close to 1 for non-padded tokens: {torch.abs(sum_exp_X[node_padding_mask] - 1)}"
        assert torch.all(
            torch.abs(sum_exp_E[edge_padding_mask] - 1) < tolerance
        ), f"Sum of exp(logits_E) is not close to 1 for non-padded tokens: {torch.abs(sum_exp_E[edge_padding_mask] - 1)}"

        return logits_X, logits_E

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            # do importance sampling by discrete noise schedule
            return self.discrete_noise.importance_sampling_transformation(t)
        return t

    def _noise_discrete(self, graph_batch: BBRxnGraph, discrete_sigma: torch.Tensor):
        """Computes the noisy samples Xt and Et.

        Only noises valid (non-padding) positions. Padding stays as zeros.
        The original node_mask is preserved in the returned graph.
        """
        X, E = graph_batch.bb_onehot, graph_batch.rxn_onehot
        node_mask = graph_batch.node_mask  # Preserve original mask
        move_chance = 1 - torch.exp(-discrete_sigma[:, None])

        # Expand move_chance to match the shape of X and E
        node_move_chance = move_chance.unsqueeze(2).expand(-1, -1, X.shape[2])
        edge_move_chance = move_chance.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, E.shape[3])

        # Generate random values for nodes and edges and expand to feature dim
        node_rand = torch.rand(X.shape[0], X.shape[1], device=X.device)
        edge_rand = torch.rand(E.shape[0], E.shape[1], E.shape[2], device=E.device)

        # Only noise valid positions (padding should stay as zeros)
        node_rand = torch.where(
            node_mask.unsqueeze(-1),
            node_rand.unsqueeze(-1),
            torch.ones_like(node_rand.unsqueeze(-1)),
        )
        node_rand = node_rand.expand_as(X)

        # Make edge_rand symmetric by averaging with its transpose, respecting edge mask
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # Valid edges
        edge_rand = 0.5 * (edge_rand + edge_rand.transpose(1, 2))
        edge_rand = torch.where(
            edge_mask.unsqueeze(-1),
            edge_rand.unsqueeze(-1),
            torch.ones_like(edge_rand.unsqueeze(-1)),
        )
        edge_rand = edge_rand.expand_as(E)

        # Generate random tensors and compare with expanded move_chance
        node_move_indices = node_rand < node_move_chance
        edge_move_indices = edge_rand < edge_move_chance

        # Create one-hot encoding for mask token
        mask_onehot_X = torch.zeros_like(X)
        mask_onehot_X[..., -1] = 1  # Set last feature to 1
        mask_onehot_E = torch.zeros_like(E)
        mask_onehot_E[..., -1] = 1  # Set last feature to 1

        # Apply masking (padding stays zeros because node_move_indices is False there)
        Xt = torch.where(node_move_indices, mask_onehot_X, X)
        Et = torch.where(edge_move_indices, mask_onehot_E, E)

        return BBRxnGraph.from_onehot(Xt, Et, node_mask=node_mask)

    def _noise_continuous(
        self,
        ground_truth_coords: Coordinates,
        ground_truth_graph_mask: torch.Tensor,
        partially_noised_graph_mask: torch.Tensor,
        t,
        prefix: str = "train",
    ):
        """Interpolate coordinates using selected interpolator with optional alignment and centering.

        Steps:
        - Sample prior C1 of same shape (here: random normal masked like C0)
        - Optional Kabsch-align C1 to C0 if train_rot_align
        - Create C0_shifted copy with atom mask for modifications
        - Special centering: center by atom-only mask
        - Interpolate via self.interpolator.forward(C0_shifted, C1, t)
        Returns (C0_shifted, coords_noised) - both Coordinates objects with pharmacophores attached if present
        """
        C0_tensor = ground_truth_coords.atom_coords

        # Sample simple prior C1 ~ N(0, 1) masked like C0
        C1 = Coordinates.random(
            shape=C0_tensor.shape,
            atom_mask=partially_noised_graph_mask,
            device=C0_tensor.device,
            dtype=C0_tensor.dtype,
        )

        # Apply scale noise if enabled (before centering, matching old implementation)
        if self.scale_noise:
            C1 = self._apply_scale_noise(C1)

        # Create C0_shifted copy with updated atom mask, preserving pharmacophores
        C0_shifted = Coordinates(
            coordinates=C0_tensor.clone(), atom_mask=partially_noised_graph_mask
        )
        if ground_truth_coords.has_pharmacophores:
            C0_shifted.attach_pharmacophores(
                pharm_coords=ground_truth_coords.pharmacophores.clone(),
                pharm_padding_mask=ground_truth_coords.pharm_padding_mask.clone(),
            )

        # Optional alignment of C1 to C0 using Kabsch
        if self.train_rot_align and prefix == "train":
            C1.kabsch_align_to(reference=C0_tensor, mask=ground_truth_graph_mask)

        # Special centering: center by atoms only
        # Compute atom-only mean and subtract from both C0_shifted and C1
        C1.center()
        C1_center_valid_only = C1.get_center(custom_mask=ground_truth_graph_mask)
        gt_mask_expanded = (
            ground_truth_graph_mask.unsqueeze(-1).expand_as(C0_shifted.atom_coords).bool()
        )
        partial_mask_expanded = (
            partially_noised_graph_mask.unsqueeze(-1).expand_as(C0_shifted.atom_coords).bool()
        )
        C0_shifted.set_coordinates(
            torch.where(
                gt_mask_expanded,
                C0_shifted.atom_coords + C1_center_valid_only.unsqueeze(1),
                C0_shifted.atom_coords,
            )
        )
        C0_shifted.set_coordinates(
            torch.where(
                ~gt_mask_expanded & partial_mask_expanded, C1.atom_coords, C0_shifted.atom_coords
            )
        )
        if C0_shifted.has_pharmacophores:
            # Shift pharmacophores by the same center offset as atoms
            C0_shifted.pharm_coords = torch.where(
                C0_shifted.pharm_padding_mask.unsqueeze(-1).bool(),
                C0_shifted.pharm_coords + C1_center_valid_only.unsqueeze(1),
                C0_shifted.pharm_coords,
            )

        # Interpolate
        assert C0_shifted.is_centered(), "C0_shifted must be centered"
        assert C1.is_centered(), "C1 must be centered"
        Ct = self.interpolator(C0_shifted.atom_coords, C1.atom_coords, t)

        # Wrap back into Coordinates object
        coords_noised = Coordinates(coordinates=Ct, atom_mask=partially_noised_graph_mask)
        if C0_shifted.has_pharmacophores:
            coords_noised.attach_pharmacophores(
                pharm_coords=C0_shifted.pharmacophores,
                pharm_padding_mask=C0_shifted.pharm_padding_mask,
            )
        return C0_shifted, coords_noised

    def sample(
        self,
        num_steps: int = 100,
        test_dataloader: DataLoader = None,
        cond: Optional[tuple] = None,
    ):
        """Sample a batch of graphs from the model.

        Args:
            num_steps: Number of diffusion steps
            test_dataloader: Optional dataloader for extracting pharmacophores (training mode)
            cond: Optional (types, pos, mask) tuple for direct pharmacophore conditioning (sampling mode)
        """
        from syncogen.constants.constants import MAX_ATOMS_PER_BB

        # 1. Get pharmacophore conditioning: prefer direct cond, fallback to dataloader
        if cond is not None:
            # Direct conditioning provided (standalone sampling)
            pharm_types, pharm_pos, pharm_padding_mask = cond
        elif self.data_manager.load_pharmacophores and test_dataloader is not None:
            # Extract from dataloader (training/validation)
            batch = next(iter(test_dataloader))
            pharm_types, pharm_pos, pharm_padding_mask = self.get_pharm_cond(batch)
            cond = (pharm_types, pharm_pos, pharm_padding_mask)
        else:
            cond = None
            pharm_pos = None
            pharm_padding_mask = None

        # 2. Initialize graph and coordinates objects ONCE (reused throughout loop)
        batch_size = self.eval_batch_size
        sampled_n_nodes = self.data_manager.sample_n_nodes(batch_size)
        max_nodes = self.data_manager.max_bbs
        max_atoms = max_nodes * MAX_ATOMS_PER_BB

        # Graph object - will update via property setters
        current_graph = BBRxnGraph.masked(
            max_nodes=max_nodes,
            batch_size=batch_size,
            n_nodes=sampled_n_nodes.tolist(),
            device=self.device,
        )

        # Coordinates object - will update via set_coordinates/set_mask
        current_coords = Coordinates.random(
            shape=(batch_size, max_atoms, 3),
            atom_mask=current_graph.partial_atom_mask.tensor,
            device=self.device,
        )

        if self.scale_noise:
            current_coords = self._apply_scale_noise(current_coords)

        if cond is not None:
            current_coords.attach_pharmacophores(
                pharm_coords=pharm_pos, pharm_padding_mask=pharm_padding_mask
            )

        current_coords.center()  # Center initial random coords like old _sample_coord_prior

        # 3. Initialize self-conditioning tensors if applicable
        if self.self_conditioning:
            X_self_cond = torch.zeros_like(current_graph.bb_onehot)
            E_self_cond = torch.zeros_like(current_graph.rxn_onehot)
            C_self_cond = torch.zeros_like(current_coords.atom_coords)

        eps = 1e-5
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps

        # 4. Step through the diffusion process
        for i in range(num_steps):

            # 4.1: Center coordinates (mutates current_coords in place)
            current_coords.center()
            if cond is not None and current_coords.has_pharmacophores:
                cond = (cond[0], current_coords.pharmacophores, cond[2])

            # 4.2: Prepare tensors for backbone (with optional self-conditioning)
            if self.self_conditioning:
                X_tensor = torch.cat([current_graph.bb_onehot, X_self_cond], dim=-1)
                E_tensor = torch.cat([current_graph.rxn_onehot, E_self_cond], dim=-1)
                C_tensor = torch.cat([current_coords.atom_coords, C_self_cond], dim=-1)
            else:
                X_tensor = current_graph.bb_onehot
                E_tensor = current_graph.rxn_onehot
                C_tensor = current_coords.atom_coords

            t = timesteps[i] * torch.ones(batch_size, 1, device=self.device)

            # 4.3: Forward pass through backbone
            logits_X, logits_E, C_pred = self.forward(
                X_tensor,
                E_tensor,
                C_tensor,
                atom_mask=current_graph.partial_atom_mask.tensor,
                cond=cond,
                sampling=True,
            )

            logits_X, logits_E = self._subs_parameterization(current_graph, logits_X, logits_E)
            p_x0 = logits_X.exp()
            p_e0 = logits_E.exp()

            # 4.4: Update via samplers (pass objects, they extract what they need)
            X_new, E_new = self.discrete_strategy.step(current_graph, p_x0, p_e0, t, dt)
            C_new = self.integrator.step(current_coords, C_pred, t, dt)

            # 4.5: Update objects via setters (triggers cache invalidation)
            current_graph.bb_onehot = X_new
            current_graph.rxn_onehot = E_new
            current_graph.apply_edge_givens()  # Enforce: diagonals=NO-EDGE, padding=zeros
            current_coords.set_coordinates(C_new, apply_mask=False)
            current_coords.set_mask(current_graph.partial_atom_mask.tensor)

            # 4.6 Update self-conditioning tensors if applicable
            if self.self_conditioning:
                X_self_cond = p_x0
                E_self_cond = p_e0
                C_self_cond = C_pred

        # 5. Final denoising step (optional)
        if self.sampling_noise_removal:
            # Center coordinates before final forward (like we do in the loop)
            current_coords.center()
            if cond is not None and current_coords.has_pharmacophores:
                cond = (cond[0], current_coords.pharmacophores, cond[2])

            t = timesteps[-1] * torch.ones(batch_size, 1, device=self.device)
            if self.self_conditioning:
                X_tensor = torch.cat([current_graph.bb_onehot, X_self_cond], dim=-1)
                E_tensor = torch.cat([current_graph.rxn_onehot, E_self_cond], dim=-1)
                C_tensor = torch.cat([current_coords.atom_coords, C_self_cond], dim=-1)
            else:
                X_tensor = current_graph.bb_onehot
                E_tensor = current_graph.rxn_onehot
                C_tensor = current_coords.atom_coords

            logits_X, logits_E, C_pred = self.forward(
                X_tensor,
                E_tensor,
                C_tensor,
                atom_mask=current_graph.partial_atom_mask.tensor,
                cond=cond,
                sampling=True,
            )
            logits_X, logits_E = self._subs_parameterization(current_graph, logits_X, logits_E)
            p_x0 = logits_X.exp()
            p_e0 = logits_E.exp()
            # Create final graph from argmax (discrete tokens)
            final_graph = BBRxnGraph.from_indices(p_x0.argmax(dim=-1), p_e0.argmax(dim=-1))
            final_graph.node_mask = current_graph.node_mask
            final_graph.apply_edge_givens()
            final_coords = Coordinates(
                coordinates=C_pred, atom_mask=current_graph.partial_atom_mask.tensor
            )
            final_coords.scale(1 / self.normalization_scale)
        else:
            # Use current state as final
            final_graph = current_graph.clone()
            final_coords = current_coords.clone()
            final_coords.scale(1 / self.normalization_scale)

        return final_graph, final_coords

    def restore_model_and_sample(
        self,
        num_steps: int,
        test_dataloader=None,
        cond: Optional[tuple] = None,
    ):
        """Generate samples under EMA weights, restoring training weights afterwards.

        Args:
            num_steps: Number of diffusion steps
            test_dataloader: Optional dataloader for extracting pharmacophores (training mode)
            cond: Optional (types, pos, mask) tuple for direct pharmacophore conditioning (sampling mode)
        """
        if self.ema:
            self.ema.store(
                itertools.chain(
                    self.backbone.parameters(),
                    self.discrete_noise.parameters(),
                )
            )
            self.ema.copy_to(
                itertools.chain(
                    self.backbone.parameters(),
                    self.discrete_noise.parameters(),
                )
            )
        self.backbone.eval()
        self.discrete_noise.eval()
        samples = self.sample(num_steps=num_steps, test_dataloader=test_dataloader, cond=cond)
        if self.ema:
            self.ema.restore(
                itertools.chain(
                    self.backbone.parameters(),
                    self.discrete_noise.parameters(),
                )
            )
        self.backbone.train()
        self.discrete_noise.train()
        return samples

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed
        )
        if distributed:
            sampler_cls = FaultTolerantDistributedSampler
        else:
            sampler_cls = RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, "shuffle"):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (
                distributed
                and self.fast_forward_epochs is not None
                and self.fast_forward_batches is not None
            ):
                dl_sampler.load_state_dict(
                    {
                        "epoch": self.fast_forward_epochs,
                        "counter": (self.fast_forward_batches * self.batch_size),
                    }
                )
            updated_dls.append(
                torch_geometric.loader.DataLoader(
                    dl.dataset,
                    batch_size=self.batch_size,
                    num_workers=self.data_manager.num_workers,
                    pin_memory=self.data_manager.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=self.data_manager.num_workers > 0,
                )
            )
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def on_train_epoch_start(self):
        pass

    def on_validation_epoch_start(self):
        if self.ema:
            self.ema.store(
                itertools.chain(
                    self.backbone.parameters(),
                    self.discrete_noise.parameters(),
                )
            )
            self.ema.copy_to(
                itertools.chain(
                    self.backbone.parameters(),
                    self.discrete_noise.parameters(),
                )
            )
        self.backbone.eval()
        self.discrete_noise.eval()

    def on_validation_epoch_end(self):
        if not self.generate_eval_samples:
            return

        # Only sample every N epochs
        if (self.current_epoch + 1) % self.sample_every_n_epochs != 0:
            return

        # Access validation dataloader directly from trainer
        val_loader = None
        if self.data_manager.load_pharmacophores:
            # Lightning stores dataloaders in trainer.val_dataloaders
            val_dataloaders = self.trainer.val_dataloaders
            if val_dataloaders is not None:
                # Can be a single loader or list of loaders
                val_loader = (
                    val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders
                )

        with torch.no_grad():
            graphs, coords = self.restore_model_and_sample(
                num_steps=self.num_sample_steps, test_dataloader=val_loader
            )

        # Build molecules once (with coordinates set)
        mols = build_molecules_from_graphs(graphs, coords.atom_coords)

        # Scalar metrics via MetricsList
        metrics = self.metrics.compute(graphs, mols)
        print(metrics)
        if metrics:
            self.log_dict(metrics, on_epoch=True, prog_bar=False, sync_dist=False)

        # Visualizations (rank zero only)
        run = getattr(self.logger, "experiment", None) or getattr(self.logger, "run", None)
        if run is not None and getattr(self.trainer, "is_global_zero", True):
            log_molecule_images(run, graphs, mols, step=self.global_step)
            log_fragment_histogram(run, graphs, step=self.global_step)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(
                itertools.chain(
                    self.backbone.parameters(),
                    self.discrete_noise.parameters(),
                )
            )

    def configure_optimizers(self):
        params = list(
            itertools.chain(
                self.backbone.parameters(),
                self.discrete_noise.parameters(),
            )
        )

        optimizer = getattr(torch.optim, self._optimizer_config.cls_name)(
            params, **self._optimizer_config.kwargs
        )

        if self._lr_scheduler_config is None:
            return optimizer

        if self._lr_scheduler_config.module == "torch.optim.lr_scheduler":
            scheduler = getattr(torch.optim.lr_scheduler, self._lr_scheduler_config.cls_name)(
                optimizer, **self._lr_scheduler_config.kwargs
            )
        elif self._lr_scheduler_config.module == "transformers":
            import transformers

            scheduler = getattr(transformers, self._lr_scheduler_config.cls_name)(
                optimizer, **self._lr_scheduler_config.kwargs
            )
        else:
            raise ValueError(f"Unknown module: {self._lr_scheduler_config.module}")

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "trainer/lr"}]

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint["ema"])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"][
            "completed"
        ]
        self.fast_forward_batches = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"][
            "current"
        ]["completed"]

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
