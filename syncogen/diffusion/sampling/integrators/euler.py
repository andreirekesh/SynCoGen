"""Euler integrator for flow matching."""

import torch

from syncogen.api.atomics.coordinates import Coordinates
from syncogen.diffusion.sampling.integrators.base import IntegratorBase


class EulerIntegrator(IntegratorBase):
    """Simple Euler integrator for flow matching.

    Updates coordinates using: C_{t-dt} = C_t + v_theta(C_t, t) * dt
    where v_theta is the velocity field pointing from C_t toward C_0.
    """

    def __init__(self, inference_annealing: bool = False, annealing_coef: float = 1.0):
        """Initialize Euler integrator.

        Args:
            inference_annealing: Whether to apply inference annealing
            annealing_coef: Coefficient for inference annealing
        """
        self.inference_annealing = inference_annealing
        self.annealing_coef = annealing_coef

    def compute_velocity(
        self, C: torch.Tensor, C0_pred: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the velocity field v_theta at time t.

        For flow matching: v_t = (C_0 - C_t) / t
        This points from C_t toward C_0.

        Args:
            C: Current noisy coordinates
            C0_pred: Predicted clean coordinates
            t: Current timestep

        Returns:
            Velocity tensor
        """
        assert not torch.any(t == 0), "t must not be zero to avoid division by zero"
        while t.dim() < C.dim():
            t = t.unsqueeze(-1)

        return (C0_pred - C) / t

    def step(
        self,
        coords: Coordinates,
        C0_pred: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Perform one Euler integration step.

        Args:
            coords: Coordinates object with current noisy state
            C0_pred: Predicted clean coordinates
            t: Current timestep
            dt: Time step size

        Returns:
            C_next: Updated coordinates tensor
        """
        # Extract coordinate tensor
        C = coords.tensor

        # Compute velocity field
        v_theta = self.compute_velocity(C, C0_pred, t)

        annealing = 1.0
        # More aggressive denoising early
        if self.inference_annealing:
            t_scalar = t.mean().item() if t.numel() > 1 else t.item()
            annealing = self.annealing_coef * t_scalar

        C_next = C + v_theta * dt * annealing

        return C_next
