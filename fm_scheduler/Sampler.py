import torch
from tqdm import tqdm
from typing import Literal, Optional


class Sampler:
    """
    A generic sampler for ODE-based generative models.

    This class provides methods for generating samples from a trained model
    using different numerical ODE solvers.
    """

    def __init__(self, model):
        """
        Initializes the Sampler.

        Args:
            model: The trained model to be used for sampling.
                   The model should have a `forward` method that takes (x, t, cond) and returns the velocity.
        """
        self.model = model
        self.device = next(model.parameters()).device

    def _euler_step(self, x: torch.Tensor, t: float, dt: float, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a single Euler step."""
        velocity = self.model(x, torch.tensor([t], device = self.device), cond)
        return x + velocity * dt

    def _midpoint_step(self, x: torch.Tensor, t: float, dt: float, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        k1 = self.model(x, torch.tensor([t], device = self.device), cond)
        k2 = self.model(x + dt * k1, torch.tensor([t + dt], device = self.device), cond)
        return x + dt * (k1 + k2) / 2

    def _rk4_step(self, x: torch.Tensor, t: float, dt: float, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        k1 = self.model(x, torch.tensor([t], device = self.device), cond)
        k2 = self.model(x + 0.5 * dt * k1, torch.tensor([t + 0.5 * dt], device = self.device), cond)
        k3 = self.model(x + 0.5 * dt * k2, torch.tensor([t + 0.5 * dt], device = self.device), cond)
        k4 = self.model(x + dt * k3, torch.tensor([t + dt], device = self.device), cond)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    @torch.no_grad()
    def sample(
            self,
            shape,
            n_steps: int,
            sampler_type: Literal['euler', 'midpoint', 'rk4'] = 'euler',
            cond: Optional[torch.Tensor] = None,
            initial_noise: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Generates samples from the model.

        Args:
            shape: The shape of the desired output tensor, e.g., (batch_size, channels, height, width).
            n_steps: The number of integration steps.
            sampler_type: The ODE solver to use ('euler', 'midpoint', or 'rk4').
            cond: Optional conditional input for the model.
            initial_noise: Optional initial noise tensor at t=1. If None, it will be sampled from a standard normal distribution.

        Returns:
            The generated samples at t=0.
        """
        shape = tuple(shape)

        if initial_noise is None:
            x = torch.randn(shape, device = self.device)
        else:
            x = initial_noise.to(self.device)

        if hasattr(self.model, 'has_cond') and self.model.has_cond and cond is not None:
            cond = cond.to(self.device)
            cond = torch.tensor(cond)

        time_steps = torch.linspace(0, 1 - 1e-5, n_steps + 1, device = self.device)

        solvers = {
            'euler': self._euler_step,
            'midpoint': self._midpoint_step,
            'rk4': self._rk4_step
        }

        if sampler_type not in solvers:
            raise ValueError(f"Unknown sampler type: {sampler_type}. Available options are {list(solvers.keys())}")

        solver_step = solvers[sampler_type]

        pbar = tqdm(range(n_steps), desc = f"Sampling with {sampler_type}", position = 1, leave = True,
                    disable = True)
        for i in pbar:
            t = time_steps[i]
            dt = time_steps[i + 1] - t
            x = solver_step(x, t.item(), dt.item(), cond)

        return x
