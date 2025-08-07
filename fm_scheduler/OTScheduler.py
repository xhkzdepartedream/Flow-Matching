import torch
from typing import Optional, Tuple
from .FlowsBase import FlowSchedulerBase


class OTScheduler(FlowSchedulerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def define_path_params(self, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_t = t.reshape(-1, 1, 1, 1) * x1
        sigma_t = 1 - t
        return mu_t, sigma_t.reshape(-1, 1, 1, 1)

    def define_path_params_derivatives(self, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x1, -torch.ones_like(t).reshape(-1, 1, 1, 1)
