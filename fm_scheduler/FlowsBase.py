import torch
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class FlowSchedulerBase(ABC):
    """
    所有Flow Scheduler的抽象基类。

    这个类定义了所有流匹配调度器必须实现的通用接口。
    不同的流派生（如VP, VE, OT-CFM）都应继承此类并实现其抽象方法。
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def define_path_params(self, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算条件概率路径 p_t(x|x1) = N(x|μ_t(x1), σ_t(x1)^2 * I) 的均值和标准差。
        即，路径为 ψ_t(x) = μ_t(x1) + σ_t(x1) * x0。

        Args:
            x1 (torch.Tensor): 真实数据样本。
            t (torch.Tensor): 时间步。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (μ_t, σ_t)，即均值和标准差。
        """
        pass

    @abstractmethod
    def define_path_params_derivatives(self, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算μ_t和σ_t在t时刻对t的导数值。
        这是计算条件向量场 u_t 的核心。

        Args:
            x1 (torch.Tensor): 真实数据样本。
            t (torch.Tensor): 时间步。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (dμ_t/dt, dσ_t/dt)，即均值和标准差的导数。
        """
        pass

    def prepare_training_pair(self, x1: torch.Tensor, t: torch.Tensor, x0: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        为训练准备模型的输入和目标。

        Args:
            x1 (torch.Tensor): 训练数据。
            t (torch.Tensor): 时间步。
            x0 (Optional[torch.Tensor]): 初始噪声。如果为None，则会自动采样。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - xt (torch.Tensor): 加噪后的数据，作为模型的输入。
                - target (torch.Tensor): 训练目标（速度或噪声）。
        """
        if x0 is None:
            x0 = torch.randn_like(x1)

        # 1. 计算路径上的点xt
        mean, std = self.define_path_params(x1, t)
        xt = mean + std * x0

        # 2. 根据prediction_type计算训练目标
        d_mean, d_std = self.define_path_params_derivatives(x1, t)

        assert mean.dim() == 4 and std.dim() == 4 and d_mean.dim() == 4 and d_std.dim() == 4

        std_safe = std + 1e-8

        target = d_mean + (d_std / std_safe) * (xt - mean)

        return xt, target
