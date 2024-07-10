import math
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm


class Denoiser(nn.Module):
    """Denoiser which estimates denoised input.

    Args:
        max_step (float): Max step of diffusion process.

    """

    def __init__(self, max_step: float) -> None:
        super().__init__()

        self.max_step = max_step

        in_features = 2

        self.linear1 = nn.Linear(in_features + 1, 64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(64, in_features)

    def forward(self, input: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Forward pass of Denoiser.

        Args:
            input (torch.Tensor): Noisy input of shape (batch_size, in_features).
            step (torch.Tensor): Step in diffusion process of shape (batch_size,).

        Returns:
            torch.Tensor: Denoised input of shape (batch_size, in_features).

        """
        normalized_input = input / (math.sqrt(2) * self.max_step)
        normalized_step = step / self.max_step
        normalized_step = normalized_step.unsqueeze(dim=-1)
        x = torch.cat([normalized_input, normalized_step], dim=-1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        output = (math.sqrt(2) * self.max_step) * self.linear4(x)

        return output


class Solver(nn.Module):
    """ODE solver based on Euler method."""

    def __init__(self, net: nn.Module, delta: float = 1) -> None:
        super().__init__()

        self.net = net
        self.delta = delta

    def forward(
        self,
        input: torch.Tensor,
        step: float,
        target_step: float,
        pbar: Optional[tqdm] = None,
    ) -> torch.Tensor:
        assert not self.training
        assert target_step <= step

        delta = self.delta

        t = step
        x_t = input
        batch_size = x_t.size(0)

        while True:
            if pbar is not None:
                pbar.update(1)

            t_delta = t - delta

            if t_delta <= target_step:
                t_delta = target_step

            step = torch.full((batch_size,), fill_value=t)
            output = self.net(x_t, step=step)
            weight = t_delta / step
            weight = weight.unsqueeze(dim=-1)

            t = t_delta
            x_t = weight * x_t + (1 - weight) * output

            if t <= target_step:
                return x_t
