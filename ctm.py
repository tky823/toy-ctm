import math

import torch
import torch.nn as nn


class ConsistencyTrajectoryModel(nn.Module):
    """Consistency trajectory model.

    Args:
        max_step (float): Max step of diffusion process.

    """

    def __init__(self, max_step: int) -> None:
        super().__init__()

        self.max_step = max_step

        in_features = 2

        self.linear1 = nn.Linear(in_features + 2, 64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(64, in_features)

    def forward(
        self, input: torch.Tensor, step: torch.Tensor, target_step: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of ConsistencyTrajectoryModel.

        Args:
            input (torch.Tensor): Noisy input of shape (batch_size, in_features).
            step (torch.Tensor): Step in diffusion process of shape (batch_size,).
            target_step (torch.Tensor): Target step in diffusion process of shape (batch_size,).

        Returns:
            torch.Tensor: Estimated data at target step of shape (batch_size, in_features).

        """
        normalized_input = input / (math.sqrt(2) * self.max_step)
        normalized_step = step / self.max_step
        normalized_step = normalized_step.unsqueeze(dim=-1)
        normalized_target_step = target_step / self.max_step
        normalized_target_step = normalized_target_step.unsqueeze(dim=-1)
        x = torch.cat(
            [normalized_input, normalized_step, normalized_target_step], dim=-1
        )
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        output = (math.sqrt(2) * self.max_step) * self.linear4(x)

        return output
