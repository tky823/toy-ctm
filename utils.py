import math
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm


def create_2d_samples(num_samples: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
    x1 = 0.3 * torch.randn((num_samples // 4, 2)) + torch.tensor([1, 1])
    x2 = 0.3 * torch.randn((num_samples // 4, 2)) + torch.tensor([1, -1])
    x3 = 0.3 * torch.randn((num_samples // 4, 2)) + torch.tensor([-1, 1])
    x4 = 0.3 * torch.randn((num_samples // 4, 2)) + torch.tensor([-1, -1])

    return torch.cat([x1, x2, x3, x4], dim=0)


def plot_loss(loss: List[float], filename: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(loss)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


class Denoiser(nn.Module):

    def __init__(self, max_step: int) -> None:
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


class ConsistencyTrajectoryModel(nn.Module):

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
        normalized_input = input / (math.sqrt(2) * self.max_step)
        normalized_step = step / self.max_step
        normalized_step = normalized_step.unsqueeze(dim=-1)
        normalized_target_step = target_step / self.max_step
        normalized_target_step = normalized_target_step.unsqueeze(dim=-1)
        x = torch.cat([normalized_input, normalized_step, normalized_target_step], dim=-1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        output = (math.sqrt(2) * self.max_step) * self.linear4(x)

        return output


class Solver(nn.Module):

    def __init__(self, net: nn.Module, delta: float = 1) -> None:
        super().__init__()

        self.net = net
        self.delta = delta

    def forward(
        self, input: torch.Tensor, step: float, target_step: float, pbar: Optional[tqdm] = None
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

            t = t_delta
            x_t = torch.lerp(x_t, output, 1 - weight.unsqueeze(dim=-1))

            if t <= target_step:
                return x_t
