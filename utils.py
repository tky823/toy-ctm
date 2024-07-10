import math
from typing import List

import matplotlib.pyplot as plt
import torch


def create_2d_samples(num_samples: int = 500, dist: str = "gmm") -> torch.Tensor:
    if dist == "gmm":
        return create_gmm_2d_samples(num_samples)
    elif dist == "roll":
        return create_roll_2d_samples(num_samples)
    else:
        raise ValueError(f"{dist} is not supported.")


def create_gmm_2d_samples(num_samples: int = 500) -> torch.Tensor:
    x1 = 0.3 * torch.randn((num_samples // 4, 2)) + torch.tensor([1, 1])
    x2 = 0.3 * torch.randn((num_samples // 4, 2)) + torch.tensor([1, -1])
    x3 = 0.3 * torch.randn((num_samples // 4, 2)) + torch.tensor([-1, 1])
    x4 = 0.3 * torch.randn((num_samples // 4, 2)) + torch.tensor([-1, -1])

    return torch.cat([x1, x2, x3, x4], dim=0)


def create_roll_2d_samples(num_samples: int = 500) -> torch.Tensor:
    theta1 = math.pi * (torch.rand((num_samples // 2)) - 0.5)
    theta2 = math.pi * (torch.rand((num_samples // 2)) + 0.5)
    x1 = torch.cos(theta1)
    y1 = torch.sin(theta1) + 0.5
    x2 = torch.cos(theta2)
    y2 = torch.sin(theta2) - 0.5

    x1 = torch.stack([x1, y1], dim=-1)
    x2 = torch.stack([x2, y2], dim=-1)

    x1 = x1 + 0.05 * torch.randn_like(x1)
    x2 = x2 + 0.05 * torch.randn_like(x2)

    return torch.cat([x1, x2], dim=0)


def plot_loss(loss: List[float], filename: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(loss)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
