from typing import List

import matplotlib.pyplot as plt
import torch


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
