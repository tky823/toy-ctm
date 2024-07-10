import matplotlib.pyplot as plt
import torch
from torch import optim
from tqdm import tqdm

from ctm import ConsistencyTrajectoryModel
from dm import Denoiser, Solver
from utils import create_2d_samples, plot_loss


def main() -> None:
    torch.manual_seed(0)

    num_samples = 1000
    dist = "gmm"  # "gmm" or "roll"
    T = 10

    # pseudo statistics of distributions
    x_0 = create_2d_samples(num_samples=num_samples, dist=dist)
    data_std, data_mean = torch.std_mean(x_0, dim=0)
    data_std = data_std / T

    # training of teacher model
    iterations = 2000
    batch_size = 2048

    dm = Denoiser(max_step=T)
    optimizer = optim.Adam(dm.parameters())
    loss = []

    for _ in tqdm(range(iterations)):
        x_0 = create_2d_samples(num_samples=batch_size, dist=dist)
        x_0 = (x_0 - data_mean) / data_std

        step = T * torch.rand((batch_size,))
        noise = step.unsqueeze(dim=-1) * torch.randn_like(x_0)
        x_t = x_0 + noise
        output = dm(x_t, step=step)
        loss_per_iter = torch.mean((x_0 - output) ** 2)

        optimizer.zero_grad()
        loss_per_iter.backward()
        optimizer.step()

        loss.append(loss_per_iter.item())

    plot_loss(loss, "loss_teacher.png")

    # training of student model
    delta = 0.05
    solver = Solver(dm, delta=delta)
    solver.eval()

    ctm = ConsistencyTrajectoryModel(max_step=T)
    optimizer = optim.Adam(ctm.parameters())
    loss_ctm = []
    loss_dsm = []
    loss = []

    for _ in tqdm(range(iterations)):
        x_0 = create_2d_samples(num_samples=batch_size, dist=dist)
        x_0 = (x_0 - data_mean) / data_std

        t = T * torch.rand(()).item()
        u = t * torch.rand(()).item()
        target_step = u * torch.rand((batch_size,))  # corresponds to s

        noise = step.unsqueeze(dim=-1) * torch.randn_like(x_0)
        x_t = x_0 + noise

        with torch.no_grad():
            x_u = solver(x_t, t, u)
            step = torch.full((batch_size,), fill_value=u)
            output = ctm(x_u, step=step, target_step=target_step)
            weight = target_step / step
            weight = weight.unsqueeze(dim=-1)
            x_s_teacher = weight * x_u + (1 - weight) * output

        step = torch.full((batch_size,), fill_value=t)
        output = ctm(x_t, step=step, target_step=target_step)
        weight = target_step / step
        weight = weight.unsqueeze(dim=-1)
        x_s_student = weight * x_t + (1 - weight) * output

        loss_ctm_per_iter = torch.mean((x_s_teacher - x_s_student) ** 2)
        loss_dsm_per_iter = torch.mean(torch.abs(x_0 - output))
        loss_per_iter = loss_ctm_per_iter + 10 * loss_dsm_per_iter

        optimizer.zero_grad()
        loss_per_iter.backward()
        optimizer.step()

        loss_ctm.append(loss_ctm_per_iter.item())
        loss_dsm.append(loss_dsm_per_iter.item())
        loss.append(loss_per_iter.item())

    plot_loss(loss_ctm, "loss_ctm.png")
    plot_loss(loss_dsm, "loss_dsm.png")
    plot_loss(loss_dsm, "loss_student.png")

    # generation by teacher model
    solver.eval()

    with torch.no_grad():
        x_0 = create_2d_samples(
            num_samples=num_samples, dist=dist
        )  # dummy samples to obtain tensor size
        x_T = T * torch.randn_like(x_0)

        pbar = tqdm()
        x_0_hat_teacher = solver(x_T, step=T, target_step=0, pbar=pbar)
        pbar.clear()

    # generation by student model
    ctm.eval()

    with torch.no_grad():
        x_0 = create_2d_samples(
            num_samples=num_samples, dist=dist
        )  # dummy samples to obtain tensor size
        x_T = T * torch.randn_like(x_0)
        step = torch.full((num_samples,), fill_value=T)
        target_step = torch.full((num_samples,), fill_value=0)
        x_0_hat_ctm = ctm(x_T, step=step, target_step=target_step)

    x_0 = create_2d_samples(num_samples=num_samples, dist=dist)
    x_0 = (x_0 - data_mean) / data_std
    vmax = max(
        torch.max(torch.abs(x_0_hat_teacher)).item(),
        torch.max(torch.abs(x_0_hat_ctm)).item(),
        torch.max(torch.abs(x_0)).item(),
    )

    plt.figure(figsize=(8, 8))
    plt.scatter(x_0[:, 0], x_0[:, 1], alpha=0.5, label="x_0")
    plt.xlim(-vmax, vmax)
    plt.ylim(-vmax, vmax)
    plt.legend()
    plt.savefig("p_0.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(x_0[:, 0], x_0[:, 1], alpha=0.5, label="x_0")
    plt.scatter(
        x_0_hat_teacher[:, 0], x_0_hat_teacher[:, 1], alpha=0.5, label="x_0 (teacher)"
    )
    plt.scatter(x_0_hat_ctm[:, 0], x_0_hat_ctm[:, 1], alpha=0.5, label="x_0 (CTM)")
    plt.xlim(-vmax, vmax)
    plt.ylim(-vmax, vmax)
    plt.legend()
    plt.savefig("p_0_hat.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
