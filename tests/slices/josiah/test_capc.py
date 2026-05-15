"""Tests for the CAPC exact reproduction (T5.5)."""

from __future__ import annotations

import torch

from src.slices.josiah.capc import (
    CAPCBranch,
    LARS,
    RSCNetBlock,
    barlow_twins_loss,
    capc_total_loss,
    capc_view_noise_then_submask,
    cpc_loss,
    make_capc_optimizer,
    pretrain_capc,
    subcarrier_mask,
    time_flip,
    time_mask,
)
from src.slices.josiah.data import CSI_A, CSI_S, CSI_T, NUM_CLASSES, StubCSI


def test_rscnet_block_output_shape() -> None:
    enc = RSCNetBlock(in_channels=CSI_A, s=CSI_S, n_f=10, embedding_dim=128)
    x = torch.randn(2, CSI_A, CSI_S, 10)
    assert enc(x).shape == (2, 128)


def test_capc_branch_encode_windows_shape() -> None:
    branch = CAPCBranch(in_channels=CSI_A, s=CSI_S, n_f=10, embedding_dim=128,
                        hidden_dim=128, future_steps=9)
    windows = torch.randn(2, 10, CSI_A, CSI_S, 10)  # (B, L, A, S, N_f)
    z = branch.encode_windows(windows)
    assert z.shape == (2, 10, 128)
    c = branch.context(z)
    assert c.shape == (2, 10, 128)


def test_barlow_twins_loss_zero_for_identical_inputs() -> None:
    torch.manual_seed(0)
    z = torch.randn(32, 16)
    loss = barlow_twins_loss(z, z, lam=0.002)
    # Self-similarity: diagonal terms are 1 (after normalisation) so (C_ii-1)^2 ~ 0;
    # off-diagonal terms encode redundancy. Should be finite but small relative to
    # the unrelated-views case.
    z2 = torch.randn(32, 16)
    loss_unrelated = barlow_twins_loss(z, z2, lam=0.002)
    assert loss.item() < loss_unrelated.item()


def test_cpc_loss_runs() -> None:
    torch.manual_seed(0)
    b, length, d, h = 4, 8, 16, 16
    z = torch.randn(b, length, d)
    c = torch.randn(b, length, h)
    W = torch.nn.ModuleList([torch.nn.Linear(h, d, bias=False) for _ in range(5)])
    loss = cpc_loss(z, c, W, t_anchor=2, future_steps=5)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_capc_total_loss_structure() -> None:
    torch.manual_seed(0)
    b, length, d, h = 4, 6, 16, 16
    z_a = torch.randn(b, length, d)
    z_b = torch.randn(b, length, d)
    c_a = torch.randn(b, length, h)
    c_b = torch.randn(b, length, h)
    W_a = torch.nn.ModuleList([torch.nn.Linear(h, d, bias=False) for _ in range(3)])
    W_b = torch.nn.ModuleList([torch.nn.Linear(h, d, bias=False) for _ in range(3)])
    total, parts = capc_total_loss(z_a, c_a, z_b, c_b, W_a, W_b,
                                   future_steps=3, beta=50.0, bt_lambda=0.002)
    assert total.dim() == 0
    assert set(parts) == {"L_BT", "L_CPC_A", "L_CPC_B"}


def test_capc_augmentations_preserve_shape() -> None:
    x = torch.randn(2, CSI_T, CSI_S, CSI_A)
    for f in (time_flip, time_mask, subcarrier_mask, capc_view_noise_then_submask):
        out = f(x)
        assert out.shape == x.shape


def test_lars_step_updates_weights() -> None:
    p = torch.nn.Parameter(torch.ones(4, requires_grad=True))
    optim = LARS([p], lr=0.1, momentum=0.0, weight_decay=0.0, trust_coef=1.0)
    p.grad = torch.ones(4)
    before = p.detach().clone()
    optim.step()
    assert not torch.equal(p.detach(), before)


def test_make_capc_optimizer_param_groups() -> None:
    branch_a = CAPCBranch(in_channels=CSI_A, s=CSI_S, n_f=10)
    branch_b = CAPCBranch(in_channels=CSI_A, s=CSI_S, n_f=10)
    optim = make_capc_optimizer(branch_a, branch_b)
    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["lr"] == 0.2
    assert optim.param_groups[1]["lr"] == 0.0048


def test_pretrain_capc_smoke() -> None:
    torch.manual_seed(0)
    ds = StubCSI(num_samples=8, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
    branch_a = CAPCBranch(in_channels=CSI_A, s=CSI_S, n_f=10, future_steps=9)
    branch_b = CAPCBranch(in_channels=CSI_A, s=CSI_S, n_f=10, future_steps=9)
    losses = pretrain_capc(branch_a, branch_b, loader, epochs=2, n_f=10)
    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
