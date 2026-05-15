"""Slice 3 (Collins) BVP-reframed augmentation: velocity-coordinate-frame jitter.

The original Slice 3 hypothesis (calibrated phase-noise injection) targeted
**chipset-invariance** on raw complex CSI. BVP is real-valued; phase doesn't
exist on it. See `papers/team/bvp-augmentation-hypotheses.md` for the
substrate pivot rationale.

The BVP-reframed hypothesis: BVP coordinate frames sometimes mis-align with
true body coordinates because the Widar3.0 offline pipeline estimates torso
orientation and receiver geometry imperfectly. A robust encoder should be
invariant to small rotations and translations in the (vx, vy) velocity plane.

Augmentation: apply a small random affine to each sample's (vx, vy) plane,
preserving the time axis and treating vx/vy as spatial dimensions. Two
independent calls produce two views differing in the random affine, suitable
as a SimCLR view-pair augmentation.

Input shape: `(T, vx, vy)` or `(B, T, vx, vy)` real-valued BVP tensor.
Default ranges chosen to be small (~5°, 1-cell shift) so the augmentation
preserves gesture-class identity while perturbing the coordinate frame.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEFAULT_ANGLE_RANGE_DEG = (-5.0, 5.0)
DEFAULT_SHIFT_RANGE_CELLS = (-1.0, 1.0)


def _sample_uniform(low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high).item())


def _affine_grid(
    theta_deg: float,
    shift_x_cells: float,
    shift_y_cells: float,
    vx: int,
    vy: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a `(1, vx, vy, 2)` sampling grid for an affine in normalized coords.

    PyTorch's ``F.affine_grid`` / ``F.grid_sample`` use coordinates in
    `[-1, 1]` for both axes; a 1-cell shift on a `vx`-cell axis is therefore
    `2 / vx` in normalized units.
    """
    theta = math.radians(theta_deg)
    cos = math.cos(theta)
    sin = math.sin(theta)
    tx = 2.0 * shift_x_cells / max(1, vx)
    ty = 2.0 * shift_y_cells / max(1, vy)
    # (2, 3) affine matrix [[cos, -sin, tx], [sin, cos, ty]].
    A = torch.tensor(
        [[cos, -sin, tx], [sin, cos, ty]],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    return F.affine_grid(A, size=(1, 1, vx, vy), align_corners=False)


def _jitter_one(
    x: torch.Tensor,
    *,
    angle_deg: float,
    shift_x: float,
    shift_y: float,
) -> torch.Tensor:
    """Apply one affine to a single ``(T, vx, vy)`` BVP sample."""
    if x.ndim != 3:
        raise ValueError(f"_jitter_one expects (T, vx, vy); got {tuple(x.shape)}")
    t, vx, vy = x.shape
    grid = _affine_grid(
        angle_deg, shift_x, shift_y, vx, vy, device=x.device, dtype=x.dtype
    )
    # Apply the same grid to every time step. Treat T as batch for grid_sample.
    x_in = x.unsqueeze(1)  # (T, 1, vx, vy)
    grid_t = grid.expand(t, vx, vy, 2)
    return F.grid_sample(
        x_in, grid_t, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(1)


def bvp_velocity_jitter(
    x: torch.Tensor,
    *,
    angle_range_deg: tuple[float, float] = DEFAULT_ANGLE_RANGE_DEG,
    shift_range_cells: tuple[float, float] = DEFAULT_SHIFT_RANGE_CELLS,
) -> torch.Tensor:
    """Velocity-coordinate-frame jitter augmentation for BVP samples.

    Accepts `(T, vx, vy)` single sample or `(B, T, vx, vy)` batched. Each
    sample gets an independent random affine sampled from the configured
    ranges, so two SimCLR view-pair calls produce two views with different
    jitter parameters.

    Args:
        x: real-valued BVP tensor.
        angle_range_deg: uniform sampling range for the in-plane rotation,
            in degrees. Default `(-5, 5)`.
        shift_range_cells: uniform sampling range for translation along each
            of (vx, vy), in cell units. Default `(-1, 1)`.

    Returns:
        Tensor of the same shape as `x`; out-of-frame pixels are zero-filled.
    """
    if x.ndim == 3:
        ang = _sample_uniform(*angle_range_deg)
        sx = _sample_uniform(*shift_range_cells)
        sy = _sample_uniform(*shift_range_cells)
        return _jitter_one(x, angle_deg=ang, shift_x=sx, shift_y=sy)
    if x.ndim == 4:
        b = x.shape[0]
        out = torch.empty_like(x)
        for i in range(b):
            ang = _sample_uniform(*angle_range_deg)
            sx = _sample_uniform(*shift_range_cells)
            sy = _sample_uniform(*shift_range_cells)
            out[i] = _jitter_one(x[i], angle_deg=ang, shift_x=sx, shift_y=sy)
        return out
    raise ValueError(
        f"bvp_velocity_jitter expects (T, vx, vy) or (B, T, vx, vy); got {tuple(x.shape)}"
    )
