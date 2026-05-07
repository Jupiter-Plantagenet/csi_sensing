"""Architecture figure generation for the Slice 1 KICS paper.

Produces ``papers/kics-george/fig1-pipeline.pdf``: a left-to-right pipeline
showing the Widar3.0 CSI input, two Doppler-warped views, the tiny CNN
encoder, the NT-Xent contrastive loss, and the downstream frozen-encoder
linear-probe branch. Visual style mirrors the prior paper's Fig. 1.

The rendered PDF lives under the gitignored ``papers/kics-george/``
directory; this script is the project-tracked codebase for the figure so
teammates can regenerate or restyle it.

Run:
    python -m src.slices.george.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DEFAULT_OUT = Path("papers/kics-george/fig1-pipeline.pdf")


def _box(ax, x, y, w, h, text, fc="#e8eef7"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04",
        linewidth=1.0,
        edgecolor="black",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def _arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        color="black",
        linewidth=0.9,
    )
    ax.add_patch(arrow)


def make_pipeline_figure(out_path: Path = DEFAULT_OUT) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(-1.2, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    _box(ax, 0.2, 2.2, 1.9, 1.6, "Widar3.0\nCSI sample", fc="#fff3cd")
    _box(
        ax,
        3.1,
        4.0,
        2.4,
        1.4,
        "View 1\nDoppler warp\nfactor $f_1$",
        fc="#d1ecf1",
    )
    _box(
        ax,
        3.1,
        0.6,
        2.4,
        1.4,
        "View 2\nDoppler warp\nfactor $f_2$",
        fc="#d1ecf1",
    )
    _box(ax, 6.6, 2.2, 2.4, 1.6, r"Tiny CNN encoder $f_\theta$", fc="#e8eef7")
    _box(ax, 10.1, 2.2, 1.8, 1.6, "NT-Xent\n(SimCLR)", fc="#f8d7da")
    _box(
        ax,
        6.6,
        -0.9,
        5.3,
        1.0,
        r"Frozen $f_\theta$ + linear probe $\to$ cross-subject prediction",
        fc="#d4edda",
    )

    _arrow(ax, 2.1, 3.4, 3.1, 4.6)
    _arrow(ax, 2.1, 2.6, 3.1, 1.3)
    _arrow(ax, 5.5, 4.6, 6.6, 3.4)
    _arrow(ax, 5.5, 1.3, 6.6, 2.6)
    _arrow(ax, 9.0, 3.0, 10.1, 3.0)
    _arrow(ax, 7.8, 2.2, 7.8, 0.1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    saved = make_pipeline_figure()
    print(f"wrote {saved}")
