"""End-to-end run for Slice 5.

Three baselines wired through `--mode`:

- `mode=supervised` (T5.1 / T5.2): cross-entropy on the full classifier.
- `mode=simclr-trivial` (T5.3): SimCLR pre-train with `random_crop` only,
  then frozen-encoder linear probe.
- `mode=simclr-handcrafted` (T5.6): SimCLR pre-train with the
  hand-crafted-augmentation baseline (added in T5.6).

Each mode supports `--real` to swap stub data for real Widar3.0 cross-subject.

Run:
    python -m src.slices.josiah.run                                   # supervised stub
    python -m src.slices.josiah.run --real --epochs 10                # supervised real
    python -m src.slices.josiah.run --mode simclr-trivial --real      # T5.3 real
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentations import gaussian_then_mask, random_crop
from .autofi import AutoFiCNN, AutoFiGSS, pretrain_autofi
from .capc import CAPCBranch, pretrain_capc
from .data import CSI_A, CSI_S, CSI_T, NUM_CLASSES, StubCSI, Widar3CrossSubject
from .encoder import SupervisedClassifier, TinyCNN, count_parameters
from .eval import evaluate, linear_probe, train_supervised
from .ssl import SimCLR, pretrain_simclr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def _autofi_linear_probe(
    encoder: AutoFiCNN,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: str,
    seed: int,
) -> float:
    """Frozen-encoder linear probe for AutoFi (expects (B, A, S, T) input)."""
    from sklearn.linear_model import LogisticRegression

    encoder.eval().to(device)

    def _features(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        feats: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for x, y in loader:
            x = x.to(device).float().permute(0, 3, 2, 1).contiguous()
            h = encoder(x)
            feats.append(h.cpu().numpy())
            labels.append(np.asarray(y))
        return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)

    tr_x, tr_y = _features(train_loader)
    te_x, te_y = _features(test_loader)
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(tr_x, tr_y)
    return float(np.mean(clf.predict(te_x) == te_y))


@torch.no_grad()
def _capc_linear_probe(
    branch: CAPCBranch,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: str,
    seed: int,
    n_f: int,
) -> float:
    """Frozen-encoder linear probe for CAPC.

    Per §IV-B "linear classifier C_phi is fine-tuned with labelled CSI
    based on the concatenated representations from all windows generated
    by the pre-trained encoder E_theta". So we encode every window and
    concatenate the per-window embeddings as the feature vector.
    """
    from sklearn.linear_model import LogisticRegression
    from .capc import _split_windows, _to_capc_input

    branch.eval().to(device)

    def _features(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        feats: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for x, y in loader:
            x = x.to(device).float()
            windows = _split_windows(_to_capc_input(x), n_f)
            z = branch.encode_windows(windows)
            feats.append(z.flatten(1).cpu().numpy())
            labels.append(np.asarray(y))
        return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)

    tr_x, tr_y = _features(train_loader)
    te_x, te_y = _features(test_loader)
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(tr_x, tr_y)
    return float(np.mean(clf.predict(te_x) == te_y))


def _make_datasets(
    real: bool, seed: int, data_root: str, cache_dir: str
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if real:
        train_ds = Widar3CrossSubject(
            root=data_root, train=True, cache_path=f"{cache_dir}/josiah_train.pt"
        )
        test_ds = Widar3CrossSubject(
            root=data_root, train=False, cache_path=f"{cache_dir}/josiah_test.pt"
        )
        print(
            f"[josiah] real Widar3.0 cross-subject; "
            f"train={len(train_ds)}, test={len(test_ds)}"
        )
    else:
        train_ds = StubCSI(num_samples=10, seed=seed)
        test_ds = StubCSI(num_samples=10, seed=seed + 1)
        print(f"[josiah] stub data; train={len(train_ds)}, test={len(test_ds)}")
    return train_ds, test_ds


def main(
    mode: str = "supervised",
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 4,
    real: bool = False,
    data_root: str = "data/widar3/raw",
    cache_dir: str = "data/widar3/cache",
) -> float:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, test_ds = _make_datasets(real, seed, data_root, cache_dir)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=(mode != "supervised")
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if mode == "supervised":
        model = SupervisedClassifier(
            in_channels=CSI_S * CSI_A, num_classes=NUM_CLASSES, feature_dim=128
        )
        print(f"[josiah] supervised model params: {count_parameters(model)}")
        losses = train_supervised(
            model, train_loader, epochs=epochs, lr=1e-3, device=device
        )
        print(f"[josiah] train losses: {[round(loss, 4) for loss in losses]}")
        acc = evaluate(model, test_loader, device=device)
        print(f"[josiah] supervised top-1 accuracy: {acc:.3f}")
        return acc

    if mode == "capc":
        # T5.5: CAPC exact reproduction (Barahimi et al. 2024). Two-branch
        # encoder + GRU + per-step bilinear predictors; hybrid loss
        # L = L_BT + beta*(L_CPC^A + L_CPC^B), beta=50. LARS optimiser.
        # The paper's `dual view` augmentation needs uplink + downlink CSI;
        # Widar3.0 ships only one direction, so we use the documented
        # CAPC* fallback (noise + subcarrier mask) — Table I row.
        branch_a = CAPCBranch(in_channels=CSI_A, s=CSI_S, n_f=10,
                              embedding_dim=128, hidden_dim=128, future_steps=9)
        branch_b = CAPCBranch(in_channels=CSI_A, s=CSI_S, n_f=10,
                              embedding_dim=128, hidden_dim=128, future_steps=9)
        params = count_parameters(branch_a) + count_parameters(branch_b)
        print(f"[T5.5] CAPC twin-branch params: {params}")
        losses = pretrain_capc(
            branch_a, branch_b, train_loader,
            epochs=epochs, n_f=10, beta=50.0, bt_lambda=0.002, device=device,
        )
        print(f"[T5.5] CAPC pre-train losses: {[round(loss, 4) for loss in losses]}")
        probe_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        acc = _capc_linear_probe(
            branch_a, probe_train_loader, test_loader, device=device, seed=seed, n_f=10
        )
        print(f"[T5.5] CAPC linear-probe accuracy: {acc:.3f}")
        return acc

    if mode == "autofi":
        # T5.4: AutoFi exact reproduction (Yang et al. 2022). Twin-branch GSS
        # pre-training on (A=3, S=30, T=CSI_T) Conv2d inputs, then linear-probe
        # the first feature extractor on the labelled set. Paper uses SGD
        # lr=0.01 momentum=0.9 batch 128 for 300 epochs; CLI defaults stay
        # small for the smoke loop.
        gss = AutoFiGSS(in_channels=CSI_A, s=CSI_S, t=CSI_T, feature_dim=128,
                        num_bins=NUM_CLASSES)
        print(f"[T5.4] GSS params: {count_parameters(gss)}")
        losses = pretrain_autofi(
            gss,
            train_loader,
            epochs=epochs,
            lr=0.01,
            momentum=0.9,
            sigma=0.05,
            lam=1.0,
            gamma=1000.0,
            device=device,
        )
        print(f"[T5.4] GSS pre-train losses: {[round(loss, 4) for loss in losses]}")

        probe_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        acc = _autofi_linear_probe(
            gss.encoder1, probe_train_loader, test_loader, device=device, seed=seed
        )
        print(f"[T5.4] AutoFi linear-probe accuracy: {acc:.3f}")
        return acc

    if mode in ("simclr-trivial", "simclr-handcrafted"):
        tag = "T5.3" if mode == "simclr-trivial" else "T5.6"
        augment_fn = random_crop if mode == "simclr-trivial" else gaussian_then_mask
        encoder = TinyCNN(in_channels=CSI_S * CSI_A, feature_dim=128)
        print(f"[{tag}] encoder params: {count_parameters(encoder)}")
        ssl_model = SimCLR(encoder, feature_dim=128, projection_dim=64)
        losses = pretrain_simclr(
            ssl_model,
            train_loader,
            epochs=epochs,
            lr=1e-3,
            temperature=0.5,
            augment_fn=augment_fn,
            device=device,
        )
        print(f"[{tag}] SSL pre-train losses: {[round(loss, 4) for loss in losses]}")

        # Linear probe on the frozen encoder
        probe_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        acc = linear_probe(
            ssl_model.encoder,
            probe_train_loader,
            test_loader,
            device=device,
            seed=seed,
        )
        print(f"[{tag}] linear-probe accuracy: {acc:.3f}")
        return acc

    raise ValueError(f"unknown mode: {mode!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["supervised", "simclr-trivial", "simclr-handcrafted", "autofi", "capc"],
        default="supervised",
        help=(
            "Baseline mode: supervised (T5.1/T5.2), simclr-trivial (T5.3), "
            "simclr-handcrafted (T5.6, the comparison-column row), "
            "autofi (T5.4, exact reproduction of Yang et al. 2022), "
            "capc (T5.5, exact reproduction of Barahimi et al. 2024)."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--real",
        action="store_true",
        help="Use real Widar3.0 data; default is stub.",
    )
    p.add_argument("--data-root", default="data/widar3/raw")
    p.add_argument("--cache-dir", default="data/widar3/cache")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        mode=args.mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        real=args.real,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
    )
