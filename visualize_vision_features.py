#!/usr/bin/env python3
"""Visualize precomputed openpi vision feature sidecars.

Example:
    python visualize_vision_features.py --episode-id 0 --frames 0 100 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FEATURE_ROOT = Path(
    "/run/determined/NAS1/public/wangshuxun/rmbench_lerobot_data/"
    "rmbench_battery_swap_repo/vision_features/pi05_base_pytorch"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--episode-id", type=int, default=None, help="Episode id. Defaults to first completed npz.")
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Frame indices inside the episode.")
    parser.add_argument("--output-dir", type=Path, default=Path("vision_feature_visualizations"))
    parser.add_argument("--max-pca-fit-tokens", type=int, default=4096)
    return parser.parse_args()


def _episode_path(feature_root: Path, episode_id: int) -> Path:
    chunk_id = episode_id // 1000
    return feature_root / f"chunk-{chunk_id:03d}" / f"episode_{episode_id:06d}.npz"


def _find_first_episode(feature_root: Path) -> Path:
    matches = sorted(feature_root.glob("chunk-*/episode_*.npz"))
    if not matches:
        raise FileNotFoundError(f"No completed episode_*.npz files under {feature_root}")
    return matches[0]


def _infer_grid(num_tokens: int) -> tuple[int, int]:
    side = int(round(num_tokens**0.5))
    if side * side == num_tokens:
        return side, side
    for h in range(side, 0, -1):
        if num_tokens % h == 0:
            return h, num_tokens // h
    return 1, num_tokens


def _fit_pca_rgb(features_by_view: dict[str, np.ndarray], max_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    samples = []
    for features in features_by_view.values():
        flat = features.reshape(-1, features.shape[-1])
        if flat.shape[0] > max_tokens:
            idx = np.linspace(0, flat.shape[0] - 1, max_tokens, dtype=np.int64)
            flat = flat[idx]
        samples.append(flat.astype(np.float32, copy=False))

    x = np.concatenate(samples, axis=0)
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:3].T
    return mean.squeeze(0), components


def _pca_to_rgb(tokens: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    rgb = (tokens.astype(np.float32) - mean) @ components
    lo = np.percentile(rgb, 1, axis=0, keepdims=True)
    hi = np.percentile(rgb, 99, axis=0, keepdims=True)
    rgb = (rgb - lo) / np.maximum(hi - lo, 1e-6)
    return np.clip(rgb, 0.0, 1.0)


def _select_frames(num_frames: int, requested: list[int] | None) -> list[int]:
    if requested:
        frames = requested
    else:
        frames = sorted(set([0, num_frames // 2, num_frames - 1]))
    bad = [idx for idx in frames if idx < 0 or idx >= num_frames]
    if bad:
        raise ValueError(f"Frame indices out of range for {num_frames} frames: {bad}")
    return frames


def _plot_episode(path: Path, output_dir: Path, frames: list[int] | None, max_pca_fit_tokens: int) -> Path:
    with np.load(path) as data:
        feature_keys = [key for key in data.files if not key.startswith("__")]
        features_by_view = {key: np.asarray(data[key]) for key in feature_keys}

    if not features_by_view:
        raise ValueError(f"No feature arrays found in {path}")

    num_frames = next(iter(features_by_view.values())).shape[0]
    selected_frames = _select_frames(num_frames, frames)
    mean, components = _fit_pca_rgb(features_by_view, max_pca_fit_tokens)

    num_views = len(feature_keys)
    fig, axes = plt.subplots(
        len(selected_frames) * 2,
        num_views,
        figsize=(4.0 * num_views, 3.2 * len(selected_frames) * 2),
        squeeze=False,
    )

    for frame_row, frame_idx in enumerate(selected_frames):
        for col, key in enumerate(feature_keys):
            tokens = features_by_view[key][frame_idx]
            h, w = _infer_grid(tokens.shape[0])

            norm_map = np.linalg.norm(tokens, axis=-1).reshape(h, w)
            ax = axes[frame_row * 2][col]
            im = ax.imshow(norm_map, cmap="magma")
            ax.set_title(f"{key} frame {frame_idx} norm")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            rgb = _pca_to_rgb(tokens, mean, components).reshape(h, w, 3)
            ax = axes[frame_row * 2 + 1][col]
            ax.imshow(rgb)
            ax.set_title(f"{key} frame {frame_idx} PCA RGB")
            ax.set_xticks([])
            ax.set_yticks([])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{path.stem}_features.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    args = _parse_args()
    if args.episode_id is None:
        episode_path = _find_first_episode(args.feature_root)
    else:
        episode_path = _episode_path(args.feature_root, args.episode_id)

    if not episode_path.is_file():
        raise FileNotFoundError(f"Episode sidecar not found: {episode_path}")

    out_path = _plot_episode(
        episode_path,
        args.output_dir,
        args.frames,
        args.max_pca_fit_tokens,
    )
    print(out_path)


if __name__ == "__main__":
    main()
