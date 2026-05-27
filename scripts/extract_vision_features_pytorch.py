#!/usr/bin/env python3
"""Precompute PyTorch PaliGemma image token embeddings for a LeRobot dataset."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib

import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import safetensors.torch
import torch

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.transforms as _transforms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="Training config name.")
    parser.add_argument("--dataset-root", required=True, help="Local LeRobot dataset root.")
    parser.add_argument("--output-root", required=True, help="Root where vision_features/{feature_id} is written.")
    parser.add_argument("--feature-id", required=True, help="Feature version/name.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--episode-start", type=int, default=0, help="First episode id to extract, inclusive.")
    parser.add_argument("--episode-end", type=int, default=None, help="Last episode id to extract, exclusive.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional smoke-test cap on frames per selected episode/range.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="Overwrite existing episode sidecars.")
    mode.add_argument("--resume", action="store_true", help="Skip existing episode sidecars.")
    return parser.parse_args()


def _make_model(config: _config.TrainConfig, device: torch.device) -> pi0_pytorch.PI0Pytorch:
    if not isinstance(config.model, pi0_config.Pi0Config):
        raise TypeError(f"Expected a Pi0Config-compatible model config, got {type(config.model).__name__}")
    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    object.__setattr__(model_cfg, "pytorch_compile_mode", None)
    model = pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    if config.pytorch_weight_path is None:
        raise ValueError(f"{config.name} does not define pytorch_weight_path.")
    model_path = pathlib.Path(config.pytorch_weight_path) / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"PyTorch weight file not found: {model_path}")
    safetensors.torch.load_model(model, str(model_path))
    model.eval()
    return model


def _collate(samples: list[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *samples)


def _episode_output_path(output_root: pathlib.Path, feature_id: str, episode_id: int) -> pathlib.Path:
    return _data.resolve_vision_feature_episode_path(output_root, feature_id, episode_id)


def _get_total_episodes(data_config: _config.DataConfig) -> int:
    if data_config.repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=data_config.dataset_root)
    return int(dataset_meta.total_episodes)


def _make_raw_dataset_for_episodes(
    data_config: _config.DataConfig,
    action_horizon: int,
    episode_ids: list[int],
) -> tuple[_data.Dataset, list[tuple[int, int]]]:
    if data_config.repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=data_config.dataset_root)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=data_config.dataset_root,
        episodes=episode_ids,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    selected_ranges: list[tuple[int, int]] = []
    starts = [0] * dataset_meta.total_episodes
    ends = [0] * dataset_meta.total_episodes
    cursor = 0
    for episode_id in episode_ids:
        length = int(dataset_meta.episodes[episode_id]["length"])
        starts[episode_id] = cursor
        cursor += length
        ends[episode_id] = cursor
        selected_ranges.append((starts[episode_id], ends[episode_id]))

    # LeRobot keeps original episode_index values in subset rows, so keep the
    # boundary tensors indexable by original episode id.
    dataset.episode_data_index = {
        "from": torch.LongTensor(starts),
        "to": torch.LongTensor(ends),
    }

    if data_config.prompt_from_task:
        dataset = _data.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset, selected_ranges


@torch.no_grad()
def _extract_episode(
    model: pi0_pytorch.PI0Pytorch,
    dataset: _data.Dataset,
    start: int,
    end: int,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    chunks_by_key: dict[str, list[np.ndarray]] = {}
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        samples = [dataset[index] for index in range(batch_start, batch_end)]
        batch = _collate(samples)
        observation = _model.Observation.from_dict(batch)
        observation = jax.tree.map(lambda x: torch.as_tensor(x, device=device), observation)
        images, _, _, _, _, _ = model._preprocess_observation(observation, train=False)  # noqa: SLF001

        for key, image in zip(observation.images, images, strict=True):
            features = model.paligemma_with_expert.embed_image(image)
            feature_np = features.detach().to(dtype=torch.float32, device="cpu").numpy()
            chunks_by_key.setdefault(key, []).append(feature_np)

    return {key: np.concatenate(chunks, axis=0) for key, chunks in chunks_by_key.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.episode_start < 0:
        raise ValueError("--episode-start must be >= 0")
    if args.episode_end is not None and args.episode_end < args.episode_start:
        raise ValueError("--episode-end must be >= --episode-start")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be >= 1")

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    data_config = dataclasses.replace(
        data_config,
        dataset_root=args.dataset_root,
        use_precomputed_vision_features=False,
    )

    total_episodes = _get_total_episodes(data_config)
    episode_end = total_episodes if args.episode_end is None else args.episode_end
    if args.episode_start > total_episodes:
        raise ValueError(
            f"--episode-start {args.episode_start} exceeds dataset episode count {total_episodes}"
        )
    if episode_end > total_episodes:
        raise ValueError(f"--episode-end {episode_end} exceeds dataset episode count {total_episodes}")

    episode_ids = list(range(args.episode_start, episode_end))
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]
    if not episode_ids:
        logging.info("No episodes selected from dataset with %d episodes.", total_episodes)
        return

    raw_dataset, selected_episode_ranges = _make_raw_dataset_for_episodes(
        data_config, config.model.action_horizon, episode_ids
    )
    dataset = _data.transform_dataset(raw_dataset, data_config, skip_norm_stats=True)
    episode_ranges = list(zip(episode_ids, selected_episode_ranges, strict=True))
    if args.max_frames is not None:
        episode_ranges = [
            (episode_id, (start, min(end, start + args.max_frames)))
            for episode_id, (start, end) in episode_ranges
        ]

    device = torch.device(args.device)
    model = _make_model(config, device)
    output_root = pathlib.Path(args.output_root)

    logging.info(
        "Extracting %d episodes [%d, %d) to %s",
        len(episode_ranges),
        args.episode_start,
        episode_end,
        output_root,
    )
    for episode_id, (start, end) in episode_ranges:
        output_path = _episode_output_path(output_root, args.feature_id, episode_id)
        if output_path.exists():
            if args.resume:
                logging.info("Skipping existing sidecar: %s", output_path)
                continue
            if not args.overwrite:
                raise FileExistsError(f"Sidecar already exists: {output_path}. Use --overwrite or --resume.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        features = _extract_episode(
            model,
            dataset,
            start,
            end,
            batch_size=args.batch_size,
            device=device,
        )
        tmp_path = output_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            **features,
            __episode_id=np.asarray(episode_id, dtype=np.int64),
            __num_frames=np.asarray(end - start, dtype=np.int64),
        )
        os.replace(tmp_path, output_path)
        logging.info("Wrote %s (%d frames)", output_path, end - start)


if __name__ == "__main__":
    main()
