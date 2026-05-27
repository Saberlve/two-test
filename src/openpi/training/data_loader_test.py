import dataclasses
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


class _DummyFrameDataset:
    def __init__(self, episode_lengths: list[int]):
        self.episode_data_index = {"from": [], "to": []}
        self._frames = []
        start = 0
        for episode_id, length in enumerate(episode_lengths):
            for pos in range(length):
                self._frames.append({"payload": np.asarray(f"{episode_id}:{pos}")})
            end = start + length
            self.episode_data_index["from"].append(start)
            self.episode_data_index["to"].append(end)
            start = end
        self.episode_data_index = {
            key: np.asarray(value, dtype=np.int64) for key, value in self.episode_data_index.items()
        }

    def __getitem__(self, index):
        return self._frames[int(index)]

    def __len__(self):
        return len(self._frames)


def test_episode_stream_iterable_dataset_shards_episodes_by_rank():
    dataset = _DummyFrameDataset([2, 2, 2, 2])
    episode_ranges = _data_loader._extract_episode_ranges(dataset)

    rank0 = _data_loader.EpisodeStreamIterableDataset(
        dataset, episode_ranges, num_replicas=2, rank=0, seed=1, infinite=False
    )
    rank1 = _data_loader.EpisodeStreamIterableDataset(
        dataset, episode_ranges, num_replicas=2, rank=1, seed=1, infinite=False
    )

    assert [int(sample["episode_id"]) for sample in rank0] == [0, 0, 2, 2]
    assert [int(sample["episode_id"]) for sample in rank1] == [1, 1, 3, 3]


def test_episode_stream_iterable_dataset_emits_sequential_episode_pos():
    dataset = _DummyFrameDataset([2, 3])
    stream = _data_loader.EpisodeStreamIterableDataset(
        dataset, _data_loader._extract_episode_ranges(dataset), seed=1, infinite=False
    )

    assert [(int(s["episode_id"]), int(s["episode_pos"]), int(s["stream_id"])) for s in stream] == [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 0),
    ]


def test_episode_stream_iterable_dataset_shards_workers_without_overlap(monkeypatch):
    dataset = _DummyFrameDataset([1, 1, 1, 1])
    stream = _data_loader.EpisodeStreamIterableDataset(
        dataset, _data_loader._extract_episode_ranges(dataset), seed=1, infinite=False
    )

    worker0 = SimpleNamespace(id=0, num_workers=2)
    worker1 = SimpleNamespace(id=1, num_workers=2)

    monkeypatch.setattr(_data_loader.torch.utils.data, "get_worker_info", lambda: worker0)
    worker0_episode_ids = [int(sample["episode_id"]) for sample in stream]

    monkeypatch.setattr(_data_loader.torch.utils.data, "get_worker_info", lambda: worker1)
    worker1_episode_ids = [int(sample["episode_id"]) for sample in stream]

    assert sorted(worker0_episode_ids + worker1_episode_ids) == [0, 1, 2, 3]
    assert set(worker0_episode_ids).isdisjoint(worker1_episode_ids)


def test_observation_from_dict_preserves_stream_metadata():
    data = {
        "image": {"base_0_rgb": np.zeros((2, 4, 4, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.array([True, True])},
        "image_features": {"base_0_rgb": np.zeros((2, 5, 8), dtype=np.float32)},
        "state": np.zeros((2, 1), dtype=np.float32),
        "episode_id": np.array([7, 8], dtype=np.int32),
        "episode_pos": np.array([3, 4], dtype=np.int32),
        "stream_id": np.array([1, 5], dtype=np.int32),
    }

    observation = _model.Observation.from_dict(data)
    processed = _model.preprocess_observation(None, observation, train=False, image_keys=("base_0_rgb",))

    assert np.array_equal(processed.episode_id, data["episode_id"])
    assert np.array_equal(processed.episode_pos, data["episode_pos"])
    assert np.array_equal(processed.stream_id, data["stream_id"])
    assert np.array_equal(processed.image_features["base_0_rgb"], data["image_features"]["base_0_rgb"])


def test_vision_feature_sidecar_dataset_loads_frame_features(tmp_path):
    dataset = _DummyFrameDataset([2])
    feature_path = _data_loader.resolve_vision_feature_episode_path(tmp_path, "unit", 0)
    feature_path.parent.mkdir(parents=True)
    features = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    np.savez_compressed(feature_path, base_0_rgb=features)

    wrapped = _data_loader.VisionFeatureSidecarDataset(
        dataset,
        _data_loader._extract_episode_ranges(dataset),
        vision_features_root=tmp_path,
        feature_id="unit",
    )

    sample = wrapped[1]
    assert np.array_equal(sample["image_features"]["base_0_rgb"], features[1])


def test_vision_feature_sidecar_dataset_fails_on_missing_episode(tmp_path):
    dataset = _DummyFrameDataset([1])

    with pytest.raises(FileNotFoundError, match="Missing precomputed vision feature sidecar"):
        _data_loader.VisionFeatureSidecarDataset(
            dataset,
            _data_loader._extract_episode_ranges(dataset),
            vision_features_root=tmp_path,
            feature_id="missing",
        )


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
