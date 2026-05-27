from types import SimpleNamespace

import torch

from openpi.models_pytorch.rmt_pytorch import PI0RMTContextPytorch
from openpi.models_pytorch.rmt_pytorch import RecurrentMemory


class _DummyPaliGemma:
    def embed_image(self, image):
        return torch.ones(image.shape[0], 5, 8, dtype=image.dtype, device=image.device)


class _DummyMemory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))

    def reset(self, batch_size, *, device, dtype):
        return torch.zeros(batch_size, 3, 8, device=device, dtype=dtype)

    def forward(self, image_embs, image_masks, state, memory_state):
        del image_embs, image_masks
        delta = state.mean(dim=-1)[:, None, None]
        next_state = memory_state + delta + 1.0
        return next_state, next_state


def _obs(episode_id, episode_pos, stream_id=0):
    return SimpleNamespace(
        episode_id=torch.tensor([episode_id]),
        episode_pos=torch.tensor([episode_pos]),
        stream_id=torch.tensor([stream_id]),
    )


def _make_context_model():
    model = object.__new__(PI0RMTContextPytorch)
    torch.nn.Module.__init__(model)
    model.rmt_memory = _DummyMemory()
    model.paligemma_with_expert = _DummyPaliGemma()
    model._memory_state_by_stream = {}
    model._episode_by_stream = {}
    model._pending_context = None
    return model


def test_recurrent_memory_updates_state_and_returns_prefix_tokens():
    memory = RecurrentMemory(
        image_dim=8,
        state_dim=4,
        prefix_dim=8,
        hidden_dim=16,
        budget=3,
        token_per_image=2,
        num_attn_heads=4,
        num_kv_heads=2,
        mini_batch_size=2,
    )
    image_embs = [torch.randn(2, 5, 8), torch.randn(2, 5, 8)]
    image_masks = [torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)]
    state = torch.randn(2, 4)
    initial = memory.reset(2, device=state.device, dtype=state.dtype)

    context, updated = memory(image_embs, image_masks, state, initial)

    assert context.shape == (2, 3, 8)
    assert updated.shape == initial.shape
    assert not torch.allclose(updated, initial)


def test_rmt_context_resets_on_episode_start_and_isolates_streams():
    model = _make_context_model()
    images = [torch.zeros(1, 3, 8, 8)]
    masks = [torch.ones(1, dtype=torch.bool)]

    context, mask, _ = model._build_rmt_context(_obs(2, 0, stream_id=0), images, masks, torch.ones(1, 4))
    assert context.shape == (1, 3, 8)
    assert mask.tolist() == [[True, True, True]]
    assert torch.allclose(model._memory_state_by_stream[0], torch.full((3, 8), 2.0))

    model._build_rmt_context(_obs(2, 1, stream_id=0), images, masks, torch.full((1, 4), 2.0))
    assert torch.allclose(model._memory_state_by_stream[0], torch.full((3, 8), 5.0))

    model._build_rmt_context(_obs(3, 0, stream_id=0), images, masks, torch.full((1, 4), 5.0))
    assert torch.allclose(model._memory_state_by_stream[0], torch.full((3, 8), 6.0))

    model._build_rmt_context(_obs(2, 1, stream_id=1), images, masks, torch.full((1, 4), 1.0))
    assert torch.allclose(model._memory_state_by_stream[1], torch.full((3, 8), 2.0))


def test_rmt_context_uses_precomputed_features_without_vision_forward():
    model = _make_context_model()
    model.paligemma_with_expert.embed_image = lambda image: (_ for _ in ()).throw(AssertionError("unexpected"))
    images = [torch.zeros(1, 3, 8, 8)]
    masks = [torch.ones(1, dtype=torch.bool)]
    image_features = [torch.ones(1, 5, 8)]

    context, mask, _ = model._build_rmt_context(
        _obs(2, 0, stream_id=0),
        images,
        masks,
        torch.ones(1, 4),
        image_features,
    )

    assert context.shape == (1, 3, 8)
    assert mask.tolist() == [[True, True, True]]
