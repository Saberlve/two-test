from types import SimpleNamespace

import torch

from openpi.models_pytorch.rmt_pytorch import (
    PI0RMTContextPytorch,
    RecurrentMemory,
    _get_token_recurrent_indices,
)


class _DummyPaliGemma:
    def embed_image(self, image):
        bsize = image.shape[0]
        values = image.flatten(start_dim=1).mean(dim=1)
        tokens = torch.arange(4 * 8, dtype=image.dtype, device=image.device).reshape(1, 4, 8)
        return tokens.expand(bsize, -1, -1) + values[:, None, None]


def _obs(episode_id, episode_pos, stream_id=0):
    return SimpleNamespace(
        episode_id=torch.tensor([episode_id]),
        episode_pos=torch.tensor([episode_pos]),
        stream_id=torch.tensor([stream_id]),
    )


def _make_rmt_memory():
    return RecurrentMemory(
        image_input_dim=8,
        pos_input_dim=12,
        pos_output_dim=4,
        state_input_dim=4,
        state_output_dim=4,
        hidden_dim=16,
        output_dim=8,
        budget=3,
        max_recur_steps=4,
        tokens_per_frame=4,
        num_attn_heads=4,
        num_kv_heads=2,
        use_pos_emb=False,
        use_state_emb=False,
    )


def _make_context_model():
    model = object.__new__(PI0RMTContextPytorch)
    torch.nn.Module.__init__(model)
    model.input_obs_horizon = 2
    model.max_recur_steps = 4
    model.token_per_image = 4
    model.num_views = 1
    model.context_image_keys = None
    model.context_pos_dim = 12
    model.context_pool_type = "mean"
    model.use_pos_emb = False
    model.use_state_emb = False
    model.rmt_memory = _make_rmt_memory()
    model.paligemma_with_expert = _DummyPaliGemma()
    model._history = {}
    model._history_episode = {}
    model._pending_context = None
    return model


def test_get_token_recurrent_indices_below_horizon():
    # step_idx < horizon -> only current step
    assert _get_token_recurrent_indices(0, 4, 10) == [0]
    assert _get_token_recurrent_indices(3, 4, 10) == [3]


def test_get_token_recurrent_indices_strided_above_horizon():
    # step_idx=10, horizon=4 -> start=10%4=2, range(2, 11, 4) = [2, 6, 10]
    assert _get_token_recurrent_indices(10, 4, 10) == [2, 6, 10]


def test_get_token_recurrent_indices_caps_to_max_recur_steps():
    # Many strided indices clipped to last `max_recur_steps`
    assert _get_token_recurrent_indices(100, 4, 3) == [92, 96, 100]


def test_recurrent_memory_updates_state_via_scan():
    memory = _make_rmt_memory()
    b, t, v, p = 2, 3, 1, 4
    recur_image_emb = torch.randn(b, t, v, p, 8)
    recur_mask = torch.ones(b, t, dtype=torch.bool)

    context, mask = memory(recur_image_emb, recur_mask, None, None)

    assert context.shape == (b, memory.budget, 8)
    assert mask.shape == (b, memory.budget)
    assert mask.all()


def test_recurrent_memory_skips_masked_frames():
    """With mask all-False the memory state should be unchanged from init."""
    memory = _make_rmt_memory()
    b, t, v, p = 1, 3, 1, 4
    recur_image_emb = torch.randn(b, t, v, p, 8)
    recur_mask = torch.zeros(b, t, dtype=torch.bool)

    context, _ = memory(recur_image_emb, recur_mask, None, None)
    expected = memory.proj(memory.recur_layer.memory_state.detach())
    assert torch.allclose(context[0], expected, atol=1e-5)


def test_rmt_context_left_pads_history_and_returns_budget_tokens():
    model = _make_context_model()
    images = [torch.zeros(1, 3, 8, 8)]
    masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.zeros(1, 4)

    context, pad_mask, att_mask = model._build_rmt_context(_obs(7, 0), images, masks, state)
    assert context.shape == (1, 3, 8)  # budget=3, output_dim=8
    assert pad_mask.shape == (1, 3) and pad_mask.all()
    assert att_mask.shape == (1, 3) and not att_mask.any()


def test_rmt_context_resets_on_new_episode_and_isolates_streams():
    model = _make_context_model()
    images = [torch.zeros(1, 3, 8, 8)]
    masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.zeros(1, 4)

    model._build_rmt_context(_obs(2, 0, stream_id=0), images, masks, state)
    model._build_rmt_context(_obs(2, 1, stream_id=0), images, masks, state)
    assert len(model._history[0]) == 2

    # New episode on same stream -> history cleared, then current frame appended
    model._build_rmt_context(_obs(3, 0, stream_id=0), images, masks, state)
    assert len(model._history[0]) == 1
    assert model._history_episode[0] == 3

    # Different stream -> isolated history
    model._build_rmt_context(_obs(2, 0, stream_id=1), images, masks, state)
    assert len(model._history[1]) == 1
    assert len(model._history[0]) == 1


def test_rmt_context_uses_precomputed_features_without_vision_forward():
    model = _make_context_model()
    model.paligemma_with_expert.embed_image = lambda image: (_ for _ in ()).throw(AssertionError("unexpected"))
    images = [torch.zeros(1, 3, 8, 8)]
    masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.zeros(1, 4)
    image_features = [torch.ones(1, 4, 8)]

    context, mask, _ = model._build_rmt_context(_obs(2, 0), images, masks, state, image_features)
    assert context.shape == (1, 3, 8)
    assert mask.all()


def test_rmt_context_samples_at_obs_horizon_stride():
    """At step 5 with horizon=2, indices = [5%2, 2, 4] = [1, 3, 5]."""
    model = _make_context_model()
    images = [torch.zeros(1, 3, 8, 8)]
    masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.zeros(1, 4)

    # Build up 6 frames (positions 0..5)
    for pos in range(6):
        model._build_rmt_context(_obs(0, pos), images, masks, state)

    indices = _get_token_recurrent_indices(5, 2, 4)
    assert indices == [1, 3, 5]
