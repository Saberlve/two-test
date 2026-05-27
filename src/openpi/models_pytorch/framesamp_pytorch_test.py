from types import SimpleNamespace

import torch

from openpi.models_pytorch.framesamp_pytorch import PI0FramesampContextPytorch


class _DummyPaliGemma:
    def embed_image(self, image):
        bsize = image.shape[0]
        values = image.flatten(start_dim=1).mean(dim=1)
        tokens = torch.arange(4 * 8, dtype=image.dtype, device=image.device).reshape(1, 4, 8)
        return tokens.expand(bsize, -1, -1) + values[:, None, None]


def _make_model():
    model = object.__new__(PI0FramesampContextPytorch)
    torch.nn.Module.__init__(model)
    model.context_window = 3
    model.context_image_keys = None
    model.context_pos_dim = 768
    model.context_use_pos_emb = False
    model.context_use_state_emb = False
    model.context_pool_type = "mean"
    model.token_per_image = 4
    model.context_width = 8
    model.context_pos_proj = None
    model.context_state_proj = None
    model.context_encoder_static = torch.nn.Linear(8, 8)
    torch.nn.init.eye_(model.context_encoder_static.weight)
    torch.nn.init.zeros_(model.context_encoder_static.bias)
    model.context_pad_token = torch.nn.Parameter(torch.zeros(8))
    model.paligemma_with_expert = _DummyPaliGemma()
    model._history = {}
    model._history_episode = {}
    model._pending_context = None
    return model


def _obs(episode_id, episode_pos, stream_id=0):
    return SimpleNamespace(
        episode_id=torch.tensor([episode_id]),
        episode_pos=torch.tensor([episode_pos]),
        stream_id=torch.tensor([stream_id]),
    )


def test_framesamp_context_right_pads_and_resets_on_new_episode():
    model = _make_model()
    images = [torch.zeros(1, 3, 8, 8)]
    masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.zeros(1, 4)

    context, mask, _ = model._build_context_tokens(_obs(7, 0), images, masks, state)
    assert context.shape == (1, 12, 8)
    assert mask.tolist() == [[True, True, True, True, False, False, False, False, False, False, False, False]]

    model._build_context_tokens(_obs(7, 1), [torch.ones(1, 3, 8, 8)], masks, state)
    _, mask, _ = model._build_context_tokens(_obs(8, 0), images, masks, state)
    assert mask.tolist() == [[True, True, True, True, False, False, False, False, False, False, False, False]]
    assert len(model._history[0]) == 1
    assert model._history_episode[0] == 8


def test_framesamp_context_uniformly_samples_full_history_and_isolates_streams():
    model = _make_model()
    masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.zeros(1, 4)
    for pos in range(5):
        image = torch.full((1, 3, 8, 8), float(pos))
        model._build_context_tokens(_obs(3, pos, stream_id=0), [image], masks, state)

    image = torch.full((1, 3, 8, 8), 5.0)
    context, mask, _ = model._build_context_tokens(_obs(3, 5, stream_id=0), [image], masks, state)
    # history after appending current = [0,1,2,3,4,5]; linspace(0,5,3) -> indices [0,2,5]
    assert context.shape == (1, 12, 8)
    assert mask.tolist() == [[True] * 12]
    # base token at position 0 of each sampled frame is value 0 + frame mean (0, 2, 5)
    assert context[0, 0, 0].item() == 0.0
    assert context[0, 4, 0].item() == 2.0
    assert context[0, 8, 0].item() == 5.0

    _, other_mask, _ = model._build_context_tokens(_obs(3, 1, stream_id=1), [image], masks, state)
    # new stream starts fresh: only current frame, rest right-padded
    assert other_mask.tolist() == [[True, True, True, True, False, False, False, False, False, False, False, False]]


def test_framesamp_context_uses_precomputed_features_without_vision_forward():
    model = _make_model()
    masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.zeros(1, 4)
    image = torch.zeros(1, 3, 8, 8)
    image_features = [torch.ones(1, 4, 8)]

    model._build_context_tokens(_obs(4, 0), [image], masks, state, image_features)
    model.paligemma_with_expert.embed_image = lambda image: (_ for _ in ()).throw(AssertionError("unexpected"))

    context, mask, _ = model._build_context_tokens(_obs(4, 1), [image], masks, state, image_features)

    assert context.shape == (1, 12, 8)
    # two cached frames (pos 0, pos 1) -> 8 valid tokens, then 4 right-pad
    assert mask.tolist() == [[True, True, True, True, True, True, True, True, False, False, False, False]]


def test_framesamp_context_includes_current_frame_and_right_pads():
    model = _make_model()
    model.context_window = 2

    image = torch.zeros(1, 3, 8, 8)
    masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.zeros(1, 4)

    features0 = [torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8)]
    context, mask, _ = model._build_context_tokens(_obs(9, 0), [image], masks, state, features0)
    assert context.shape == (1, 8, 8)
    assert mask.tolist() == [[True, True, True, True, False, False, False, False]]
    assert torch.allclose(context[0, :4], features0[0][0])

    features1 = [torch.full((1, 4, 8), 2.0)]
    context, mask, _ = model._build_context_tokens(_obs(9, 1), [image], masks, state, features1)
    assert mask.tolist() == [[True, True, True, True, True, True, True, True]]
    assert torch.allclose(context[0, :4], features0[0][0])
    assert torch.allclose(context[0, 4:], features1[0][0])
