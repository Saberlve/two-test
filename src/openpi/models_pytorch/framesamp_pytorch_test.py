from types import SimpleNamespace

import torch

from openpi.models_pytorch.framesamp_pytorch import PI0FramesampContextPytorch


class _DummyPaliGemma:
    def embed_image(self, image):
        bsize = image.shape[0]
        values = image.flatten(start_dim=1).mean(dim=1)
        tokens = torch.arange(5 * 8, dtype=image.dtype, device=image.device).reshape(1, 5, 8)
        return tokens.expand(bsize, -1, -1) + values[:, None, None]


def _make_model():
    model = object.__new__(PI0FramesampContextPytorch)
    torch.nn.Module.__init__(model)
    model.context_window = 3
    model.frame_sample_stride = 2
    model.token_per_image = 2
    model.context_budget = 6
    model.context_state_proj = torch.nn.Linear(4, 8)
    model.context_pos_embedding = torch.nn.Parameter(torch.zeros(3, 8))
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


def test_framesamp_context_left_pads_and_resets_on_new_episode():
    model = _make_model()
    images = [torch.zeros(1, 3, 8, 8)]
    masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.zeros(1, 4)

    context, mask, _ = model._build_context_tokens(_obs(7, 0), images, masks, state)
    assert context.shape == (1, 6, 8)
    assert mask.tolist() == [[False, False, False, False, False, False]]

    model._build_context_tokens(_obs(7, 1), [torch.ones(1, 3, 8, 8)], masks, state)
    _, mask, _ = model._build_context_tokens(_obs(8, 0), images, masks, state)
    assert mask.tolist() == [[False, False, False, False, False, False]]
    assert len(model._history[0]) == 1
    assert model._history_episode[0] == 8


def test_framesamp_context_uses_stride_and_isolates_streams():
    model = _make_model()
    masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.zeros(1, 4)
    for pos in range(5):
        image = torch.full((1, 3, 8, 8), float(pos))
        model._build_context_tokens(_obs(3, pos, stream_id=0), [image], masks, state)

    image = torch.full((1, 3, 8, 8), 5.0)
    _, mask, _ = model._build_context_tokens(_obs(3, 5, stream_id=0), [image], masks, state)
    assert mask.tolist() == [[True, True, True, True, True, True]]

    _, other_mask, _ = model._build_context_tokens(_obs(3, 1, stream_id=1), [image], masks, state)
    assert other_mask.tolist() == [[False, False, False, False, False, False]]
