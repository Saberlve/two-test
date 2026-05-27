from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class _DummyPaliGemma(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(embed_tokens=torch.nn.Embedding(16, 8)),
        )

    def embed_image(self, image):
        raise AssertionError("embed_image should not be called when image_features are provided")

    def embed_language_tokens(self, tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], 8, device=tokens.device)


def test_embed_prefix_uses_precomputed_image_features_without_vision_forward():
    model = object.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.paligemma_with_expert = _DummyPaliGemma()
    model._apply_checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

    images = [torch.zeros(2, 3, 8, 8), torch.zeros(2, 3, 8, 8)]
    image_masks = [torch.ones(2, dtype=torch.bool), torch.tensor([True, False])]
    image_features = [torch.ones(2, 5, 8), torch.ones(2, 5, 8) * 2]
    lang_tokens = torch.zeros(2, 4, dtype=torch.long)
    lang_masks = torch.ones(2, 4, dtype=torch.bool)

    embs, pad_masks, att_masks = model.embed_prefix(images, image_masks, lang_tokens, lang_masks, image_features)

    assert embs.shape == (2, 14, 8)
    assert pad_masks.shape == (2, 14)
    assert att_masks.shape == (2, 14)
    assert torch.allclose(embs[:, :5], image_features[0])
    assert torch.allclose(embs[:, 5:10], image_features[1])
    assert pad_masks[:, :5].tolist() == [[True] * 5, [True] * 5]
    assert pad_masks[:, 5:10].tolist() == [[True] * 5, [False] * 5]


def test_embed_prefix_rejects_mismatched_feature_count():
    model = object.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.paligemma_with_expert = _DummyPaliGemma()

    with pytest.raises(ValueError, match="image_features length"):
        model.embed_prefix(
            [torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8)],
            [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)],
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.bool),
            [torch.ones(1, 5, 8)],
        )
