from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


@dataclass
class _CachedFrame:
    episode_id: int
    episode_pos: int
    images: list[torch.Tensor]
    image_masks: list[torch.Tensor]
    state: torch.Tensor
    image_features: list[torch.Tensor] | None = None


def _pool_tokens_to_size(tokens: torch.Tensor, target_size: int, pool_type: str) -> torch.Tensor:
    """Pool square image token grids to match RoboMME's 4x4/2x2 memory sidecars."""
    if tokens.ndim != 2:
        raise ValueError(f"Expected image tokens with shape (tokens, dim), got {tuple(tokens.shape)}")
    source_size = tokens.shape[0]
    if source_size == target_size:
        return tokens

    source_hw = int(math.sqrt(source_size))
    target_hw = int(math.sqrt(target_size))
    if source_hw * source_hw != source_size:
        raise ValueError(f"Image token count must be square, got {source_size}")
    if target_hw * target_hw != target_size:
        raise ValueError(f"target_size must be square, got {target_size}")
    if source_hw % target_hw != 0:
        raise ValueError(f"Cannot pool {source_hw}x{source_hw} tokens to {target_hw}x{target_hw}")

    pool_size = source_hw // target_hw
    grid = tokens.reshape(source_hw, source_hw, -1).permute(2, 0, 1)[None]
    if pool_type == "mean":
        pooled = F.avg_pool2d(grid, kernel_size=pool_size, stride=pool_size)
    elif pool_type == "max":
        pooled = F.max_pool2d(grid, kernel_size=pool_size, stride=pool_size)
    else:
        raise ValueError(f"Invalid context_pool_type: {pool_type!r}")
    return pooled[0].permute(1, 2, 0).reshape(target_size, -1)


def _posemb_3d(positions: torch.Tensor, spatial_size: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Torch equivalent of RoboMME's PosEmb3D for pooled 16x16 SigLIP tokens."""
    if dim % 6 != 0:
        raise ValueError(f"context_pos_dim must be divisible by 6, got {dim}")
    if spatial_size < 1 or 16 % spatial_size != 0:
        raise ValueError(f"spatial_size must divide 16, got {spatial_size}")

    device = positions.device
    width = dim // 6
    omega = torch.arange(width, device=device, dtype=torch.float32) / (width - 1)
    temporal_omega = 1.0 / (10_000**omega)
    spatial_omega = 1.0 / (1_000**omega)

    pos = positions.to(dtype=torch.float32)
    temporal = pos[:, None] * temporal_omega[None]
    temporal_pe = torch.cat([torch.sin(temporal), torch.cos(temporal)], dim=-1)
    temporal_pe = temporal_pe[:, None, :].expand(-1, spatial_size * spatial_size, -1)

    y, x = torch.meshgrid(
        torch.arange(spatial_size, device=device, dtype=torch.float32),
        torch.arange(spatial_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    stride = 16 // spatial_size
    offset = stride / 2.0
    y = (stride * y.flatten() + offset)[:, None] * spatial_omega[None]
    x = (stride * x.flatten() + offset)[:, None] * spatial_omega[None]
    spatial_pe = torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=-1)
    spatial_pe = spatial_pe[None].expand(positions.shape[0], -1, -1)
    return torch.cat([temporal_pe, spatial_pe], dim=-1).to(dtype=dtype)


class PI0FramesampContextPytorch(PI0Pytorch):
    """PI0/PI05 PyTorch baseline with frame-sampled prefix context tokens.

    Behavior matches RoboMME's `perceptual-framesamp-context`: the current frame
    is appended to history before sampling, uniform `linspace` sampling, right
    padding when history is shorter than `context_window`.
    """

    def __init__(self, config):
        super().__init__(config)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self.context_window = int(config.context_window)
        self.context_image_keys = None if config.context_image_keys is None else tuple(config.context_image_keys)
        self.context_pos_dim = int(config.context_pos_dim)
        self.context_use_pos_emb = bool(config.context_use_pos_emb)
        self.context_use_state_emb = bool(config.context_use_state_emb)
        self.context_pool_type = str(config.context_pool_type)
        self.token_per_image = int(config.token_per_image)
        self.context_width = paligemma_config.width
        self.context_pad_token = nn.Parameter(torch.zeros(paligemma_config.width))

        encoder_input_dim = paligemma_config.width
        if self.context_use_pos_emb:
            self.context_pos_proj = nn.Linear(self.context_pos_dim, int(config.context_pos_hidden_dim))
            encoder_input_dim += int(config.context_pos_hidden_dim)
        else:
            self.context_pos_proj = None
        if self.context_use_state_emb:
            self.context_state_proj = nn.Linear(config.action_dim, int(config.context_state_hidden_dim))
            encoder_input_dim += int(config.context_state_hidden_dim)
        else:
            self.context_state_proj = None
        self.context_encoder_static = nn.Linear(encoder_input_dim, paligemma_config.width)

        self._history: dict[int, list[_CachedFrame]] = {}
        self._history_episode: dict[int, int] = {}
        self._pending_context: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def reset_context_cache(self) -> None:
        self._history.clear()
        self._history_episode.clear()

    def _metadata(self, observation, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stream_id = observation.stream_id
        episode_id = observation.episode_id
        episode_pos = observation.episode_pos
        if stream_id is None:
            stream_id = torch.arange(batch_size, device=device, dtype=torch.long)
        if episode_id is None:
            episode_id = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        if episode_pos is None:
            episode_pos = torch.arange(batch_size, device=device, dtype=torch.long)
        return stream_id.to(device), episode_id.to(device), episode_pos.to(device)

    def _sample_history(self, history: list[_CachedFrame]) -> list[_CachedFrame]:
        max_frames = self.context_window
        if len(history) <= max_frames:
            return list(history)
        indices = torch.linspace(0, len(history) - 1, max_frames, dtype=torch.long).tolist()
        return [history[int(index)] for index in indices]

    def _context_token_count(self, num_images: int) -> int:
        return self.context_window * max(1, num_images) * self.token_per_image

    def _embed_context_image(self, image: torch.Tensor, image_feature: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
        if image_feature is None:
            image_emb = self.paligemma_with_expert.embed_image(image[None])[0]
        else:
            image_emb = image_feature.to(device=image.device, dtype=dtype)
        return _pool_tokens_to_size(image_emb, self.token_per_image, self.context_pool_type)

    def _encode_robomme_context(
        self,
        image_emb: torch.Tensor,
        state: torch.Tensor,
        episode_pos: int,
    ) -> torch.Tensor:
        encoder_dtype = self.context_encoder_static.weight.dtype
        parts = [image_emb.to(dtype=encoder_dtype)]
        if self.context_use_pos_emb:
            spatial_size = int(math.sqrt(self.token_per_image))
            pos = torch.tensor([episode_pos], device=image_emb.device, dtype=torch.long)
            pos_emb = _posemb_3d(pos, spatial_size, self.context_pos_dim, image_emb.dtype)[0]
            pos_emb = pos_emb.to(dtype=self.context_pos_proj.weight.dtype)
            pos_emb = F.silu(self.context_pos_proj(pos_emb))
            parts.append(pos_emb)
        if self.context_use_state_emb:
            state_emb = F.silu(
                self.context_state_proj(state.to(device=image_emb.device, dtype=self.context_state_proj.weight.dtype)[None])
            )[0]
            parts.append(state_emb[None, :].expand(image_emb.shape[0], -1))
        return self.context_encoder_static(torch.cat(parts, dim=-1))

    def _build_context_tokens(
        self,
        observation,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        state: torch.Tensor,
        image_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        device = state.device
        dtype = self.context_pad_token.dtype
        stream_id, episode_id, episode_pos = self._metadata(observation, batch_size, device)
        context_token_count = self._context_token_count(len(images))

        batch_embs = []
        batch_masks = []
        for row in range(batch_size):
            sid = int(stream_id[row].detach().cpu())
            eid = int(episode_id[row].detach().cpu())
            epos = int(episode_pos[row].detach().cpu())
            if epos == 0 or self._history_episode.get(sid) != eid:
                self._history[sid] = []
                self._history_episode[sid] = eid

            cached_images = [image[row].detach() for image in images]
            cached_masks = [mask[row].detach() for mask in img_masks]
            cached_state = state[row].detach()
            cached_features = None
            if image_features is not None:
                if len(image_features) != len(images):
                    raise ValueError(
                        f"image_features length ({len(image_features)}) must match images length ({len(images)})"
                    )
                cached_features = [feature[row].detach() for feature in image_features]

            self._history.setdefault(sid, []).append(
                _CachedFrame(
                    episode_id=eid,
                    episode_pos=epos,
                    images=cached_images,
                    image_masks=cached_masks,
                    state=cached_state,
                    image_features=cached_features,
                )
            )

            row_tokens = []
            row_masks = []
            history = self._sample_history(self._history.get(sid, []))
            for cached in history:
                cached_features = cached.image_features or [None] * len(cached.images)
                for image, image_mask, image_feature in zip(
                    cached.images, cached.image_masks, cached_features, strict=True
                ):
                    image = image.to(device=device)
                    image_emb = self._embed_context_image(image, image_feature, dtype)
                    image_emb = self._encode_robomme_context(image_emb, cached.state, cached.episode_pos)
                    row_tokens.append(image_emb)
                    valid = bool(image_mask.detach().cpu())
                    row_masks.append(torch.full((image_emb.shape[0],), valid, dtype=torch.bool, device=device))

            if row_tokens:
                tokens = torch.cat(row_tokens, dim=0)[:context_token_count]
                masks = torch.cat(row_masks, dim=0)[:context_token_count]
            else:
                tokens = torch.empty((0, self.context_width), dtype=dtype, device=device)
                masks = torch.empty((0,), dtype=torch.bool, device=device)

            pad = context_token_count - tokens.shape[0]
            if pad > 0:
                pad_tokens = self.context_pad_token.to(device=device, dtype=tokens.dtype)[None].expand(pad, -1)
                pad_masks = torch.zeros((pad,), dtype=torch.bool, device=device)
                tokens = torch.cat([tokens, pad_tokens], dim=0)
                masks = torch.cat([masks, pad_masks], dim=0)

            batch_embs.append(tokens)
            batch_masks.append(masks)

        context_embs = torch.stack(batch_embs, dim=0)
        context_pad_masks = torch.stack(batch_masks, dim=0)
        context_att_masks = torch.zeros_like(context_pad_masks)
        return context_embs, context_pad_masks, context_att_masks

    def _select_context_items(
        self,
        keys: list[str],
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        image_features: list[torch.Tensor] | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None]:
        if self.context_image_keys is None:
            return images, img_masks, image_features

        indices = []
        for key in self.context_image_keys:
            if key not in keys:
                raise ValueError(f"context image key {key!r} is missing from observation images: {keys}")
            indices.append(keys.index(key))
        context_images = [images[index] for index in indices]
        context_masks = [img_masks[index] for index in indices]
        context_features = None
        if image_features is not None:
            context_features = [image_features[index] for index in indices]
        return context_images, context_masks, context_features

    def _preprocess_observation(self, observation, *, train=True):
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        keys = list(observation.images)
        images = list(observation.images.values())
        img_masks = list(observation.image_masks.values())
        image_features = None
        if observation.image_features is not None:
            image_features = [observation.image_features[key] for key in observation.images]
        context_images, context_masks, context_features = self._select_context_items(
            keys, images, img_masks, image_features
        )
        self._pending_context = self._build_context_tokens(
            observation, context_images, context_masks, observation.state, context_features
        )
        return (
            images,
            img_masks,
            image_features,
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, image_features=None):
        prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(
            images, img_masks, lang_tokens, lang_masks, image_features
        )
        if self._pending_context is None:
            return prefix_embs, prefix_pad_masks, prefix_att_masks
        context_embs, context_pad_masks, context_att_masks = self._pending_context
        self._pending_context = None
        return (
            torch.cat([context_embs.to(dtype=prefix_embs.dtype), prefix_embs], dim=1),
            torch.cat([context_pad_masks, prefix_pad_masks], dim=1),
            torch.cat([context_att_masks.to(dtype=prefix_att_masks.dtype), prefix_att_masks], dim=1),
        )
