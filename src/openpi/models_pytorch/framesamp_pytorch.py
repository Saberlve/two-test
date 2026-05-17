from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import openpi.models.gemma as _gemma
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


@dataclass
class _CachedFrame:
    episode_id: int
    images: list[torch.Tensor]
    image_masks: list[torch.Tensor]
    state: torch.Tensor


class PI0FramesampContextPytorch(PI0Pytorch):
    """PI0/PI05 PyTorch baseline with frame-sampled prefix context tokens."""

    def __init__(self, config):
        super().__init__(config)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self.context_window = int(config.context_window)
        self.frame_sample_stride = int(config.frame_sample_stride)
        self.token_per_image = int(config.token_per_image)
        self.context_budget = int(config.budget)
        self.context_state_proj = nn.Linear(config.action_dim, paligemma_config.width)
        self.context_pos_embedding = nn.Parameter(torch.zeros(self.context_window, paligemma_config.width))
        self.context_pad_token = nn.Parameter(torch.zeros(paligemma_config.width))
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
        sampled = history[::-self.frame_sample_stride][: self.context_window]
        return list(reversed(sampled))

    def _build_context_tokens(
        self,
        observation,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        device = state.device
        dtype = next(self.context_state_proj.parameters()).dtype
        stream_id, episode_id, episode_pos = self._metadata(observation, batch_size, device)
        width = self.context_pad_token.shape[-1]

        batch_embs = []
        batch_masks = []
        for row in range(batch_size):
            sid = int(stream_id[row].detach().cpu())
            eid = int(episode_id[row].detach().cpu())
            epos = int(episode_pos[row].detach().cpu())
            if epos == 0 or self._history_episode.get(sid) != eid:
                self._history[sid] = []
                self._history_episode[sid] = eid

            row_tokens = []
            row_masks = []
            history = self._sample_history(self._history.get(sid, []))
            left_pad_frames = self.context_window - len(history)
            for hist_pos, cached in enumerate(history, start=left_pad_frames):
                state_emb = self.context_state_proj(cached.state.to(device=device, dtype=dtype)[None])[0]
                pos_emb = self.context_pos_embedding[hist_pos].to(device=device, dtype=state_emb.dtype)
                for image, image_mask in zip(cached.images, cached.image_masks, strict=True):
                    image = image.to(device=device)
                    image_emb = self.paligemma_with_expert.embed_image(image[None])[0, : self.token_per_image]
                    image_emb = image_emb + state_emb[None, :] + pos_emb[None, :]
                    row_tokens.append(image_emb)
                    valid = bool(image_mask.detach().cpu())
                    row_masks.append(torch.full((image_emb.shape[0],), valid, dtype=torch.bool, device=device))

            if row_tokens:
                tokens = torch.cat(row_tokens, dim=0)[-self.context_budget :]
                masks = torch.cat(row_masks, dim=0)[-self.context_budget :]
            else:
                tokens = torch.empty((0, width), dtype=dtype, device=device)
                masks = torch.empty((0,), dtype=torch.bool, device=device)

            pad = self.context_budget - tokens.shape[0]
            if pad > 0:
                pad_tokens = self.context_pad_token.to(device=device, dtype=tokens.dtype)[None].expand(pad, -1)
                tokens = torch.cat([pad_tokens, tokens], dim=0)
                masks = torch.cat([torch.zeros((pad,), dtype=torch.bool, device=device), masks], dim=0)

            batch_embs.append(tokens)
            batch_masks.append(masks)

            cached_images = [image[row].detach() for image in images]
            cached_masks = [mask[row].detach() for mask in img_masks]
            cached_state = state[row].detach()
            self._history.setdefault(sid, []).append(
                _CachedFrame(episode_id=eid, images=cached_images, image_masks=cached_masks, state=cached_state)
            )
            self._history[sid] = self._history[sid][-self.context_window * self.frame_sample_stride :]

        context_embs = torch.stack(batch_embs, dim=0)
        context_pad_masks = torch.stack(batch_masks, dim=0)
        context_att_masks = torch.zeros_like(context_pad_masks)
        return context_embs, context_pad_masks, context_att_masks

    def _preprocess_observation(self, observation, *, train=True):
        images, img_masks, lang_tokens, lang_masks, state = super()._preprocess_observation(observation, train=train)
        self._pending_context = self._build_context_tokens(observation, images, img_masks, state)
        return images, img_masks, lang_tokens, lang_masks, state

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if self._pending_context is None:
            return prefix_embs, prefix_pad_masks, prefix_att_masks
        context_embs, context_pad_masks, context_att_masks = self._pending_context
        self._pending_context = None
        return (
            torch.cat([context_embs.to(dtype=prefix_embs.dtype), prefix_embs], dim=1),
            torch.cat([context_pad_masks, prefix_pad_masks], dim=1),
            torch.cat([context_att_masks.to(dtype=prefix_att_masks.dtype), prefix_att_masks], dim=1),
        )
