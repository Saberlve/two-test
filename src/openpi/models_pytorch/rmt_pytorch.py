from __future__ import annotations

import torch
from torch import nn

import openpi.models.gemma as _gemma
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class FeatureEncoder(nn.Module):
    def __init__(self, image_dim: int, state_dim: int, hidden_dim: int, token_per_image: int):
        super().__init__()
        self.token_per_image = int(token_per_image)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.view_embedding = nn.Parameter(torch.zeros(3, hidden_dim))

    def forward(
        self,
        image_embs: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = self.state_proj.weight.dtype
        state_emb = self.state_proj(state.to(dtype=dtype))
        tokens = []
        masks = []
        for view_idx, (image_emb, image_mask) in enumerate(zip(image_embs, image_masks, strict=True)):
            view_tokens = self.image_proj(image_emb[:, : self.token_per_image].to(dtype=dtype))
            view_tokens = view_tokens + state_emb[:, None, :]
            if view_idx < self.view_embedding.shape[0]:
                view_tokens = view_tokens + self.view_embedding[view_idx][None, None, :]
            tokens.append(view_tokens)
            masks.append(image_mask[:, None].expand(image_mask.shape[0], view_tokens.shape[1]))
        return torch.cat(tokens, dim=1), torch.cat(masks, dim=1)


class RMTLayer(nn.Module):
    def __init__(self, hidden_dim: int, budget: int, num_attn_heads: int, num_kv_heads: int, mini_batch_size: int):
        super().__init__()
        self.budget = int(budget)
        self.mini_batch_size = int(mini_batch_size)
        self.num_attn_heads = int(num_attn_heads)
        self.num_kv_heads = int(num_kv_heads)
        if hidden_dim % self.num_attn_heads != 0:
            raise ValueError("num_attn_heads must divide hidden_dim")
        if self.num_attn_heads % self.num_kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_attn_heads")
        self.head_dim = hidden_dim // self.num_attn_heads
        self.memory_init = nn.Parameter(torch.randn(budget, hidden_dim) * (hidden_dim**-0.5))
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def reset(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.memory_init.to(device=device, dtype=dtype)[None].expand(batch_size, -1, -1).clone()

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor, memory_state: torch.Tensor) -> torch.Tensor:
        memory = memory_state
        for start in range(0, tokens.shape[1], self.mini_batch_size):
            chunk = tokens[:, start : start + self.mini_batch_size]
            chunk_mask = token_mask[:, start : start + self.mini_batch_size]
            kv = torch.cat([chunk, memory], dim=1)
            kv = self.pre_norm(kv)
            query = self.pre_norm(memory)
            memory_mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
            kv_mask = torch.cat([chunk_mask, memory_mask], dim=1)
            attended = self._attention(query, kv, kv_mask)
            memory = self.post_norm(memory + attended)
            memory = self.post_norm(memory + self.ffn(memory))
        return memory

    def _attention(self, query: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        batch_size = query.shape[0]
        q = self.q_proj(query).view(batch_size, query.shape[1], self.num_attn_heads, self.head_dim)
        k = self.k_proj(kv).view(batch_size, kv.shape[1], self.num_kv_heads, self.head_dim)
        v = self.v_proj(kv).view(batch_size, kv.shape[1], self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_kv_heads != self.num_attn_heads:
            repeat = self.num_attn_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim**-0.5)
        scores = scores.masked_fill(~kv_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, query.shape[1], self.num_attn_heads * self.head_dim)
        return self.out_proj(out)


class RecurrentMemory(nn.Module):
    def __init__(
        self,
        image_dim: int,
        state_dim: int,
        prefix_dim: int,
        hidden_dim: int,
        budget: int,
        token_per_image: int,
        num_attn_heads: int,
        num_kv_heads: int,
        mini_batch_size: int,
    ):
        super().__init__()
        self.feature_encoder = FeatureEncoder(image_dim, state_dim, hidden_dim, token_per_image)
        self.rmt_layer = RMTLayer(hidden_dim, budget, num_attn_heads, num_kv_heads, mini_batch_size)
        self.output_proj = nn.Linear(hidden_dim, prefix_dim)

    def reset(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.rmt_layer.reset(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        image_embs: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        state: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, token_mask = self.feature_encoder(image_embs, image_masks, state)
        memory_state = self.rmt_layer(tokens, token_mask, memory_state)
        return self.output_proj(memory_state), memory_state


class PI0RMTContextPytorch(PI0Pytorch):
    """PI0/PI05 PyTorch baseline with per-stream recurrent memory tokens."""

    def __init__(self, config):
        super().__init__(config)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self.rmt_memory = RecurrentMemory(
            image_dim=paligemma_config.width,
            state_dim=config.action_dim,
            prefix_dim=paligemma_config.width,
            hidden_dim=int(config.memory_hidden_dim),
            budget=int(config.budget),
            token_per_image=int(config.token_per_image),
            num_attn_heads=int(config.num_attn_heads),
            num_kv_heads=int(config.num_kv_heads),
            mini_batch_size=int(config.mini_batch_size),
        )
        self._memory_state_by_stream: dict[int, torch.Tensor] = {}
        self._episode_by_stream: dict[int, int] = {}
        self._pending_context: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def reset_context_cache(self) -> None:
        self._memory_state_by_stream.clear()
        self._episode_by_stream.clear()

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

    def _build_rmt_context(
        self,
        observation,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        state: torch.Tensor,
        image_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image_features is None:
            image_embs = [self.paligemma_with_expert.embed_image(image) for image in images]
        else:
            if len(image_features) != len(images):
                raise ValueError(
                    f"image_features length ({len(image_features)}) must match images length ({len(images)})"
                )
            image_embs = [feature.detach().to(device=state.device) for feature in image_features]
        batch_size = state.shape[0]
        device = state.device
        stream_id, episode_id, episode_pos = self._metadata(observation, batch_size, device)
        context_tokens = []
        next_states = []

        memory_dtype = next(self.rmt_memory.parameters()).dtype
        for row in range(batch_size):
            sid = int(stream_id[row].detach().cpu())
            eid = int(episode_id[row].detach().cpu())
            epos = int(episode_pos[row].detach().cpu())
            if epos == 0 or self._episode_by_stream.get(sid) != eid:
                self._memory_state_by_stream[sid] = self.rmt_memory.reset(1, device=device, dtype=memory_dtype)[0]
                self._episode_by_stream[sid] = eid
            prev_state = self._memory_state_by_stream[sid].to(device=device, dtype=memory_dtype)
            row_image_embs = [emb[row : row + 1] for emb in image_embs]
            row_img_masks = [mask[row : row + 1] for mask in img_masks]
            row_context, row_next_state = self.rmt_memory(
                row_image_embs,
                row_img_masks,
                state[row : row + 1].to(dtype=memory_dtype),
                prev_state[None],
            )
            context_tokens.append(row_context[0])
            next_states.append((sid, row_next_state[0].detach()))

        for sid, next_state in next_states:
            self._memory_state_by_stream[sid] = next_state

        context_tokens = torch.stack(context_tokens, dim=0)
        context_pad_masks = torch.ones(context_tokens.shape[:2], dtype=torch.bool, device=device)
        context_att_masks = torch.zeros_like(context_pad_masks)
        return context_tokens, context_pad_masks, context_att_masks

    def _preprocess_observation(self, observation, *, train=True):
        images, img_masks, image_features, lang_tokens, lang_masks, state = super()._preprocess_observation(
            observation, train=train
        )
        self._pending_context = self._build_rmt_context(observation, images, img_masks, state, image_features)
        return images, img_masks, image_features, lang_tokens, lang_masks, state

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
