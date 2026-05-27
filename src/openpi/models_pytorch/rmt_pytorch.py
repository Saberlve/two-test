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
    """Pool square image token grids to match RoboMME's 8x8 recurrent sidecar."""
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


def _precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary frequencies (cos, sin) matching RoboMME's complex form."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using paired (re, im) interpretation, mirroring RoboMME's `apply_rotary_emb`."""
    dtype = xq.dtype
    xq_r = xq.to(dtype=torch.float32).reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.to(dtype=torch.float32).reshape(*xk.shape[:-1], -1, 2)
    xq_re, xq_im = xq_r[..., 0], xq_r[..., 1]
    xk_re, xk_im = xk_r[..., 0], xk_r[..., 1]

    cos = freqs_cos.to(device=xq.device)
    sin = freqs_sin.to(device=xq.device)
    while cos.dim() < xq_re.dim() - 1:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    out_re_q = xq_re * cos - xq_im * sin
    out_im_q = xq_re * sin + xq_im * cos
    out_re_k = xk_re * cos - xk_im * sin
    out_im_k = xk_re * sin + xk_im * cos
    xq_out = torch.stack([out_re_q, out_im_q], dim=-1).reshape(xq.shape)
    xk_out = torch.stack([out_re_k, out_im_k], dim=-1).reshape(xk.shape)
    return xq_out.to(dtype), xk_out.to(dtype)


def _get_token_recurrent_indices(
    step_idx: int, input_obs_horizon: int, max_recur_steps: int
) -> list[int]:
    """Mirror of RoboMME's `get_token_recurrent_indices` with `exec_start_idx == 0`."""
    if step_idx < input_obs_horizon:
        indices = [step_idx]
    else:
        start_idx = step_idx % input_obs_horizon
        indices = list(range(start_idx, step_idx + 1, input_obs_horizon))
        indices = indices[-max_recur_steps:]
    return indices


class FeatureEncoder(nn.Module):
    """Concat-style encoder matching `mem_encoder.FeatureEncoder.encode_recurrent_memory`."""

    def __init__(
        self,
        image_input_dim: int,
        pos_input_dim: int,
        pos_output_dim: int,
        state_input_dim: int,
        state_output_dim: int,
        output_dim: int,
        use_pos_emb: bool,
        use_state_emb: bool,
    ):
        super().__init__()
        self.use_pos_emb = bool(use_pos_emb)
        self.use_state_emb = bool(use_state_emb)
        encoder_input_dim = image_input_dim
        if self.use_pos_emb:
            self.pos_proj = nn.Linear(pos_input_dim, pos_output_dim)
            encoder_input_dim += pos_output_dim
        else:
            self.pos_proj = None
        if self.use_state_emb:
            self.state_proj = nn.Linear(state_input_dim, state_output_dim)
            encoder_input_dim += state_output_dim
        else:
            self.state_proj = None
        self.encoder_recur = nn.Linear(encoder_input_dim, output_dim)

    def forward(
        self,
        image_emb: torch.Tensor,
        pos_emb: torch.Tensor | None,
        state_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode `(b, t, v, p, d_img)` → `(b, t, v, p, hidden_dim)`."""
        parts = [image_emb]
        if self.use_pos_emb and pos_emb is not None:
            parts.append(F.silu(self.pos_proj(pos_emb)))
        if self.use_state_emb and state_emb is not None:
            state_proj = F.silu(self.state_proj(state_emb))
            v, p = image_emb.shape[2], image_emb.shape[3]
            state_proj = state_proj[:, :, None, None, :].expand(-1, -1, v, p, -1)
            parts.append(state_proj)
        return self.encoder_recur(torch.cat(parts, dim=-1))


class RMTLayer(nn.Module):
    """Recurrent memory cross-attention layer, aligned with RoboMME's `RMTLayer`."""

    def __init__(
        self,
        hidden_dim: int,
        budget: int,
        num_attn_heads: int,
        num_kv_heads: int,
        max_input_tokens: int,
    ):
        super().__init__()
        if hidden_dim % num_attn_heads != 0:
            raise ValueError("num_attn_heads must divide hidden_dim")
        if num_attn_heads % num_kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_attn_heads")
        self.hidden_dim = int(hidden_dim)
        self.budget = int(budget)
        self.num_attn_heads = int(num_attn_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = hidden_dim // num_attn_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.post_norm = nn.LayerNorm(hidden_dim)

        self.memory_state = nn.Parameter(
            torch.randn(self.budget, hidden_dim) * (hidden_dim**-0.5)
        )

        freqs_cos, freqs_sin = _precompute_freqs_cis(self.head_dim, (max_input_tokens + self.budget) * 2)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def reset(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.memory_state.to(device=device, dtype=dtype)[None].expand(batch_size, -1, -1).clone()

    def _process_mini_batch(self, x_mb: torch.Tensor, mem_state: torch.Tensor) -> torch.Tensor:
        """Cross-attention with rotary PE; returns updated memory `(b, mem_slots, d)`."""
        mem_slots = mem_state.shape[1]
        inputs = torch.cat([x_mb, mem_state], dim=1)
        inputs = self.pre_norm(inputs)
        bsize, seq_len, _ = inputs.shape

        q = self.q_proj(inputs).view(bsize, seq_len, self.num_attn_heads, self.head_dim)
        k = self.k_proj(inputs).view(bsize, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(inputs).view(bsize, seq_len, self.num_kv_heads, self.head_dim)

        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]
        q, k = _apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        if self.num_kv_heads != self.num_attn_heads:
            repeat = self.num_attn_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim**-0.5)
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v)
        attended = attended.transpose(1, 2).reshape(bsize, seq_len, self.num_attn_heads * self.head_dim)
        out = self.to_out(attended[:, -mem_slots:])
        return self.post_norm(out + mem_state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        mem_state: torch.Tensor,
    ) -> torch.Tensor:
        """`hidden_states` (b, nm, mb, d), `mask` (b, nm), `mem_state` (b, mem_slots, d)."""
        nm = hidden_states.shape[1]
        for t in range(nm):
            x_mb = hidden_states[:, t]
            mask_t = mask[:, t]
            new_mem = self._process_mini_batch(x_mb, mem_state)
            mem_state = torch.where(mask_t[:, None, None], new_mem, mem_state)
        return mem_state


class RecurrentMemory(nn.Module):
    """Window-scan recurrent memory matching RoboMME's `RecurrentMemory`."""

    def __init__(
        self,
        image_input_dim: int,
        pos_input_dim: int,
        pos_output_dim: int,
        state_input_dim: int,
        state_output_dim: int,
        hidden_dim: int,
        output_dim: int,
        budget: int,
        max_recur_steps: int,
        tokens_per_frame: int,
        num_attn_heads: int,
        num_kv_heads: int,
        use_pos_emb: bool,
        use_state_emb: bool,
    ):
        super().__init__()
        self.budget = int(budget)
        self.max_recur_steps = int(max_recur_steps)
        self.tokens_per_frame = int(tokens_per_frame)
        self.max_input_tokens = self.max_recur_steps * self.tokens_per_frame

        self.feature_encoder = FeatureEncoder(
            image_input_dim=image_input_dim,
            pos_input_dim=pos_input_dim,
            pos_output_dim=pos_output_dim,
            state_input_dim=state_input_dim,
            state_output_dim=state_output_dim,
            output_dim=hidden_dim,
            use_pos_emb=use_pos_emb,
            use_state_emb=use_state_emb,
        )
        self.recur_layer = RMTLayer(
            hidden_dim=hidden_dim,
            budget=self.budget,
            num_attn_heads=num_attn_heads,
            num_kv_heads=num_kv_heads,
            max_input_tokens=self.tokens_per_frame,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def reset(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.recur_layer.reset(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        recur_image_emb: torch.Tensor,  # (b, t, v, p, d_img)
        recur_mask: torch.Tensor,  # (b, t)
        recur_pos_emb: torch.Tensor | None,  # (b, t, v, p, d_pos)
        recur_state_emb: torch.Tensor | None,  # (b, t, d_state)
        memory_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = recur_image_emb.shape[0]
        device = recur_image_emb.device
        dtype = next(self.parameters()).dtype

        if memory_state is None:
            memory_state = self.recur_layer.reset(batch_size, device=device, dtype=dtype)

        hidden_states = self.feature_encoder(
            recur_image_emb.to(dtype=dtype),
            recur_pos_emb.to(dtype=dtype) if recur_pos_emb is not None else None,
            recur_state_emb.to(dtype=dtype) if recur_state_emb is not None else None,
        )
        b, t, v, p, d = hidden_states.shape
        hidden_states = hidden_states.reshape(b, t, v * p, d)

        memory_state = self.recur_layer(hidden_states, recur_mask.to(device=device), memory_state)
        final_output = self.proj(memory_state[:, -self.budget:])
        final_mask = torch.ones((b, self.budget), dtype=torch.bool, device=device)
        return final_output, final_mask


class PI0RMTContextPytorch(PI0Pytorch):
    """PI0/PI05 PyTorch baseline with RoboMME-aligned recurrent memory context.

    Per inference step: sample a window of historical frames via RoboMME's
    `get_token_recurrent_indices` (no video prefix), left-pad to
    `max_recur_steps`, encode and scan from a learned initial state. Memory
    state does NOT persist across inference calls — it is rebuilt each step,
    matching RoboMME.
    """

    def __init__(self, config):
        super().__init__(config)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self.input_obs_horizon = int(config.input_obs_horizon)
        self.max_recur_steps = int(config.max_recur_steps)
        self.token_per_image = int(config.token_per_image)
        self.num_views = int(config.num_views)
        self.image_input_dim = int(paligemma_config.width)
        self.context_image_keys = None if config.context_image_keys is None else tuple(config.context_image_keys)
        self.context_pos_dim = int(config.context_pos_dim)
        self.context_pool_type = str(config.context_pool_type)
        self.use_pos_emb = bool(config.use_pos_emb)
        self.use_state_emb = bool(config.use_state_emb)

        self.rmt_memory = RecurrentMemory(
            image_input_dim=paligemma_config.width,
            pos_input_dim=int(config.context_pos_dim),
            pos_output_dim=int(config.context_pos_hidden_dim),
            state_input_dim=int(config.action_dim),
            state_output_dim=int(config.context_state_hidden_dim),
            hidden_dim=int(config.memory_hidden_dim),
            output_dim=paligemma_config.width,
            budget=int(config.budget),
            max_recur_steps=int(config.max_recur_steps),
            tokens_per_frame=self.num_views * self.token_per_image,
            num_attn_heads=int(config.num_attn_heads),
            num_kv_heads=int(config.num_kv_heads),
            use_pos_emb=bool(config.use_pos_emb),
            use_state_emb=bool(config.use_state_emb),
        )

        self._history: dict[int, dict[int, _CachedFrame]] = {}
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

    def _embed_context_image(self, image: torch.Tensor, image_feature: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
        if image_feature is None:
            image_emb = self.paligemma_with_expert.embed_image(image[None])[0]
        else:
            image_emb = image_feature.to(device=image.device, dtype=dtype)
        return _pool_tokens_to_size(image_emb, self.token_per_image, self.context_pool_type)

    def _compute_frame_pos_emb(self, episode_pos: int, num_views: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return pos_emb of shape `(num_views, token_per_image, pos_dim)`.

        Mirrors RoboMME's `pos_emb_dict[spatial][step*v : (step+1)*v]` where each
        view of a frame occupies a separate temporal slot in the linearised
        timeline `step * v + view_idx`.
        """
        spatial_size = int(math.sqrt(self.token_per_image))
        positions = torch.arange(
            episode_pos * num_views,
            episode_pos * num_views + num_views,
            device=device,
            dtype=torch.long,
        )
        return _posemb_3d(positions, spatial_size, self.context_pos_dim, dtype)

    def _build_rmt_context(
        self,
        observation,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        state: torch.Tensor,
        image_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        device = state.device
        num_views = len(images)
        if num_views != self.num_views:
            raise ValueError(
                f"observation provides {num_views} views but config.num_views = {self.num_views}"
            )
        dtype = next(self.rmt_memory.parameters()).dtype
        stream_id, episode_id, episode_pos = self._metadata(observation, batch_size, device)

        batch_image_emb = []
        batch_pos_emb = []
        batch_state_emb = []
        batch_mask = []

        for row in range(batch_size):
            sid = int(stream_id[row].detach().cpu())
            eid = int(episode_id[row].detach().cpu())
            epos = int(episode_pos[row].detach().cpu())
            if self._history_episode.get(sid) != eid:
                self._history[sid] = {}
                self._history_episode[sid] = eid

            cached_images = [image[row].detach() for image in images]
            cached_masks = [mask[row].detach() for mask in img_masks]
            cached_state = state[row].detach()
            cached_features = None
            if image_features is not None:
                if len(image_features) != num_views:
                    raise ValueError(
                        f"image_features length ({len(image_features)}) must match images length ({num_views})"
                    )
                cached_features = [feature[row].detach() for feature in image_features]

            self._history.setdefault(sid, {})[epos] = _CachedFrame(
                episode_id=eid,
                episode_pos=epos,
                images=cached_images,
                image_masks=cached_masks,
                state=cached_state,
                image_features=cached_features,
            )

            indices = _get_token_recurrent_indices(epos, self.input_obs_horizon, self.max_recur_steps)
            row_image_emb = []
            row_pos_emb = []
            row_state_emb = []
            row_mask = []
            for src_idx in indices:
                cached = self._history[sid].get(src_idx)
                if cached is None:
                    # Missing historical frame (e.g. training-time random sampling).
                    # Insert a zero placeholder and mask it out so RMT skips the update.
                    row_image_emb.append(
                        torch.zeros((num_views, self.token_per_image, self.image_input_dim), dtype=dtype, device=device)
                    )
                    if self.use_pos_emb:
                        row_pos_emb.append(
                            self._compute_frame_pos_emb(src_idx, num_views, dtype, device)
                        )
                    row_state_emb.append(torch.zeros_like(cached_state.to(device=device, dtype=dtype)))
                    row_mask.append(False)
                    continue
                frame_image_embs = []
                features = cached.image_features or [None] * len(cached.images)
                for image, image_feature in zip(cached.images, features, strict=True):
                    image_emb = self._embed_context_image(image.to(device=device), image_feature, dtype)
                    frame_image_embs.append(image_emb)
                row_image_emb.append(torch.stack(frame_image_embs, dim=0))  # (v, p, d_img)
                if self.use_pos_emb:
                    row_pos_emb.append(
                        self._compute_frame_pos_emb(cached.episode_pos, num_views, dtype, device)
                    )
                row_state_emb.append(cached.state.to(device=device, dtype=dtype))
                row_mask.append(True)

            row_image = torch.stack(row_image_emb, dim=0)
            row_state = torch.stack(row_state_emb, dim=0)
            row_pos = torch.stack(row_pos_emb, dim=0) if self.use_pos_emb else None
            row_mask_tensor = torch.tensor(row_mask, dtype=torch.bool, device=device)

            pad = self.max_recur_steps - row_image.shape[0]
            if pad > 0:
                pad_img = torch.zeros((pad, *row_image.shape[1:]), dtype=row_image.dtype, device=device)
                pad_state = torch.zeros((pad, *row_state.shape[1:]), dtype=row_state.dtype, device=device)
                pad_mask = torch.zeros((pad,), dtype=torch.bool, device=device)
                row_image = torch.cat([pad_img, row_image], dim=0)
                row_state = torch.cat([pad_state, row_state], dim=0)
                row_mask_tensor = torch.cat([pad_mask, row_mask_tensor], dim=0)
                if row_pos is not None:
                    pad_pos = torch.zeros((pad, *row_pos.shape[1:]), dtype=row_pos.dtype, device=device)
                    row_pos = torch.cat([pad_pos, row_pos], dim=0)

            batch_image_emb.append(row_image)
            batch_state_emb.append(row_state)
            batch_mask.append(row_mask_tensor)
            if row_pos is not None:
                batch_pos_emb.append(row_pos)

        recur_image_emb = torch.stack(batch_image_emb, dim=0)
        recur_pos_emb = torch.stack(batch_pos_emb, dim=0) if self.use_pos_emb else None
        recur_state_emb = torch.stack(batch_state_emb, dim=0) if self.use_state_emb else None
        recur_mask = torch.stack(batch_mask, dim=0)
        del batch_image_emb, batch_pos_emb, batch_state_emb, batch_mask

        context_tokens, context_pad_masks = self.rmt_memory(
            recur_image_emb, recur_mask, recur_pos_emb, recur_state_emb
        )
        context_att_masks = torch.zeros_like(context_pad_masks)
        return context_tokens, context_pad_masks, context_att_masks

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
        self._pending_context = self._build_rmt_context(
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
