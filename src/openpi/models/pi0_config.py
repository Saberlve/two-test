import dataclasses
import math
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.models_pytorch.lora_pytorch as lora_pytorch
import safetensors
from openpi.models_pytorch import rmt_pytorch

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class Pi0RMTContextConfig(Pi0Config):
    """PyTorch-only PI0/PI05 config with RoboMME-aligned recurrent memory transformer context."""

    max_recur_steps: int = 64
    input_obs_horizon: int = 8
    budget: int = 8
    token_per_image: int = 64
    num_views: int = 1
    memory_hidden_dim: int = 256
    num_attn_heads: int = 8
    num_kv_heads: int = 1
    context_image_keys: tuple[str, ...] | None = None
    context_pos_dim: int = 768
    context_pos_hidden_dim: int = 768
    context_state_hidden_dim: int = 512
    context_pool_type: str = "mean"
    use_pos_emb: bool = True
    use_state_emb: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.max_recur_steps < 1:
            raise ValueError(f"max_recur_steps must be >= 1, got {self.max_recur_steps}")
        if self.input_obs_horizon < 1:
            raise ValueError(f"input_obs_horizon must be >= 1, got {self.input_obs_horizon}")
        if self.budget < 1:
            raise ValueError(f"budget must be >= 1, got {self.budget}")
        if self.token_per_image < 1:
            raise ValueError(f"token_per_image must be >= 1, got {self.token_per_image}")
        if int(math.isqrt(self.token_per_image)) ** 2 != self.token_per_image:
            raise ValueError(f"token_per_image must be a perfect square, got {self.token_per_image}")
        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}")
        if self.memory_hidden_dim < 1:
            raise ValueError(f"memory_hidden_dim must be >= 1, got {self.memory_hidden_dim}")
        if self.num_attn_heads < 1 or self.memory_hidden_dim % self.num_attn_heads != 0:
            raise ValueError("num_attn_heads must divide memory_hidden_dim")
        if self.num_kv_heads < 1:
            raise ValueError(f"num_kv_heads must be >= 1, got {self.num_kv_heads}")
        if self.num_attn_heads % self.num_kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_attn_heads")
        head_dim = self.memory_hidden_dim // self.num_attn_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for rotary embeddings")
        if self.context_pos_dim % 6 != 0:
            raise ValueError(f"context_pos_dim must be divisible by 6, got {self.context_pos_dim}")
        spatial_size = int(math.isqrt(self.token_per_image))
        if 16 % spatial_size != 0:
            raise ValueError(
                f"sqrt(token_per_image)={spatial_size} must divide 16 (RoboMME PosEmb3D constraint)"
            )
        if self.context_pool_type not in ("mean", "max"):
            raise ValueError(f"context_pool_type must be 'mean' or 'max', got {self.context_pool_type!r}")

    @override
    def load_pytorch(self, train_config, weight_path: str):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"train_config: {train_config}")
        model = rmt_pytorch.PI0RMTContextPytorch(config=train_config.model)

        # Apply LoRA if configured before loading weights.
        if hasattr(train_config, "lora_config") and train_config.lora_config is not None:
            lora_pytorch.apply_lora_to_pi0_pytorch(model, train_config.lora_config)
            logger.info("Applied LoRA to PI0RMTContextPytorch model.")

        safetensors.torch.load_model(model, weight_path)
        return model
