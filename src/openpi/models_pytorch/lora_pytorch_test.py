import torch

from openpi.models_pytorch import lora_pytorch


class _DummyWrappedMemoryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.memory_attn = torch.nn.Linear(2, 2)
        self.base_model.memory_residual_gate = torch.nn.Parameter(torch.ones(1))
        self.base_model.other = torch.nn.Linear(2, 2)
        self.moment_tokens = torch.nn.Parameter(torch.ones(2, 2))
        self.memory_module = torch.nn.Linear(2, 2)
        self.wrapper_extra = torch.nn.Linear(2, 2)


def test_freeze_for_lora_training_keeps_only_memory_extras_trainable_on_wrapped_model():
    model = _DummyWrappedMemoryModel()
    lora_config = lora_pytorch.LoRATrainingConfig(
        enabled=True,
        train_non_lora_layers=True,
        train_vision_encoder=False,
        extra_trainable_modules=["moment_tokens", "memory_module"],
    )

    lora_pytorch.freeze_for_lora_training(model, lora_config)

    assert model.moment_tokens.requires_grad
    assert model.memory_module.weight.requires_grad
    assert model.memory_module.bias.requires_grad

    assert not model.base_model.memory_attn.weight.requires_grad
    assert not model.base_model.memory_attn.bias.requires_grad
    assert not model.base_model.memory_residual_gate.requires_grad
    assert not model.base_model.other.weight.requires_grad
    assert not model.base_model.other.bias.requires_grad
    assert not model.wrapper_extra.weight.requires_grad
    assert not model.wrapper_extra.bias.requires_grad
