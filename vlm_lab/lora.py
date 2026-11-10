"""LoRA for the language model only: the vision tower and projector stay frozen and unadapted."""
from __future__ import annotations

# every attention/MLP linear layer whose full module name does NOT contain "visual". The negative lookahead keeps
# this working across transformers versions (the language model lives at model.language_model.* in new releases
# and model.layers.* in old ones), and keeps the vision blocks (which have similarly named layers) untouched.
LM_LINEAR_REGEX = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.0):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LM_LINEAR_REGEX, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def trainable_summary(model) -> dict:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": tr, "total": total, "pct": 100.0 * tr / max(total, 1)}


def lora_module_names(model) -> list[str]:
    """Names of the layers that received a LoRA adapter."""
    return [n for n, m in model.named_modules() if hasattr(m, "lora_A") and hasattr(m, "base_layer")]
