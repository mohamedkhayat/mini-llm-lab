"""The single loss path shared by training and evaluation.

``build_loss`` resolves FLCE availability once (at build time) and returns
a callable ``(x, y) -> loss`` that runs the (possibly compiled) model
forward inside the run's autocast context. ``liger_kernel`` is imported
lazily inside the FLCE branch only, so ``use_flce: false`` (or a CPU run)
never requires Triton/Liger at import time.
"""

from contextlib import nullcontext

import torch


def build_loss(model, raw_model, use_flce, device, use_bf16=False):
    """Return ``(x, y) -> loss`` for ``model`` (possibly compiled).

    ``raw_model`` is the uncompiled model — the FLCE branch reads
    ``lm_head.weight`` / ``lm_head.bias`` from it because the compiled
    wrapper hides them.
    """
    flce = bool(use_flce) and device.type == "cuda"
    if use_flce and not flce:
        print(
            "Warning: FLCE requested but CUDA is not available. "
            "Falling back to CrossEntropyLoss."
        )

    def autocast_ctx():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    if flce:
        # Lazy: Liger/Triton are only needed when FLCE is actually active.
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        criterion = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

        def loss_fn(x, y):
            with autocast_ctx():
                hidden_states = model(x, return_logits=False)
                return criterion(
                    raw_model.lm_head.weight,
                    hidden_states.flatten(0, 1),
                    y.flatten(),
                    bias=raw_model.lm_head.bias,
                )

        return loss_fn

    criterion = torch.nn.CrossEntropyLoss()

    def loss_fn(x, y):
        with autocast_ctx():
            logits = model(x)
            return criterion(logits.flatten(0, 1), y.flatten())

    return loss_fn
