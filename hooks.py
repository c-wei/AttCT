"""
MLP forward hook utilities for MLP Consistency Training.

Captures intermediate MLP states (post-activation hidden or block output)
by registering forward hooks on the down-projection layers of transformer
MLP blocks.  Works across architectures (LLaMA, GPT-2, Mistral, Gemma, etc.)
by matching module names rather than checking types.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


def find_mlp_down_proj_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Find down-projection linear layers inside MLP blocks.

    Matches common naming conventions:
        LLaMA / Mistral / Gemma : *.mlp.down_proj
        GPT-2                   : *.mlp.c_proj
        Generic                 : *.ffn.fc2, *.feed_forward.w2

    Returns an *ordered* list of (name, module) tuples, one per transformer
    layer, sorted by the layer index that appears in the name.
    """
    modules: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        name_lower = name.lower()
        is_mlp_context = any(
            s in name_lower for s in ("mlp", "feed_forward", "ffn")
        )
        is_down_proj = any(
            name.endswith(s) for s in (".down_proj", ".fc2", ".w2")
        )
        is_mlp_c_proj = name.endswith(".c_proj") and "mlp" in name_lower

        if is_mlp_context and (is_down_proj or is_mlp_c_proj):
            modules.append((name, module))

    # Sort by layer index extracted from the name (e.g. "layers.3.mlp.down_proj" → 3).
    def _layer_idx(name: str) -> int:
        for part in name.split("."):
            if part.isdigit():
                return int(part)
        return 0

    modules.sort(key=lambda t: _layer_idx(t[0]))
    return modules


class MLPHookManager:
    """
    Manages forward hooks on MLP down-projection layers to capture
    intermediate states for MLP consistency training.

    Usage::

        mgr = MLPHookManager(model, variant="hidden")
        mgr.install()

        # Forward pass
        outputs = model(**inputs)
        states = mgr.get_states()   # list of tensors, one per layer

        mgr.remove()                # clean up

    For ``variant="hidden"``: captures **input** to down_proj —
    the post-activation neuron activations (expanded dim, e.g. 4×hidden_dim).

    For ``variant="output"``: captures **output** of down_proj —
    the MLP block's contribution to the residual stream (hidden_dim).
    """

    def __init__(self, model: nn.Module, variant: str = "hidden"):
        if variant not in ("hidden", "output"):
            raise ValueError(
                f"variant must be 'hidden' or 'output', got '{variant}'"
            )
        self.variant = variant
        self._modules = find_mlp_down_proj_modules(model)
        if not self._modules:
            raise RuntimeError(
                "Could not find MLP down-projection modules in the model. "
                "Looked for names matching *.mlp.down_proj, *.mlp.c_proj, "
                "*.ffn.fc2, or *.feed_forward.w2. "
                "Check that the model architecture has recognisable MLP blocks."
            )
        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._states: List[Optional[torch.Tensor]] = [None] * len(self._modules)
        # When False, hook callbacks short-circuit — no capture, no autograd
        # references held. Used by MLPCTInterleavedTrainer to suppress capture
        # during the KL step (whose forwards don't read MLP states).
        self._active: bool = True

    @property
    def num_layers(self) -> int:
        return len(self._modules)

    def set_active(self, active: bool) -> None:
        """Enable/disable capture without removing hook handles. Cheap toggle
        used to skip capture during forwards that don't need MLP states."""
        self._active = bool(active)

    def install(self) -> "MLPHookManager":
        """Install forward hooks on all MLP down-proj modules. Returns self."""
        self.remove()  # idempotent — clear any previous hooks
        self._states = [None] * len(self._modules)
        mgr = self  # closure-bind to avoid late-binding pitfalls

        for idx, (_name, module) in enumerate(self._modules):
            if self.variant == "hidden":
                # Capture input to down_proj = post-activation hidden states.
                def hook(mod, inp, out, i=idx, mgr=mgr):
                    if mgr._active:
                        mgr._states[i] = inp[0]
            else:
                # Capture output of down_proj = MLP block output.
                def hook(mod, inp, out, i=idx, mgr=mgr):
                    if mgr._active:
                        mgr._states[i] = out

            self._handles.append(module.register_forward_hook(hook))
        return self

    def remove(self):
        """Remove all hooks and release references."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_states(self) -> List[torch.Tensor]:
        """
        Return captured MLP states as a **new** list of tensors (one per layer).

        Safe to call, store the result, then run another forward pass — the
        returned list holds independent references to the captured tensors.

        Note: some architectures (e.g. Gemma 3 with hybrid local/global layers)
        may not fire all hooks on every forward pass. We return only the states
        that fired rather than erroring, since the loss averages over available layers.
        """
        out = [s for s in self._states if s is not None]
        if len(out) == 0:
            raise RuntimeError(
                "MLPHookManager: no layers fired. Was a forward pass run after install()?"
            )
        return out

    def clear(self):
        """Clear captured states to free memory."""
        self._states = [None] * len(self._modules)

    def __repr__(self) -> str:
        return (
            f"MLPHookManager(variant={self.variant!r}, "
            f"num_layers={self.num_layers}, "
            f"hooks_active={len(self._handles) > 0})"
        )
