"""DeepSeek V4 hook extensions kept outside RouteAudit's universal core."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from routeaudit.model.hooks import HookCapture, MoEHookManager


@dataclass
class MHCMapCapture:
    """One HyperConnection call retained on the model device."""

    hidden_streams: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    collapsed: torch.Tensor


@dataclass
class DeepSeekV4Capture(HookCapture):
    mhc_maps: dict[int, dict[str, MHCMapCapture]] = None

    def __post_init__(self) -> None:
        self.mhc_maps = {} if self.mhc_maps is None else self.mhc_maps

    def clear(self) -> None:
        super().clear()
        self.mhc_maps.clear()


class DeepSeekV4HookManager(MoEHookManager):
    """Adds mHC map capture to the core, architecture-neutral hook manager."""

    def __init__(self, model: torch.nn.Module, spec=None):
        super().__init__(model, spec)
        self.capture = DeepSeekV4Capture()
        self._capture_mhc_maps = False

    def capture_mhc_maps(self) -> "DeepSeekV4HookManager":
        self._capture_mhc_maps = True
        return self

    def _install(self) -> None:
        super()._install()
        self._install_mhc_hooks()

    def _install_mhc_hooks(self) -> None:
        layers = getattr(self._base_module(), self.spec.layers_attr)
        manager = self
        for layer_idx, layer in enumerate(layers):
            for site, attr in (("attn", "attn_hc"), ("ffn", "ffn_hc")):
                module = getattr(layer, attr, None)
                if not isinstance(module, torch.nn.Module):
                    continue

                def make_hook(li=layer_idx, site_name=site):
                    def hook(_module, inputs, output):
                        if not manager._capture_mhc_maps or not inputs or not isinstance(output, tuple):
                            return output
                        if len(output) < 3 or not isinstance(inputs[0], torch.Tensor):
                            return output
                        post, comb, collapsed = output[:3]
                        if not all(isinstance(x, torch.Tensor) for x in (post, comb, collapsed)):
                            return output
                        manager.capture.mhc_maps.setdefault(li, {})[site_name] = MHCMapCapture(
                            inputs[0].detach(), post.detach(), comb.detach(), collapsed.detach()
                        )
                        return output
                    return hook

                self._handles.append(module.register_forward_hook(make_hook()))

    def remove(self) -> None:
        super().remove()
