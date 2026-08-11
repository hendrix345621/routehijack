"""Optional DeepSeek V4/mHC integration for RouteAudit."""

from .config import register
from .hooks import DeepSeekV4HookManager, MHCMapCapture

register()

__all__ = ["DeepSeekV4HookManager", "MHCMapCapture", "register"]
