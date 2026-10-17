"""Attention/token-mixer registry.

Only HEPT and DeltaNet are needed for the apples-to-apples comparison in this
repo. The other baselines from the original HEPT codebase (performer, reformer,
smyrf, sb, flt, pct, flatformer) depend on extra third-party libraries
(fast_transformers, reformer_pytorch, ...) that are not installed here, so their
imports are made optional: a missing baseline becomes ``None`` instead of
crashing the whole package.
"""

# Always-available mixers (no extra deps beyond torch/einops).
from .hept import HEPTAttention
from .deltanet import DeltaNetAttention

# Optional baselines: import if their dependencies happen to be installed.
_OPTIONAL = {
    "PerformerAttention": (".performer", "PerformerAttention"),
    "ReformerAttention": (".reformer", "ReformerAttention"),
    "SmyrfAttention": (".smyrf", "SmyrfAttention"),
    "SBAttention": (".sb", "SBAttention"),
    "FLTAttention": (".flt", "FLTAttention"),
    "PCTAttention": (".pct", "PCTAttention"),
    "FlatformerAttention": (".flatformer", "FlatformerAttention"),
}

import importlib as _importlib

for _name, (_mod, _attr) in _OPTIONAL.items():
    try:
        _m = _importlib.import_module(_mod, __name__)
        globals()[_name] = getattr(_m, _attr)
    except Exception:  # noqa: BLE001 - optional baseline, dependency missing
        globals()[_name] = None
