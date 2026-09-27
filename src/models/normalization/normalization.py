from models.normalization.layerNorm import LayerNorm
from models.normalization.rms_norm import RMSNorm

# Keys are case- and underscore-insensitive so config/dict spelling drift
# (``layer_norm`` vs ``layernorm``) never breaks a run.
NORM_IMPL = {
    "layernorm": LayerNorm,
    "rmsnorm": RMSNorm,
}


def get_norm(norm_method="layer_norm"):
    key = str(norm_method).strip().lower().replace("_", "")
    if key in NORM_IMPL:
        return NORM_IMPL[key]
    else:
        raise KeyError(f"{norm_method} does not exist, use {sorted(NORM_IMPL)}")
