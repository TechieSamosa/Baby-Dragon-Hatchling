from models.bdh_base import BDHBase
from models.bdh_nomul import BDHNoMul
from models.bdh_lowdim import BDHLowDim
from models.bdh_improved import BDHImproved
from models.transformer import Transformer

# Registry of models
VARIANTS = {
    "transformer": Transformer,
    "bdh_base": BDHBase,
    "bdh_nomul": BDHNoMul,
    "bdh_lowdim": BDHLowDim,
    "bdh_improved": BDHImproved,
}

def get_model(model_name: str, config):
    if model_name not in VARIANTS:
        raise ValueError(f"Unknown model variant: {model_name}")
    return VARIANTS[model_name](config)
