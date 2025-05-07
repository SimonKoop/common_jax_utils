import equinox as eqx
import optax

from common_dl_utils.type_registry import register_type

__all__ = ["Schedule", "register_type"]

# do some initialization of this module
register_type(eqx.Module)
register_type(optax.GradientTransformation)

@register_type
class Schedule:
    pass
