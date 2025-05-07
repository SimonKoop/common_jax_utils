""" 
This is a collection of functions for creating optimizers
Because much of optax works with factory functions, these don't always
play nice with the config_realization tools from common_dl_utils when it comes to type annotations
(e.g. the type annotation for mask in adamw is Optional[Union[Any, Callable[[optax.Params], Any]]])
"""
from typing import Optional, Callable, Union

import optax

from common_jax_utils.types import Schedule

def single_optimizer(
        optimizer_type: type[optax.GradientTransformation],
        optimizer_config: dict,
        optimizer_mask:Optional[Callable]=None,
        learning_rate_schedule:Union[None, Schedule, list[Schedule], tuple[Schedule, ...]]=None,
        schedule_boundaries:Optional[list[int]]=None,
        )->optax.GradientTransformation:
    """single_optimizer
    Create a single optimizer (without synced parameters) optionally with a learning rate schedule and a mask function

    :param optimizer_type: factory for an optax.GradientTransformation (e.g. optax.adam)
    :param optimizer_config: configuration for the optimizer
    :param optimizer_mask: optional callable for masking parameters, defaults to None
        if this is not None, it is added to (a copy of) optimizer_config under the key 'mask'
    :param learning_rate_schedule: optional learning rate schedule, defaults to None
        if this is not None, it is added to (a copy of) optimizer_config under the key 'learning_rate'
        if this is a list or tuple, it is joined using optax.join_schedules with schedule_boundaries for boundaries
    :param schedule_boundaries: boundaries for optax.join_schedules
        only used if learning_rate_schedule is a list or tuple
    :return: an optax optimizer
    """
    optimizer_config = dict(optimizer_config)
    if learning_rate_schedule is not None:
        if isinstance(learning_rate_schedule, (list, tuple)):
            learning_rate_schedule = optax.join_schedules(learning_rate_schedule, schedule_boundaries)
        optimizer_config['learning_rate'] = learning_rate_schedule
    if optimizer_mask is not None:
        optimizer_config['mask'] = optimizer_mask
    return optimizer_type(**optimizer_config)
