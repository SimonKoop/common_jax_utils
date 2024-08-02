import inspect
import functools
from os import PathLike
from typing import Callable, Union

import jax


def load_arrays(func:Callable):
    """
    Transform the signature of func so that any argument that should
    be a jax.Array will also accept a string representing a path
    If such a path is provided, the array will be loaded from the path and passed to the underlying function.
    """
    # first we find out based on the original signature what parameters might need loading from a path
    # and what the new signature should be
    original_signature = inspect.signature(func)
    array_variables = tuple((
        param_name for param_name, param in original_signature.parameters.items()
        if param.annotation == jax.Array  # we might want to make this a bit more flexible at some point
    ))

    target_parameters = {
        param_name: (param if param.annotation != jax.Array else inspect.Parameter(
            name=param_name,
            kind=param.kind,
            annotation=Union[jax.Array, str, PathLike],
        ))
        for param_name, param in original_signature.parameters.items()
    }

    new_signature = original_signature.replace(parameters=target_parameters.values())
    try:
        func.__signature__ = new_signature
    except AttributeError:
        # happens when func is a bound method
        func.__func__.__signature = new_signature

    # now it's time to work on actually loading arrays from paths when necessary
    def requires_modification(param_name, value):  # function for finding out if a value should be loaded from a path
        return (param_name in array_variables and (isinstance(value, PathLike) or isinstance(value, str)))

    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        # load arrays from paths where necessary
        new_args = [
            value if not requires_modification(param_name, value) else jax.numpy.load(value)
            for value, param_name in zip(args, target_parameters)
        ]
        new_kwargs = {
            param_name: (value if not requires_modification(param_name, value) else jax.numpy.load(value))
            for param_name, value in kwargs.items()
        }
        # execute function
        return func(*new_args, **new_kwargs)
    return wrapped_function
