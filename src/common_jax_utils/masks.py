""" 
Module with various masking functions for the use of e.g. weight decay on parts of models
"""
from typing import Union, Callable
from functools import partial

import jax
from jax import tree_util
import equinox as eqx
from jaxtyping import PyTree

def array_mask(tree: PyTree):
    return jax.tree.map(eqx.is_array, tree, is_leaf=lambda x: x is None or eqx.is_array(x))

def inexact_array_mask(tree: PyTree):
    return jax.tree.map(eqx.is_inexact_array, tree, is_leaf=lambda x: x is None or eqx.is_array(x))

def _sub_model_masking_function(key_path, value, sub_model_path):
    return all(actual.name == target for actual, target in zip(key_path, sub_model_path))

def sub_model_mask(
        model: eqx.Module,
        sub_model_path:Union[str, tuple[str,...], list[str]],
    ):
    """ 
    Masking function that returns a mask with True for each parameter in the sub-model specified by sub_model_path and False everywhere else
    :parameter model: eqx.Module on which to base the mask
    :parameter sub_model_path: 
        either a string with the name of a direct sub-model in model
        or a tuple/list of strings specifying a dotted path to a sub-model
        e.g. if sub_model_path = ('a', 'b') then every array in model.a.b will be True and everything else False
    """
    if isinstance(sub_model_path, str):
        sub_model_path = (sub_model_path,)
    return tree_util.tree_map_with_path(
        partial(_sub_model_masking_function, sub_model_path=sub_model_path),
        model
    )

def true_mask(model: eqx.Module):
    """ 
    create a mask for model with only True values
    """
    return tree_util.tree_map(lambda x: True, model)
    
def _union(*values:Union[bool, None]):
    if not isinstance(values[0], bool):
        return None
    return any(values)

def _intersection(*values:Union[bool, None]):
    if not isinstance(values[0], bool):
        return None
    return all(values)

def union_mask(model:eqx.Module, *mask_functions:Callable):
    """ 
    return the union of several masks (logical or / any)
    """
    masks = [mask_function(model) for mask_function in mask_functions]
    return tree_util.tree_map(_union, *masks)

def intersection_mask(model:eqx.Module, *mask_functions:Callable):
    """ 
    return the intersection of several masks (logical and / all)
    """
    masks = [mask_function(model) for mask_function in mask_functions]
    return tree_util.tree_map(_intersection, *masks)
