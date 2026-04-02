from typing import Callable, Any, Optional, Hashable
from functools import partial

import jax
from jax import tree_util
from jax import numpy as jnp
import numpy as np
import equinox as eqx

Pytree = Any

def is_array_like(leaf):
    return isinstance(leaf, (jax.Array, np.ndarray, np.generic, float, complex, bool, int))

def prng_tree_map(f:Callable, tree:Pytree, *rest:Pytree, key: jax.Array, is_leaf:Optional[Callable[[Any], bool]]=None):
    """ 
    Like jax.tree.map but for random functions, i.e. functions expecting a prng key.

    :parameter f: function that takes `1 + len(rest)` positional arguments and a mandatory keyword argument `key`
    :parameter tree: pytree over which to tree-map `f`
    :parameter rest: optional additional pytrees, each of which should have the same structure as tree or have tree as a prefix
    :parameter key: a prng key for random sampling
    :parameter is_leaf: an optionally specified function that will be called at each flattening step. 
        It should return a boolean, which indicates whether the flattening should traverse the current object, 
        or if it should be stopped immediately, with the whole subtree being treated as a leaf.

    :returns: a new pytree with the same structure as tree, but with the value at each leaf given by `f(x, *xs, key=subkey)`
        where `x` is the value at the corresponding leaf in `tree`, `xs` is the tuple of values at the corresponding nodes in `rest`,
        and `subkey` the result of splitting `key` into N subkeys where N is the number of leafs of `tree`.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    keys = jax.random.split(key, len(leaves))
    return treedef.unflatten(f(*xs, key=leaf_key) for xs, leaf_key in zip(zip(*all_leaves), keys))

def prng_tree_map_with_path(f:Callable, tree:Pytree, *rest:Pytree, key: jax.Array, is_leaf:Optional[Callable[[Any], bool]]=None):
    """ 
    Like jax.tree_util.tree_map_with_path but for random functions, i.e. functions expecting a prng key.

    :parameter f: function that takes `2 + len(rest)` positional arguments and a mandatory keyword argument `key`
    :parameter tree: pytree over which to tree-map `f`
    :parameter rest: optional additional pytrees, each of which should have the same structure as tree or have tree as a prefix
    :parameter key: a prng key for random sampling
    :parameter is_leaf: an optionally specified function that will be called at each flattening step. 
        It should return a boolean, which indicates whether the flattening should traverse the current object, 
        or if it should be stopped immediately, with the whole subtree being treated as a leaf.

    :returns: a new pytree with the same structure as tree, but with the value at each leaf given by `f(kp, x, *xs, key=subkey)`
        where `kp` is the key path of the leaf at the corresponding leaf in `tree`, `x` is the value at the corresponding leaf in `tree`, 
        `xs` is the tuple of values at the corresponding nodes in `rest`, and `subkey` the result of splitting `key` into N subkeys 
        where N is the number of leafs of `tree`.
    """
    keypath_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree, is_leaf)
    keypath_leaves = list(zip(*keypath_leaves))
    all_keypath_leaves = keypath_leaves + [treedef.flatten_up_to(r) for r in rest]
    keys=jax.random.split(key, len(keypath_leaves[0]))
    return treedef.unflatten(f(*xs, key=leaf_key) for xs, leaf_key in zip(zip(*all_keypath_leaves), keys))


def normal_like(key:jax.Array, tree:Pytree, is_leaf:Optional[Callable[[Any], bool]]=None):
    """ 
    Create a tree full of standard normally distributed arrays with the same tree structure as tree and from random keys split from key.
    
    :parameter key: prng key
    :parameter tree: pytree dictating the shape of returned tree
    :parameter is_leaf: optionally specified function used in the flattening step of tree
    
    :returns: a new pytree with the same structure as tree, but with the value at each leaf sampled using jax.random.normal
    """
    return prng_tree_map(f=lambda x, key: jax.random.normal(key, shape=x.shape), tree=tree, key=key, is_leaf=is_leaf)


# all of this ended up no longer beeing useful because jax introduced jax.tree.broadcast :)

# class _ValueWrapper(eqx.Module):
#     value: Pytree

# def _insert_path_and_value_into_nested_dictionary(path:tree_util.KeyPath, value:Any, dictionary:dict):
#     if len(path) == 0:
#         raise ValueError("Cannot insert empty path into nested dictionary.")
#     elif len(path) == 1:
#         dictionary[path[0]] = _ValueWrapper(value)
#     else:
#         current = path[0]
#         rest = path[1:]
#         sub_dictionary = dictionary.setdefault(current, {})
#         _insert_path_and_value_into_nested_dictionary(rest, value, sub_dictionary)

# class PrefixTree(eqx.Module):
#     pytree: Pytree
#     path_dict: dict
#     _is_leaf: Callable

#     def __init__(self, pytree:Pytree, include_None:bool):
#         self.pytree = tree_util.tree_map(lambda x: x, pytree)  # copies the pytree
#         self._is_leaf = (staticmethod(lambda x: x is None) if include_None else None)
#         self.path_dict = {}
#         for path, value in tree_util.tree_leaves_with_path(
#             pytree,
#             is_leaf=self._is_leaf
#         ):
#             _insert_path_and_value_into_nested_dictionary(path, value, self.path_dict)
    
#     def __getitem__(self, path:tree_util.KeyPath):
#         value = self.path_dict
#         full_path = path
#         if not path:
#             raise KeyError("Cannot get item from empty path")
#         while path:
#             current_key = path[0]
#             path = path[1:]
#             try:
#                 value = value[current_key]
#             except KeyError:
#                 raise KeyError(f"{full_path} is not a valid path into {self}.")
#             if isinstance(value, _ValueWrapper):
#                 return value.value
            
#     def broadcast_to_pytree(self, pytree:Pytree, is_leaf:Optional[Callable]=None):
#         paths_and_values, tree_def = tree_util.tree_flatten_with_path(pytree, is_leaf=is_leaf)
#         resulting_paths_and_values = []
#         for path, _ in paths_and_values:
#             resulting_paths_and_values[path] = self[path]
        
        

