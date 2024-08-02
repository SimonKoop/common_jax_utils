import jax

def summary(obj):
    """ 
    Give a more readable summary of a pytree. 
    """
    return str(jax.tree_map(
        lambda x: f"array{x.shape}" if hasattr(x, 'shape') else repr(x),
        obj
    ))