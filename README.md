This is a package with stuff that is useful when working in jax in general. It's not specific to any project.
Part of this package is centered around making the tools from common_dl_utils work optimally for jax/equinox.
As such it relies heavily on common_dl_utils which can be installed using `pip install git+https://github.com/SimonKoop/common_dl_utils.git --upgrade`

The package is organized as follows:
- types: registers relevant types in the type_registry from common_dl_utils
- tree_utils: some utilities for working with pytrees
- run_utils: extends the tools from common_dl_utils.config_realization to create models or run experiments from a config together with a jax prng key.
- wandb_utils: hooks the tools from run_utils to wandb. 
- debug_utils: tools for debugging
- metrics: a collection of metrics based on the framework setup in common_dl_utils.metrics
- decorators: a collection of decorators that are useful when working with jax
- masks: a collection of masking functions for masking parts of equinox modules for use in optax optimizers (e.g. for weight decay)