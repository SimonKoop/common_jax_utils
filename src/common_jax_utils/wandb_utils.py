""" 
This module provides a function to run an experiment from a wandb config.
It uses the tools from run_utils and from common_dl_utils to create the model and experiment from the wandb config.
This is particularly useful for running hyperparameter sweeps.
"""
import random
from typing import Union
from types import ModuleType
import warnings
import pprint
import gc
import sys
import traceback

import jax

import common_dl_utils as cdu
from common_jax_utils import run_utils
from common_dl_utils.type_registry import type_registry
from common_dl_utils.module_loading import MultiModule


def run_from_wandb(  # mostly meant to be wrapped in a partial for actual use
        model_kwarg_in_trainer:str="model",
        model_prompt:str = "model_type",
        trainer_prompt:str = "trainer_type",
        model_default_module_key:str="architecture",
        trainer_default_module_key:Union[str, None]="trainer_module",
        registry:list=type_registry,
        model_keys:Union[None, list[Union[str, tuple[str,...]]]]=None, # usually, you don't need this
        trainer_keys:Union[None, list[Union[str, tuple[str,...]]]]=None, # usually, you don't need this
        missing_model_kwargs:Union[None, dict]=None,
        missing_trainer_kwargs:Union[None, dict]=None,
        sub_config_postfix:str = '_config',
        sub_config_from_param_name:bool = True,
        model_sub_config_name_base:Union[None, str]=None,
        trainer_sub_config_name_base:Union[None, str]=None,
        add_model_module_to_architecture_default_module:bool=True,
        additional_architecture_default_modules:Union[None, list[ModuleType], MultiModule]=None,
        additional_trainer_default_modules:Union[None, list[ModuleType], MultiModule]=None,
        prng_key_kwarg_name:str='key',
        prng_seed_key:str='prng_seed',
        trainer_activation_method:Union[None, str]=None,
        provide_prng_key_to_trainer_activation_method:bool=True,
        raise_exception_on_cpu:bool=True
        ):  # meant to be run by a wandb agent
    """
    Run an experiment from wandb.
    This is meant to be run by a wandb agent during a hyperparameter sweep.
    :parameter model_kwarg_in_trainer: str, the argument/parameter in the trainer class that specifies the model
    :parameter model_prompt: str, the key in config that specifies the model class
        i.e. prompt=config[model_prompt] should point to where the model class is located
    :parameter trainer_prompt: str, the key in config that specifies the trainer class
        i.e. prompt=config[trainer_prompt] should point to where the trainer class/function is located
    :parameter model_default_module_key: str, optional key in config pointing to a module containing architectures (e.g. for encoders, decoders, etc.)
        if not None, config[model_default_module_key] should be a path to a python file or package
    :parameter trainer_default_module_key: str, optional key in config pointing to a module/package containing classes relevant for the trainer
        e.g. optimizers, schedulers, etc.
        if not None, config[trainer_default_module_key] should be a path to a python file or package
    :parameter registry: list, list of types that require initialization (or just calling) based on config before being returned
        defaults to common_dl_utils.type_registry.type_registry
        Note that this default type_registry has been filled by common_jax_utils.types
    :parameter model_keys: list, optional list of keys in config that contain sub-configs for the model
        keys are in order of ascending priority, i.e. the last key in the list has the highest priority
        if B has higher priority than A, parameters are pulled from B first, and only missing parameters will be pulled from A (etc.)
        A key can either be a string or a tuple of strings
        in case of a tuple of strings, it's treated as config[key[0]][key[1]]...
        NB these are *not* keys used for PRNG but dictionary keys
    :parameter trainer_keys: list, optional list of keys in config that contain sub-configs for the trainer
        see model_keys for more details
    :parameter missing_model_kwargs: dict, optional dict of keyword arguments to be passed to the model class (can overrule the values specified in config)
    :parameter missing_trainer_kwargs: dict, optional dict of keyword arguments to be passed to the trainer class (can overrule the values specified in config)
    :parameter sub_config_postfix: str, string used for finding names of subconfigs
    :parameter sub_config_from_param_name: bool, if True, uses the parameter name to look for sub-configs, otherwise uses the class name
    :parameter model_sub_config_name_base: str, optional string to use for finding the sub-config for this model
        if provided, will look for f'{model_sub_config_name_base}{sub_config_postfix}'
        otherwise, will look for f'{class_name}{sub_config_postfix}', where class_name is the class name specified in the model prompt in the config.
        NB if the config specifies a dotted path such as 'ModelClass.from_config', the first part of this dotted path, i.e. 'ModelClass', will be used as the class_name
    :parameter trainer_sub_config_name_base: str, optional string to use for finding the sub-config for this trainer
        if provided, will look for f'{trainer_sub_config_name_base}{sub_config_postfix}'
        otherwise, will look for f'{class_name}{sub_config_postfix}', where class_name is the class name specified in the trainer prompt in the config.
        NB if the config specifies a dotted path such as 'TrainerClass.some_class_method', the first part of this dotted path, i.e. 'TrainerClass', will be used as the class_name
    :parameter add_model_module_to_architecture_default_module: bool, if True, will add the model module to the default module specified by model_default_module_key
        NB in this case, the model_prompt should be a tuple or list of length 2, where the first element is the path to the model module
            so set this to False if you provide the module for the model class through model_default_module_key 
            or through additional_architecture_default_modules
    :parameter additional_architecture_default_modules: optional additional modules to be added to the default module specified by model_default_module_key
    :parameter additional_trainer_default_modules: optional additional modules to be added to the default module specified by trainer_default_module_key
        If the trainer_default_module_key was None, and additional_trainer_default_modules is not None, the latter will simply be used as the default module
        If a list of multiple modules is provided, a common_dl_utils.module_loading.MultiModule instance will be created from them
    :parameter prng_key_kwarg_name: str, argument/kwarg name that indicates a class requires a key for pseudo random number generation (default 'key')
        this is used for provide_prng_keys_if_needed
    :parameter prng_seed_key: str, key in config that specifies the seed for the pseudo random number generator
        if not found in config, a random seed will be generated using random.randint(0, 2**31-1) and logged to wandb
    :parameter trainer_activation_method: str, optional name of a method in the trainer class that should be called after initialization
        this is useful for starting the training loop
        if None, no method will be called
            NB if the trainer is a function instead of an instance of some Trainer class, it will be called already due to initialize=True
            in this case, it is better to set this to None
    :parameter provide_prng_key_to_trainer_activation_method: bool, if True, provides a prng key to the trainer activation method
        if False, the trainer activation method will be called without a prng key
    :parameter raise_exception_on_cpu: bool, if True, raises a RuntimeError if jax is running on cpu
    """
    import wandb  # import inside function so that the rest of common_jax_utils can still be used if wandb is not installed
    try:
        with wandb.init():
            wandb_config = wandb.config
            config = cdu.config_creation.make_nested_config(wandb_config)
            wandb.log({'config': pprint.pformat(config)})
            jax.print_environment_info()  # for debugging and bug reporting

            if prng_seed_key in config:
                prng_seed = config[prng_seed_key]
            else:
                prng_seed = random.randint(0, 2**31-1)
                wandb.log({'prng_seed': prng_seed})
                warnings.warn(f"{prng_seed_key} not found in config, using {prng_seed} as prng seed")
            try:
                key = jax.random.PRNGKey(prng_seed).block_until_ready()
                print(f"Created {key=} for prng from {prng_seed=}.")
            except Exception as e:
                print(f"The following exception arose during creation of the PRNGKey: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                raise e

            default_device = jax.default_backend()
            if default_device == 'cpu' and raise_exception_on_cpu:
                raise RuntimeError(f"Jax is running on cpu ({jax.default_backend()=}) but {raise_exception_on_cpu=}.")
            print("Start loading and running experiment")
            print(f"    Running on {default_device}")
            sys.stdout.flush()
            
            if trainer_activation_method is not None and provide_prng_key_to_trainer_activation_method:
                initialization_key, run_key = jax.random.split(key, 2)
            else:
                initialization_key, run_key = key, None

            # time for work
            experiment = run_utils.get_experiment_from_config_and_key(
                prng_key=initialization_key,
                config=config,
                model_kwarg_in_trainer=model_kwarg_in_trainer,
                model_prompt=model_prompt,
                trainer_prompt=trainer_prompt,
                model_default_module_key=model_default_module_key,
                trainer_default_module_key=trainer_default_module_key,
                registry=registry,
                model_keys=model_keys,
                trainer_keys=trainer_keys,
                missing_model_kwargs=missing_model_kwargs,
                missing_trainer_kwargs=missing_trainer_kwargs,
                sub_config_postfix=sub_config_postfix,
                sub_config_from_param_name=sub_config_from_param_name,
                model_sub_config_name_base=model_sub_config_name_base,
                trainer_sub_config_name_base=trainer_sub_config_name_base,
                add_model_module_to_architecture_default_module=add_model_module_to_architecture_default_module,
                additional_architecture_default_modules=additional_architecture_default_modules,
                additional_trainer_default_modules=additional_trainer_default_modules,
                initialize=True,
                prng_key_kwarg_name=prng_key_kwarg_name
            )
            if trainer_activation_method is not None:
                # if the trainer is a function instead of an instance of some Trainer class,
                # it is called already due to initialize=True
                activation_method = getattr(experiment, trainer_activation_method)
                if provide_prng_key_to_trainer_activation_method:
                    activation_method(run_key)
                else:
                    activation_method()
    finally:  # this was a desperate attempt to solve some memory issues with wandb. It failed. 
        sys.stdout.flush()
        wandb.teardown()
        gc.collect()
