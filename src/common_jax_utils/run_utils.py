""" 
Utilities for creating models and loading experiments from config dictionaries.
This extends the tools from common_dl_utils.config_realization to create models or run experiments from a config together with a jax prng key.
Note that the keys referred to by many parameters are *not* prng keys but dictionary keys. This is to keep in line with common_dl_utils.config_realization.
Whenever prng keys are meant, this is made explicit.
"""
from typing import Union
from types import ModuleType
from typing import Any

import jax

from common_jax_utils import key_generator
from common_dl_utils.type_registry import type_registry
from common_dl_utils.config_realization import PostponedInitialization, get_model_from_config, prep_class_from_config
from common_dl_utils.module_loading import MultiModule, load_from_path


def provide_prng_keys_if_needed(recipient:PostponedInitialization, prng_key: jax.Array, expected_kwarg_name='key'):
    """
    provide PostponedInitialization instances with prng keys if necessary
    NB this operation happens in-place i.e. this is *not* a pure function

    :param recipient: the PostponedInitialization instance potentially requiring prng keys
    :param prng_key: the prng key to use for the creation of further keys
        NB this key is not directly given to the recipient. 
        It is first split in two and the right half is given to the recipient
    :param expected_kwarg_name: the argument name to look for, defaults to 'key'
        any *missing* argument with this name will be provided with a key
        this is done recursively (as recipient may have stored kwargs also requiring prng keys)
    :return: None, as this is an in-place operation
    """
    key_gen = key_generator(prng_key)
    # first provide a key to recipient itself if it needs one
    if expected_kwarg_name in recipient.missing_args:
        recipient.resolve_missing_args({expected_kwarg_name: next(key_gen)})
    
    # now recipient may have PostponedInitialization instances among its stored kwargs
    # which might also need to be provided with keys
    for value in recipient.kwargs.values():
        if isinstance(value, PostponedInitialization):
            provide_prng_keys_if_needed(value, next(key_gen), expected_kwarg_name=expected_kwarg_name)
        elif isinstance(value, (tuple, list)):
            for v in value:
                if isinstance(v, PostponedInitialization):
                    provide_prng_keys_if_needed(v, next(key_gen), expected_kwarg_name=expected_kwarg_name)


def get_model_from_config_and_key(
        prng_key: jax.Array,
        config:dict,
        model_prompt:str="model_type", 
        default_module_key:str="architecture", 
        registry:list=type_registry, 
        keys:Union[None, list[Union[str, tuple[str,...]]]]=None, 
        missing_kwargs:Union[None, dict]=None,
        sub_config_postfix:str = '_config',
        sub_config_from_param_name:bool = True,
        model_sub_config_name_base:Union[None, str]=None,
        add_model_module_to_architecture_default_module:bool=True,
        additional_architecture_default_modules:Union[None, tuple[ModuleType], list[ModuleType], ModuleType]=None,
        initialize:bool=True,
        prng_key_kwarg_name:str='key'
    ):
    """
    A wrapper around common_dl_utils.config_realization.get_model_from_config
    meant for dealing with the need to explicitly provide prng keys to the model components

    :param prng_key: prng key used for providing the model with prng keys for initialization
    :param config: config dict specifying the model
    :param model_prompt: key in config. prompt=config[model_prompt] should point to where the model class is located 
    :param default_module_key: optional key in config pointing to a module containing architectures (e.g. for encoders, decoders, etc.)
        if not None, config[default_module_key] should be a path to a python file or package
    :param registry: list of types that require initialization (or just calling) based on config before being returned
        defaults to common_dl_utils.type_registry.type_registry
    :param keys: optional list of keys in config that contain sub-configs
        keys are in order of ascending priority, i.e. the last key in the list has the highest priority
        if B has higher priority than A, parameters are pulled from B first, and only missing parameters will be pulled from A (etc.)
        A key can either be a string or a tuple of strings
        in case of a tuple of strings, it's treated as config[key[0]][key[1]]...
        NB these are *not* keys used for PRNG but dictionary keys 
    :param missing_kwargs: optional dict of keyword arguments to be passed to the model class (can overrule the values specified in config)
    :param sub_config_postfix: string used for finding names of subconfigs 
    :param sub_config_from_param_name: if True, uses the parameter name to look for sub-configs, otherwise uses the class name
    :param model_sub_config_name_base: optional string to use for finding the sub-config for this model
        if provided, will look for f'{model_sub_config_name_base}{sub_config_postfix}'
        otherwise, will look for f'{class_name}{sub_config_postfix}', where class_name is the class name specified in the model prompt in the config.
    :param add_model_module_to_architecture_default_module: if True, will add the model module to the default module specified by default_module_key
        NB in this case, the model_prompt should be a tuple or list of length 2, where the first element is the path to the model module
    :param additional_architecture_default_modules: optional additional modules to be added to the default module specified by default_module_key
    :param initialize: if True, initialize the model before returning it, otherwise return a PostponedInitialization instance
    :param prng_key_kwarg_name: argument/kwarg name that indicates a class requires a key for pseudo random number generation (default: 'key')
    :returns: an initialized model as specified by config and missing_kwargs if initialize is True, else a PostponedInitialization instance
    """
    un_initialized_model = get_model_from_config(
        config=config,
        model_prompt=model_prompt,
        default_module_key=default_module_key,
        registry=registry,
        keys=keys,
        missing_kwargs=missing_kwargs,
        sub_config_postfix=sub_config_postfix,
        sub_config_from_param_name=sub_config_from_param_name,
        model_sub_config_name_base=model_sub_config_name_base,
        add_model_module_to_architecture_default_module=add_model_module_to_architecture_default_module,
        additional_architecture_default_modules=additional_architecture_default_modules,
        initialize=False
    )
    provide_prng_keys_if_needed(
        recipient=un_initialized_model,
        prng_key=prng_key,
        expected_kwarg_name=prng_key_kwarg_name
        )
    if initialize:
        return un_initialized_model.initialize()
    return un_initialized_model

def get_experiment_from_config_and_key(
        prng_key:jax.Array,
        config: dict,
        model_kwarg_in_trainer:str="model",
        model_prompt:str = "model_type",
        trainer_prompt:str = "trainer_type",
        model_default_module_key:str="architecture",
        trainer_default_module_key:Union[str, None]=None,
        registry:list=type_registry,
        model_keys:Union[None, list[Union[str, tuple[str,...]]]]=None, # usually, you don't need this
        trainer_keys:Union[None, list[Union[str, tuple[str,...]]]]=None,  # usually, you don't need this
        missing_model_kwargs:Union[None, dict]=None,  # for overriding config
        missing_trainer_kwargs:Union[None, dict]=None,  # for overriding config
        sub_config_postfix:str = '_config',
        sub_config_from_param_name:bool = True,
        model_sub_config_name_base:Union[None, str]=None,
        trainer_sub_config_name_base:Union[None, str]=None,
        add_model_module_to_architecture_default_module:bool=True,  # set this to False if you provide the module for the model class through model_default_module_key or through additional_architecture_default_modules
        additional_architecture_default_modules:Union[None, list[ModuleType], MultiModule]=None,
        additional_trainer_default_modules:Union[None, list[ModuleType], MultiModule]=None,  # e.g. provide optax for loading optimizers
        initialize:bool=True,
        prng_key_kwarg_name:str='key',
        )-> Union[Any, PostponedInitialization]:
    """ 
    Create a model and load an experiment from a config dict.
    :parameter prng_key: jax.Array, the prng key to use for providing keys to the model and trainer
    :parameter config: dict, the config dict specifying the model and trainer
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
    :parameter initialize: bool, if True, initialize the trainer before returning it, otherwise return a PostponedInitialization instance
    :parameter prng_key_kwarg_name: str, argument/kwarg name that indicates a class requires a key for pseudo random number generation (default 'key')
        this is used for provide_prng_keys_if_needed
    :returns: an initialized trainer as specified by config and missing_trainer_kwargs if initialize is True, else a PostponedInitialization instance
    """
    key_gen = key_generator(prng_key)
    # first get the model from the config
    model = get_model_from_config_and_key(
        prng_key=next(key_gen),
        config=config, 
        model_prompt=model_prompt,
        default_module_key=model_default_module_key,
        registry=registry,
        keys=model_keys,
        missing_kwargs=missing_model_kwargs,
        sub_config_postfix=sub_config_postfix,
        sub_config_from_param_name=sub_config_from_param_name,
        model_sub_config_name_base=model_sub_config_name_base,
        add_model_module_to_architecture_default_module=add_model_module_to_architecture_default_module,
        additional_architecture_default_modules=additional_architecture_default_modules,
        initialize=initialize,
        prng_key_kwarg_name=prng_key_kwarg_name
    )

    # now get the trainer
    # first the appropriate prompt and default module
    prompt = config[trainer_prompt]
    if isinstance(prompt, str) and trainer_default_module_key is None:
        raise ValueError(f"No module for trainer was specified: trainer_prompt={prompt} but trainer_default_module_key is None")
    default_modules = []
    if trainer_default_module_key is not None:
        tdm = config[trainer_default_module_key]
        if not isinstance(tdm, ModuleType):
            tdm = load_from_path(name='trainer_default_module', path=tdm)
        default_modules.append(tdm)
    if isinstance(additional_trainer_default_modules, (list, tuple)):
        default_modules += list(additional_trainer_default_modules)
    elif additional_trainer_default_modules is not None:
        default_modules.append(additional_trainer_default_modules)
    default_modules = MultiModule(*default_modules) if default_modules else None

    # now get the uninitialized trainer from config
    resolution = {model_kwarg_in_trainer: model}
    if missing_trainer_kwargs:
        resolution.update(missing_trainer_kwargs)
    uninitialized_trainer = prep_class_from_config(
        prompt=prompt,
        config=config,
        default_module=default_modules,
        registry=registry,
        keys=trainer_keys,
        new_key_postfix=sub_config_postfix,
        new_key_body=trainer_sub_config_name_base,
        new_key_base_from_param_name=sub_config_from_param_name,
        ignore_params=list(resolution.keys())
    )
    uninitialized_trainer.resolve_missing_args(resolution)
    # now provide prng keys where needed
    provide_prng_keys_if_needed(uninitialized_trainer, next(key_gen), expected_kwarg_name=prng_key_kwarg_name)
    # and return an initialized (or not) trainer
    if initialize:
        return uninitialized_trainer.initialize()
    return uninitialized_trainer

