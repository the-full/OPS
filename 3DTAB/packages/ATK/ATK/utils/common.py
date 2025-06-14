import logging
import datetime
import sys
import os
import random
import timeit
import functools
import platform
import socket
import yaml
import inspect
from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

import subprocess
import psutil


def get_log(filename: Optional[str] = None, propagate: bool = True) -> logging.Logger:
    def beijing(sec=None, what=None):
        beijing_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()
    logging.Formatter.converter = beijing
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(name=filename)
    logger.propagate = propagate
    if filename:
        file_handler = logging.FileHandler(filename=filename)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

    return logger

def _log(logger, log_str):
    if exists(logger):
        if logger == True:
            logger = get_log()
        logger.info(log_str)


def get_cpu_info() -> dict:
    cpu_info = {
        'physical_cores': psutil.cpu_count(logical=False),
        'total_cores': psutil.cpu_count(logical=True),
        'cpu_usage': psutil.cpu_percent(interval=1),
    }
    return cpu_info


def get_disk_info() -> dict:
    disk_info = {
        'total': psutil.disk_usage('/').total,
        'used': psutil.disk_usage('/').used,
        'free': psutil.disk_usage('/').free,
        'percent': psutil.disk_usage('/').percent,
    }
    return disk_info


def get_net_info() -> dict:
    addrs = psutil.net_if_addrs()
    net_info = {}
    for interface_name, interface_addresses in addrs.items():
        for address in interface_addresses:
            if str(address.family) == 'AddressFamily.AF_INET':
                net_info[interface_name] = {
                    'ip_address': address.address,
                    'netmask': address.netmask,
                    'broadcast_ip': address.broadcast,
                }
    return net_info


def get_sys_info() -> dict:
    sys_info = {
        'platform': platform.system(),
        'platform-release': platform.release(),
        'platform-version': platform.version(),
        'architecture': platform.machine(),
        'hostname': socket.gethostname(),
    }
    return sys_info


def get_process_info(process_id: int) -> Optional[dict]:
    try:
        process = psutil.Process(process_id)
        return {
            'username': process.username(),
            'status': process.status(),
            'create_time': process.create_time(),
            'cpu_times': process.cpu_times(),
            'memory_info': process.memory_info(),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def get_nvidia_smi_output() -> list:
    def get_process_info(process_id: int) -> Optional[tuple]:
        try:
            process = psutil.Process(process_id)
            return process.username(), process.status()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
    try:
        result = subprocess.run(['nvidia-smi',
                                 '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.total,memory.used,memory.free,gpu_uuid',
                                 '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True)
        gpu_lines = result.stdout.strip().split('\n')
        gpu_info_list = [line.split(', ') for line in gpu_lines]
        gpu_uuid_index_map = {line[-1]: line[0] for line in gpu_info_list}
        result = subprocess.run(['nvidia-smi',
                                 '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                 '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True)

        index_process_dict = {}
        if result.stdout != '':
            pid_lines = result.stdout.strip().split('\n')
            for i in range(len(pid_lines)):
                pid, uuid, used_mem = pid_lines[i].strip().split(', ')
                user, proc_status = get_process_info(int(pid)) if pid != 'N/A' else None
                index = gpu_uuid_index_map[uuid]
                if index not in index_process_dict:
                    index_process_dict[index] = []
                index_process_dict[index].append([user, used_mem, proc_status])

        for index, info in index_process_dict.items():
            gpu_info_list[int(index)].append(info)
        return gpu_info_list
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def show_gpu_status(logger: Optional[logging.Logger] = None):
    if logger == None:
        logger = get_log()
    gpu_info_list = get_nvidia_smi_output()
    if gpu_info_list:
        logger.info('GPU Status:')
        for gpu_info in gpu_info_list:
            info_str = '[{}] | {} | {:>3} °C, {:>3} % | T/U/F: {:>5} / {:>5} / {:>5} MB | '.format(*gpu_info)
            if isinstance(gpu_info[-1], list):
                for proc_info in gpu_info[-1]:
                    info_str += '{}({}M)->{} '.format(*proc_info)
            logger.info(info_str)
    else:
        logger.info('Unable to retrieve GPU status.')


def show_env(logger: Optional[logging.Logger] = None):
    if logger == None:
        logger = get_log()
    logger.info('The version information:')
    logger.info('Python: {}'.format(''.join(sys.version.split('\n'))))
    logger.info(f'PyTorch: {torch.__version__}')
    if torch.cuda.is_available() == True:
        logger.info('CUDA is available')
        show_gpu_status(logger)
    else:
        logger.info('CUDA is not available')


def timeit_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timings = timeit.repeat(lambda: func(*args, **kwargs), repeat=3, number=100)
        mean_time = sum(timings) / len(timings)
        print(f"{func.__name__} 运行时间：{mean_time / 100} 秒（最小值）")
        return func(*args, **kwargs)
    return wrapper


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def set_seed(seed: int = 2024, need_consist: bool = True, log: Optional[logging.Logger] = None):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if need_consist == True:
        torch.backends.cudnn.benchmark = False # type: ignore
        torch.backends.cudnn.deterministic = True # type: ignore

    _log(log, f'The random seed is fixed to {seed}')


def show_model_param(model):
    print("Model structure: ", model, "\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values: \n {param[:2]} \n")


def mkdir_if_not_exist(dir_: str, log: Optional[logging.Logger] = None):
    if os.path.exists(dir_) == False:
        _log(log, f'create directory: {dir_}')
        os.makedirs(dir_)


def load_weights_to_model(model, weight_path, k: Optional[int] = None, l: Optional[int] = None, strict: bool = True):
    pretrained_state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    if pretrained_state_dict.get('model') is not None:
        pretrained_state_dict = pretrained_state_dict.get('model')
    if pretrained_state_dict.get('state_dict') is not None:
        pretrained_state_dict = pretrained_state_dict.get('state_dict')

    model_state_dict = model.state_dict()
    loaded_state_dict = {}

    layer_count = min(len(model_state_dict), len(pretrained_state_dict))
    if k is not None and l is not None:
        index = range(k, l)
    elif k is None and l is not None:
        index = range(l)
    elif k is not None and l is None:
        index = range(k, layer_count)
    else:
        index = range(layer_count)

    for i in index:
        model_key = list(model_state_dict.keys())[i]
        pretrained_key = list(pretrained_state_dict.keys())[i]
        pretrained_weight = pretrained_state_dict[pretrained_key]

        if model_state_dict[model_key].shape != pretrained_weight.shape:
            if strict:
                raise ValueError(f"Shape mismatch for layer {i}: {model_key} vs {pretrained_key}")
            else:
                print(f"Skipping layer {i} due to shape mismatch: {model_key} vs {pretrained_key}")
                continue

        loaded_state_dict[model_key] = pretrained_weight

    if not strict and len(loaded_state_dict) < len(model_state_dict):
        print(f"Warning: Some layers in your model did not receive weights from the pretrained model.")

    if k is not None and k < len(model_state_dict):
        for i in range(k, len(model_state_dict)):
            model_key = list(model_state_dict.keys())[i]
            if len(model_state_dict[model_key].shape) <= 1:
                nn.init.zeros_(model_state_dict[model_key])
            else:
                nn.init.kaiming_normal_(model_state_dict[model_key], nonlinearity='relu')

    model.load_state_dict(loaded_state_dict, strict=False)

    return model


def merge_cfg_to_cfg(user_config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    merged_config = default_config.copy()
    for key in user_config:
        if key not in default_config:
            raise KeyError(f"配置错误: '{key}' 不是有效的配置键")
        merged_config[key] = user_config[key]
    return merged_config


def load_cfg_from_yaml(cfg_path):
    with open(cfg_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_cls_position(cls_obj):
    class_file = inspect.getsourcefile(cls_obj)
    return class_file

def setup_ddp(local_rank=None, logger=None):
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    _log(logger, f"[{os.getpid()}] (rank {dist.get_rank()}) training...")

def cleanup_ddp():
    dist.destroy_process_group()

def logger_info(out_str: str, local_rank: Optional[int], logger=None) -> None:
    if local_rank is not None:
        out_str = f"(rank {local_rank}): " + out_str
    _log(logger, out_str)

def print_gpu_memory_usage():
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1e+9 
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1e+9  
        logger = get_log()
        logger.info(f"Current GPU memory usage: Allocated - {gpu_memory_allocated:.2f} GB, Reserved - {gpu_memory_reserved:.2f} GB")
    else:
        print("No available GPU in the current environment")

def to_tensor(data):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    return data

def config_to_yaml_file(config, file_path, force=False):
    if os.path.exists(file_path) and not force:
        pass
    else:
        yaml_str = yaml.dump(config, indent=2, width=100, default_flow_style=False)

        with open(file_path, 'w') as yaml_file:
            yaml_file.write(yaml_str)

def do_if_not_none(var, func, *args, **kwargs):
    if var is not None:
        func(*args, **kwargs)


# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union, Any
def is_method_overridden(method: str, base_class: type,
                         derived_class: Union[type, Any]) -> bool:
    """Check if a method of base class is overridden in derived class.

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
def print_log(
    msg, 
    logger: Optional[Union[logging.Logger, str]] = None,
    level=logging.INFO
):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:

            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif logger == 'current':
        logger_instance = get_log()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        # If the type of `logger` is `str`, but not with value of `current` or
        # `silent`, we assume it indicates the name of the logger. If the
        # corresponding logger has not been created, `print_log` will raise
        # a `ValueError`.
        warnings.warn('get_log does not consider this scenario, and its usage here may be problematic')
        logger_instance = get_log(logger)
        logger_instance.log(level, msg)
    else:
        raise TypeError(
            '`logger` should be either a logging.Logger object, str, '
            f'"silent", "current" or None, but got {type(logger)}')

def is_cuda_available():
    return torch.cuda.is_available()

def return_first_item(result):
    if isinstance(result, tuple):
        return result[0]
    else:
        return result
    
def check_option(op, op_list):
    if op not in op_list:
        raise Exception("Unsupported option {} in {}".format(op, op_list))


from typing import Union, Dict, Iterable, Dict, Mapping
from torch import Tensor
def cast_data_device(
    data: Union[Tensor, Mapping, Iterable],
    device: torch.device,
    out: Optional[Union[Tensor, Mapping, Iterable]] = None
) -> Union[Tensor, Mapping, Iterable]:
    if out is not None:
        if type(data) != type(out):
            raise TypeError(
                'out should be the same type with data, but got data is '
                f'{type(data)} and out is {type(data)}')

        if isinstance(out, set):
            raise TypeError('out should not be a set')

    if isinstance(data, Tensor):
        if get_data_device(data) == device:
            data_on_device = data
        else:
            data_on_device = data.to(device)

        if out is not None:
            # modify the value of out inplace
            out.copy_(data_on_device)  # type: ignore

        return data_on_device

    elif isinstance(data, Mapping):
        data_on_device = {}
        if out is not None:
            data_len = len(data)
            out_len = len(out)  # type: ignore
            if data_len != out_len:
                raise ValueError('length of data and out should be same, '
                                 f'but got {data_len} and {out_len}')

            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device,
                                                     out[k])  # type: ignore
        else:
            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device)

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        # To ensure the type of output as same as input, we use `type(data)`
        # to wrap the output
        return type(data)(data_on_device)  # type: ignore

    elif isinstance(data, Iterable) and not isinstance(
            data, str) and not isinstance(data, np.ndarray):
        data_on_device = []
        if out is not None:
            for v1, v2 in zip(data, out):
                data_on_device.append(cast_data_device(v1, device, v2))
        else:
            for v in data:
                data_on_device.append(cast_data_device(v, device))

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        return type(data)(data_on_device)  # type: ignore
    else:
        raise TypeError('data should be a Tensor, list of tensor or dict, '
                        f'but got {data}')


def get_data_device(data: Union[Tensor, Mapping, Iterable]) -> torch.device:
    if isinstance(data, Tensor):
        return data.device
    elif isinstance(data, Mapping):
        pre = None
        for v in data.values():
            cur = get_data_device(v)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        'device type in data should be consistent, but got '
                        f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    elif isinstance(data, Iterable) and not isinstance(data, str):
        pre = None
        for item in data:
            cur = get_data_device(item)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        'device type in data should be consistent, but got '
                        f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    else:
        raise TypeError('data should be a Tensor, sequence of tensor or dict, '
                        f'but got {data}')

