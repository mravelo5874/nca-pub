from argparse import Namespace, ArgumentParser
import scripts.nca.perception as _perception
import scripts.nca.trainer as _trainer
from colorama import init, Style, Fore
import scripts.nca.model as _model
from types import ModuleType
import numpy as np
import torch
import os

init()
PROGRAM = f'{Style.DIM}[{os.path.basename(__file__)}]{Style.RESET_ALL}'
LOG_FILE_PATH = None
DEFAULT_PAD = 2

def set_log_file_path(_path: str) -> None:
    global LOG_FILE_PATH
    LOG_FILE_PATH = _path

def log(log_line: str, _print: bool = True) -> None:
    if LOG_FILE_PATH is None: 
        print (f'{PROGRAM} {Fore.RED}error!{Style.RESET_ALL} log file path not set - use {Fore.YELLOW}set_log_file_path(){Style.RESET_ALL} before calling {Fore.YELLOW}log(){Style.RESET_ALL}')
        return
    if _print: print(log_line)
    with open(LOG_FILE_PATH, 'a') as file:
        file.write(f'{log_line}\n')

def parse_train_nca_args() -> Namespace:
    parser = ArgumentParser()
    # model specific arguments
    parser.add_argument('--name', '-n', help=f'{Fore.WHITE}the name given to trained nca model{Style.RESET_ALL}', type=str)
    parser.add_argument('--target', '-v', help=f'{Fore.WHITE}path to .vox model to use as the model\'s target structure{Style.RESET_ALL}', type=str)
    parser.add_argument('--seed', '-s', help=f'{Fore.WHITE}path to .vox model to use as the model\'s starting seed{Style.RESET_ALL}', type=str)
    parser.add_argument('--perception', '-p', help=f'{Fore.WHITE}the perception type the model will use - perception types: {Fore.YELLOW}{pretty_print_str_list(get_module_valid_classes(_perception))}{Style.RESET_ALL}', type=str)
    parser.add_argument('--model', '-m', help=f'{Fore.WHITE}the type of nca model to train - model types: {Fore.YELLOW}{pretty_print_str_list(get_module_valid_classes(_model))}{Style.RESET_ALL}', type=str)
    parser.add_argument('--trainer', '-t', help=f'{Fore.WHITE}the training regimen to use on the model - trainer types: {Fore.YELLOW}{pretty_print_str_list(get_module_valid_classes(_trainer))}{Style.RESET_ALL}', type=str)
    parser.add_argument('--channels', '-c', help=f'{Fore.WHITE}the number of channels per cell\'s state vector (including rgba){Style.RESET_ALL}', default=16, type=int)
    parser.add_argument('--hidden', '-d', help=f'{Fore.WHITE}the number of hidden channels used in the neural update{Style.RESET_ALL}', default=128, type=int)
    parser.add_argument('--stochastic_rate', '-sr', help=f'{Fore.WHITE}the chance each cell has to be updated each update step{Style.RESET_ALL}', default=0.5, type=float)
    # training specific arguments
    parser.add_argument('--epochs', '-e', help=f'{Fore.WHITE}the number of epochs to train for{Style.RESET_ALL}', default=10_000, type=int)
    parser.add_argument('--pool_size', '-o', help=f'{Fore.WHITE}the size of the training pool{Style.RESET_ALL}', default=64, type=int)
    parser.add_argument('--batch_size', '-b', help=f'{Fore.WHITE}the number of models to take from the pool each epoch{Style.RESET_ALL}', default=4, type=int)
    parser.add_argument('--start_lr', '-slr', help=f'{Fore.WHITE}the starting learning rate{Style.RESET_ALL}', default=1e-3, type=float)
    parser.add_argument('--end_lr', '-elr', help=f'{Fore.WHITE}the ending learning rate{Style.RESET_ALL}', default=1e-5, type=float)
    parser.add_argument('--factor_sched', '-fs', help=f'{Fore.WHITE}the factor schedule used for lr-optimization{Style.RESET_ALL}', default=0.5, type=float)
    parser.add_argument('--patience_sched', '-ps', help=f'{Fore.WHITE}the patience schedule used for lr-optimization{Style.RESET_ALL}', default=500, type=int)
    parser.add_argument('--damage_num', '-dn', help=f'{Fore.WHITE}the number of (lowest loss) models in a batch to apply damage to{Style.RESET_ALL}', default=2, type=int)
    parser.add_argument('--damage_rate', '-dr', help=f'{Fore.WHITE}the rate at which to apply damage (every x epochs){Style.RESET_ALL}', default=5, type=int)
    # logging specific arguments
    parser.add_argument('--log_file', '-l', help=f'{Fore.WHITE}the log file (created within model directory) to write to during training{Style.RESET_ALL}', default='trainlog.txt', type=str)
    parser.add_argument('--info_rate', '-i', help=f'{Fore.WHITE}the rate at which to print out training information{Style.RESET_ALL}', default=100, type=int)
    return parser.parse_args()

def assert_train_nca_args(args: Namespace) -> any:
    missing_args = []
    if args.name is None: missing_args.append('--name')
    if args.target is None: missing_args.append('--target')
    if args.seed is None: missing_args.append('--seed')
    if args.perception is None: missing_args.append('--perception')
    if args.model is None: missing_args.append('--model')
    if args.trainer is None: missing_args.append('--trainer')
    if args.channels is None: missing_args.append('--channels')
    if args.hidden is None: missing_args.append('--hidden')
    if args.stochastic_rate is None: missing_args.append('--stochastic_rate')
    if args.epochs is None: missing_args.append('--epochs')
    if args.pool_size is None: missing_args.append('--pool_size')
    if args.batch_size is None: missing_args.append('--batch_size')
    if args.start_lr is None: missing_args.append('--start_lr')
    if args.end_lr is None: missing_args.append('--end_lr')
    if args.factor_sched is None: missing_args.append('--factor_sched')
    if args.patience_sched is None: missing_args.append('--patience_sched')
    if args.damage_num is None: missing_args.append('--damage_num')
    if args.damage_rate is None: missing_args.append('--damage_rate')
    if args.log_file is None: missing_args.append('--log_file')
    if args.info_rate is None: missing_args.append('--info_rate')
    if len(missing_args) > 0: return False, missing_args
    return True, []

def pretty_print_str_list(_list: list[str]) -> str:
    list_str = ''
    for x in _list: list_str += f'{x}, '
    return list_str[:-2]

def pretty_print_args(_args: Namespace) -> str:
    args_str = '\n'
    for a in _args.__dict__: args_str += f'{Fore.WHITE} - {a}: {Fore.CYAN}{_args.__dict__[a]}{Style.RESET_ALL}\n'
    return args_str[:-1]

def render_tensor_as_vox(_tensor: torch.Tensor):
    from scripts.vox.vox import vox
    vox().load_from_tensor(_tensor).render()

def load_vox_as_tensor(_vox_model_path: str):
    from scripts.vox.vox import vox
    if _vox_model_path.endswith('vox'):
        target = vox().load_from_file(_vox_model_path)
        target_ten = target.tensor()
    elif _vox_model_path.endswith('npy'):
        with open(_vox_model_path, 'rb') as f:
            target_ten = torch.from_numpy(np.load(f))
    return target_ten

def generate_pool(_args: Namespace, _seed: torch.Tensor, _isotype: int):
    pool_size = _args.pool_size
    size = _seed.shape[-1]
    with torch.no_grad():
        pool = _seed.clone().repeat(pool_size, 1, 1, 1, 1)
        # * randomize channel(s)
        if _isotype == 1:
            for j in range(pool_size):
                pool[j, -1:] = torch.rand(size, size, size)*np.pi*2.0
        elif _isotype == 3:
            for j in range(pool_size):
                pool[j, -1:] = torch.rand(size, size, size)*np.pi*2.0
                pool[j, -2:-1] = torch.rand(size, size, size)*np.pi*2.0
                pool[j, -3:-2] = torch.rand(size, size, size)*np.pi*2.0
    return pool

def half_volume_mask(_size, _type):
    mask_types = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-', 'rand']
    if _type == 'rand':
        _type = mask_types[np.random.randint(0, 6)]
    mat = np.zeros([_size, _size, _size])
    half = _size//2
    if _type == 'x+':
        mat[:half, :, :] = 1.0
    elif _type == 'x-':
        mat[-half:, :, :] = 1.0
    if _type == 'y+':
        mat[:, :half, :] = 1.0
    elif _type == 'y-':
        mat[:, -half:, :] = 1.0
    if _type == 'z+':
        mat[:, :, :half] = 1.0
    elif _type == 'z-':
        mat[:, :, -half:] = 1.0
    return mat > 0.0

# thanks to: https://stackoverflow.com/questions/5520580/how-do-you-get-all-classes-defined-in-a-module-but-not-imported
def get_module_valid_classes(_module: ModuleType) -> list[str]:
    md = _module.__dict__
    return [c for c in md if (isinstance(md[c], type) and md[c].__module__ == _module.__name__)]

def get_perception(_perception_name: str) -> ModuleType:
    valid_perceptions = get_module_valid_classes(_perception)
    if _perception_name not in valid_perceptions:
        print (f'{PROGRAM} {Fore.RED}error!{Style.RESET_ALL} invalid perception type:',
            f'{Fore.YELLOW}\'{_perception_name}\'{Style.RESET_ALL} - try using one of the following:',
            f'{Fore.YELLOW}{pretty_print_str_list(valid_perceptions)}{Style.RESET_ALL}')
        return None
    for c in _perception.__dict__: 
        if c == _perception_name: return  _perception.__dict__[c]
    return None

def get_model(_model_name: str) -> ModuleType:
    valid_models = get_module_valid_classes(_model)
    if _model_name not in valid_models: 
        print (f'{PROGRAM} {Fore.RED}error!{Style.RESET_ALL} invalid model type:',
            f'{Fore.YELLOW}\'{_model_name}\'{Style.RESET_ALL} - try using one of the following:',
            f'{Fore.YELLOW}{pretty_print_str_list(valid_models)}{Style.RESET_ALL}')
        return None
    for c in _model.__dict__:
        if c == _model_name: return  _model.__dict__[c]
    return None

def get_trainer(_trainer_name: str) -> ModuleType:
    valid_trainers = get_module_valid_classes(_trainer)
    if _trainer_name not in valid_trainers: 
        print (f'{PROGRAM} {Fore.RED}error!{Style.RESET_ALL} invalid trainer type:',
            f'{Fore.YELLOW}\'{_trainer_name}\'{Style.RESET_ALL} - try using one of the following:',
            f'{Fore.YELLOW}{pretty_print_str_list(valid_trainers)}{Style.RESET_ALL}')
        return None
    for c in _trainer.__dict__:
        if c == _trainer_name: return  _trainer.__dict__[c]
    return None

# * yoinked (and modified) from https://stackoverflow.com/questions/54616049/converting-a-rotation-matrix-to-euler-angles-and-back-special-case
def eul2rot(ax, ay, az):
    import torch
    r11 = torch.cos(ay)*torch.cos(az)
    r12 = torch.sin(ax)*torch.sin(ay)*torch.cos(az) - torch.sin(az)*torch.cos(ax)
    r13 = torch.sin(ay)*torch.cos(ax)*torch.cos(az) + torch.sin(ax)*torch.sin(az)
    
    r21 = torch.sin(az)*torch.cos(ay)
    r22 = torch.sin(ax)*torch.sin(ay)*torch.sin(az) + torch.cos(ax)*torch.cos(az)
    r23 = torch.sin(ay)*torch.sin(az)*torch.cos(ax) - torch.sin(ax)*torch.cos(az)
    
    r31 = -torch.sin(ax)
    r32 = torch.sin(ax)*torch.cos(ay)
    r33 = torch.cos(ax)*torch.cos(ay)
    
    b, c = ax.shape
    import numpy as np
    R = torch.tensor(np.zeros([b, c, 3, 3]), dtype=torch.float)
    
    R[:, :, 0, 0] = r11
    R[:, :, 0, 1] = r12
    R[:, :, 0, 2] = r13
    
    R[:, :, 1, 0] = r21
    R[:, :, 1, 1] = r22
    R[:, :, 1, 2] = r23
    
    R[:, :, 2, 0] = r31
    R[:, :, 2, 1] = r32
    R[:, :, 2, 2] = r33
