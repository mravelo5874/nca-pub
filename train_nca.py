from colorama import init, Style, Fore
from argparse import Namespace
from scripts.nca import utils
import datetime
import torch
import os

init()
PROGRAM = f'{Style.DIM}[{os.path.basename(__file__)}]{Style.RESET_ALL}'
TIMESTAMP = None

def main(args: Namespace) -> None:
    # create model directory name using timestamp
    args.model_dir = f'{args.name}-{TIMESTAMP}'

    # assert that target and seed paths exist
    assert os.path.exists(args.seed)
    assert os.path.exists(args.target)

    # create model directory
    if not os.path.exists(f'models/{args.model_dir}'):
        os.mkdir(f'models/{args.model_dir}')
    
    # setup log file and print
    utils.set_log_file_path(f'models/{args.model_dir}/{args.log_file}')
    utils.log( f'{PROGRAM} created model directory at: {Fore.WHITE}models/{args.model_dir}{Style.RESET_ALL}')
    utils.log(f'{PROGRAM} using torch version: {Fore.WHITE}{torch.__version__}{Style.RESET_ALL}')
    utils.log(f'{PROGRAM} using arguments: {utils.pretty_print_args(args)}')

    # prepare torch / cuda
    if not torch.cuda.is_available():
        print (f'{PROGRAM} {Fore.RED}error!{Style.RESET_ALL} cuda is required for training - please use a machine with cuda capabilities')
        return
    torch.cuda.empty_cache()
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float)

    # create perception, model, and trainer - assert none are None
    perception = utils.get_perception(args.perception)
    assert perception is not None
    model = utils.get_model(args.model)(args, perception)
    assert model is not None
    trainer = utils.get_trainer(args.trainer)(model)
    assert trainer is not None

    # start training
    trainer.begin()

if __name__ == '__main__':
    TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d@%H-%M-%S')
    args = utils.parse_args()
    is_valid, missing = utils.assert_args(args)
    if not is_valid:
        print (f'{PROGRAM} {Fore.RED}error!{Style.RESET_ALL} missing required argument(s): {Fore.YELLOW}{utils.pretty_print_str_list(missing)}{Style.RESET_ALL}')
        print (f'{PROGRAM} run script with {Fore.WHITE}--help{Style.RESET_ALL} flag for more information')
    else:
        main(args)