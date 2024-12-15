from argparse import Namespace
import datetime
import torch
import os

from scripts.nca import utils
from scripts.nca.model import nca_model
from scripts.nca.trainer import nca_trainer

PROGRAM = None
MODEL_DIR = None
LOG_FILE = None
TIMESTAMP = None

def main(args: Namespace) -> None:
    global LOG_FILE
    global MODEL_DIR
    LOG_FILE = args.log_file
    MODEL_DIR = f'{args.name}-{TIMESTAMP}'
    args.model_dir = MODEL_DIR

    # create model directory
    if not os.path.exists(f'models/{MODEL_DIR}'):
        os.mkdir(f'models/{MODEL_DIR}')
    log(f'[{PROGRAM}] created model directory at: models/{MODEL_DIR}')
    log(f'[{PROGRAM}] using torch version: {torch.__version__}')
    log(f'[{PROGRAM}] using arguments: {args}')

    # prepare torch / cuda
    torch.cuda.empty_cache()
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float)

    # * create model and trainer - begin training
    model = nca_model(args)
    trainer = nca_trainer(model)
    trainer.begin()

def log(log_line: str) -> None:
    assert LOG_FILE is not None
    print(log_line)
    with open(f'models/{MODEL_DIR}/{LOG_FILE}', 'a') as file:
        file.write(f'{log_line}\n')

if __name__ == '__main__':
    PROGRAM = os.path.basename(__file__)
    TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d@%H-%M-%S')
    args = utils.parse_args()
    is_valid, missing = utils.assert_args(args)        
    if not is_valid:
        print (f'[{PROGRAM}] error! missing required argument(s): {missing}')
        print (f'[{PROGRAM}] run script with --help flag for more information')
    else:
        main(args)