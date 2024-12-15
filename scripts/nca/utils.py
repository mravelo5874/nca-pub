from argparse import Namespace, ArgumentParser
from typing import List, Tuple

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # model specific arguments
    parser.add_argument('--name', '-n', help='name given to trained nca model')
    parser.add_argument('--target', '-t', help='path to .vox model to use as the model\'s target structure')
    parser.add_argument('--seed', '-s', help='path to .vox model to use as the model\'s starting seed')
    parser.add_argument('--model', '-m', help=f'the type of nca model to train - model types: {list_model_types()}')
    parser.add_argument('--perception', '-p', help=f'the perception type the model will use: {list_perception_types()}')
    parser.add_argument('--channels', '-c', help='the number of channels per cell\'s state vector', default=16)
    parser.add_argument('--hidden', '-d', help='the number of hidden channels used in the neural update', default=128)
    # training specific arguments
    parser.add_argument('--epochs', '-e', help='the number of epochs to train for', default=10_000)
    parser.add_argument('--pool_size', '-o', help='the size of the training pool', default=64)
    parser.add_argument('--batch_size', '-b', help='the number of models to take from the pool each epoch', default=4)
    parser.add_argument('--start_lr', '-slr', help='the starting learning rate', default=1e-3)
    parser.add_argument('--end_lr', '-elr', help='the ending learning rate', default=1e-5)
    parser.add_argument('--factor_sched', '-fs', help='the factor schedule used for lr-optimization', default=0.5)
    parser.add_argument('--patience_sched', '-ps', help='the patience schedule used for lr-optimization', default=500)
    parser.add_argument('--damage_num', '-dn', help='the number of (lowest loss) models in a batch to apply damage to', default=2)
    parser.add_argument('--damage_rate', '-dr', help='the rate at which to apply damage (every x epochs)', default=5)
    # logging specific arguments
    parser.add_argument('--log_file', '-l', help='the log file (created within model directory) to write to during training', default='trainlog.txt')
    parser.add_argument('--info_rate', '-i', help='the rate at which to print out training information', default=100)
    return parser.parse_args()

def assert_args(args: Namespace) -> any:
    missing_args = []
    if args.name is None: missing_args.append('--name')
    if args.target is None: missing_args.append('--target')
    if args.seed is None: missing_args.append('--seed')
    if args.model is None: missing_args.append('--model')
    if args.perception is None: missing_args.append('--perception')
    if args.channels is None: missing_args.append('--channels')
    if args.hidden is None: missing_args.append('--hidden')
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
    if len(missing_args) > 0:
        return False, missing_args
    return True, []


def list_model_types() -> list[str]:
    return ['']

def list_perception_types() -> list[str]:
    return ['']
