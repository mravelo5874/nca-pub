from argparse import Namespace
import torch

class nca_model(torch.nn.Module):
    def __init__(
        self,
        _args: Namespace,
    ):
        self.args = _args