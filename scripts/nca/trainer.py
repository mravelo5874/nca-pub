from argparse import Namespace
import torch

class nca_trainer():
    def __init__(
        self,
        _model: torch.nn.Module,
    ):
        self.model = _model
        self.args = _model.args

    def begin(self):
        pass