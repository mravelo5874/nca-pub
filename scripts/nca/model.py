from argparse import Namespace
import torch

# template nca model - build new models from this
class _base_nca_model_(torch.nn.Module):
    def __init__(
        self,
        _args: Namespace,
    ):
        self.args = _args

    def save(self):
        raise NotImplementedError
        # this method must be implemented by sub-classes!

    def forward(self):
        raise NotImplementedError
        # this method must be implemented by sub-classes!

class thesis_nca_model(_base_nca_model_):
    def __init__(
        self,
        _args: Namespace,
    ):
        super.__init__(_args)

    def save(self):
        pass

    def forward(self):
        pass