from argparse import Namespace
import torch

# template nca model - build new models from this
class _base_nca_perception_():
    def __init__(
        self,
        _args: Namespace,
    ):
        self.args = _args

class thesis_anisotropic_perception(_base_nca_perception_):
    def __init__(
        self,
        _args: Namespace,
    ):
        super.__init__(_args)

class thesis_isotropic_perception(_base_nca_perception_):
    def __init__(
        self,
        _args: Namespace,
    ):
        super.__init__(_args)