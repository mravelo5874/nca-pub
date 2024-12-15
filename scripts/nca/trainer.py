import torch

# template nca trainer - build new trainers from this
class _base_nca_trainer_():
    def __init__(
        self,
        _model: torch.nn.Module,
    ):
        self.model = _model
        self.args = _model.args

    def begin(self):
        raise NotImplementedError
        # this method must be implemented by sub-classes!

class thesis_nca_trainer(_base_nca_trainer_):
    def __init__(
        self,
        _model: torch.nn.Module,
    ):
        super.__init__(_model)

    def begin(self):
        pass