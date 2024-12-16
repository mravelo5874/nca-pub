from scripts.nca.perception import _base_nca_perception_, orientation_channels
import torch.nn.functional as func
from argparse import Namespace
from numpy import pi
import torch

# template nca model - build new models from this
class _base_nca_model_(torch.nn.Module):
    def __init__(
        self,
        _args: Namespace,
        _perception: _base_nca_perception_
    ):
        self.args = _args
        self.perception = _perception

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
        _perception: _base_nca_perception_
    ):
        super(thesis_nca_model, self).__init__(_args, _perception)

    def get_alive_mask(self, _x):
        return func.max_pool3d(_x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def save(self, _model_name):
        # save model weights
        import pathlib
        model_path = pathlib.Path(f'{self.args.model_dir}/{self.args.name}/{_model_name}.pt')
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), model_path)

        # save model arguments
        import json
        json_object = json.dumps(self.args.__dict__, indent=4)
        args_json_path = pathlib.Path(f'{self.args.model_dir}/{self.args.name}/{_model_name}_args.json')
        with open(args_json_path, 'w') as outfile:
            outfile.write(json_object)

    def forward(self, _x: torch.Tensor):
        # * send to device
        x = _x
        
        # * get alive mask
        alive_mask = self.get_alive_mask(x)
    
        # * perception step
        p = self.perception.percieve(x)
        
        # * update step
        p = self.conv1(p)
        p = self.relu(p)
        p = self.conv2(p)
        
        # * create stochastic mask
        stochastic_mask = (torch.rand(_x[:, :1, :, :, :].shape) <= self.rate)
        
        # * perform stochastic update
        x = x + p * stochastic_mask
        
        # * final isotropic concatination + apply alive mask
        ori_channels = orientation_channels(self.perception)
        if ori_channels == 1:
            states = x[:, :-1]*alive_mask
            angle = x[:, -1:] % (pi*2.0)
            x = torch.cat([states, angle], 1)
            
        elif ori_channels == 3:
            states = x[:, :-3]*alive_mask
            ax = x[:, -1:] % (pi*2.0)
            ay = x[:, -2:-1] % (pi*2.0)
            az = x[:, -3:-2] % (pi*2.0)
            x = torch.cat([states, az, ay, ax], 1)
            
        else:
            x = x * alive_mask
           
        return x