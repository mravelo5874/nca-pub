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
        super(_base_nca_model_, self).__init__()
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

        # * calculate hidden channel values
        perception_channels = self.perception.percieve(self.perception, torch.zeros([1, self.args.channels, 8, 8, 8]).to('cuda')).shape[1]
        hidden_channels = 8*1024 // (perception_channels+self.args.channels)
        hidden_channels = (self.args.hidden+31) // 32*32
        
        # * model layers
        self.conv1 = torch.nn.Conv3d(perception_channels, hidden_channels, 1).to('cuda')
        self.relu = torch.nn.ReLU(inplace=True).to('cuda')
        self.conv2 = torch.nn.Conv3d(hidden_channels, self.args.channels, 1, bias=False).to('cuda')
        with torch.no_grad():
            self.conv2.weight.data.zero_()

    def get_alive_mask(self, _x):
        return func.max_pool3d(_x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def save(self, _model_name):
        # save model weights
        model_path = f'models/{self.args.model_dir}/{_model_name}.pt'
        torch.save(self.state_dict(), model_path)

        # save model arguments
        import json
        import os
        args_json_path = f'models/{self.args.model_dir}/{self.args.name}_args.json'
        if not os.path.exists(args_json_path):
            json_object = json.dumps(self.args.__dict__, indent=4)
            with open(args_json_path, 'w') as outfile:
                outfile.write(json_object)

    def forward(self, _x: torch.Tensor):
        # * send to device
        x = _x.to('cuda')
        
        # * get alive mask
        alive_mask = self.get_alive_mask(x)
    
        # * perception step
        p = self.perception.percieve(self.perception, x)
        
        # * update step
        p = self.conv1(p)
        p = self.relu(p)
        p = self.conv2(p)
        
        # * create stochastic mask
        stochastic_mask = (torch.rand(_x[:, :1, :, :, :].shape) <= self.args.stochastic_rate).to('cuda')
        
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