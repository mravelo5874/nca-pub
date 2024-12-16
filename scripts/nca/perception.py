import scripts.nca.kernels as kernels
import torch.nn.functional as func
import scripts.nca.utils as utils
from argparse import Namespace
from types import ModuleType
import torch

# template nca model - build new models from this
class _base_nca_perception_():
    def __init__(
        self,
        _args: Namespace,
    ):
        self.args = _args

    def percieve(self, _x: torch.Tensor):
        raise NotImplementedError
        # this method must be implemented by sub-classes!

def orientation_channels(self, _module: ModuleType):
    if _module == thesis_anisotropic_perception:
        return 0
    if _module == thesis_1axis_isotropic_perception:
        return 1
    if _module == thesis_isotropic_perception:
        return 3
    
# performs a convolution per filter per channel
def per_channel_conv3d(self, _x, _filters):
    batch_size, channels, height, width, depth = _x.shape
    # * reshape x to make per-channel convolution possible + pad 1 on each side
    y = _x.reshape(batch_size*channels, 1, height, width, depth).to('cuda')
    y = func.pad(y, (1, 1, 1, 1, 1, 1), 'constant')
    # * perform per-channel convolutions
    _filters = _filters.to('cuda')
    y = func.conv3d(y, _filters[:, None])
    y = y.reshape(batch_size, -1, height, width, depth)
    return y

class thesis_anisotropic_perception(_base_nca_perception_):
    def __init__(
        self,
        _args: Namespace,
    ):
        super(self, thesis_anisotropic_perception).__init__(_args)

    def percieve(self, _x: torch.Tensor):
        _x = _x.to('cuda')
        # per channel convolutions
        gx = self.per_channel_conv3d(_x, kernels.X_SOBEL[None, :])
        gy = self.per_channel_conv3d(_x, kernels.Y_SOBEL[None, :])
        gz = self.per_channel_conv3d(_x, kernels.Z_SOBEL_DOWN[None, :])
        lap = self.per_channel_conv3d(_x, kernels.LAP_KERN_27[None, :])
        return torch.cat([_x, gx, gy, gz, lap], 1)

class thesis_1axis_isotropic_perception(_base_nca_perception_):
    def __init__(
        self,
        _args: Namespace,
    ):
        super(self, thesis_isotropic_perception).__init__(_args)

    def percieve(self, _x: torch.Tensor):
        # separate states and angle channels
        states, angle = _x[:, :-1], _x[:, -1:]
        
        # * calculate gx and gy
        gx = self.per_channel_conv3d(states, kernels.X_SOBEL_2D_XY[None, :])
        gy = self.per_channel_conv3d(states, kernels.Y_SOBEL_2D_XY[None, :])
        
        # calculate lap2d and lap3d
        lap2d = self.per_channel_conv3d(states, kernels.LAP_2D_XY[None, :])
        lap3d = self.per_channel_conv3d(states, kernels.LAP_KERN_7[None, :])
           
        # compute px and py 
        _cos, _sin = angle.cos(), angle.sin()
        px = (gx*_cos)+(gy*_sin)
        py = (gy*_cos)-(gx*_sin)
        return torch.cat([states, lap2d, px, py, lap3d], 1)

class thesis_isotropic_perception(_base_nca_perception_):
    def __init__(
        self,
        _args: Namespace,
    ):
        super(self, thesis_isotropic_perception).__init__(_args)

    def percieve(self, _x):
        # * separate states and angle channels
        states, ax, ay, az = _x[:, :-3], _x[:, -1:], _x[:, -2:-1], _x[:, -3:-2]

        # * per channel convolutions
        px = self.per_channel_conv3d(states, kernels.X_SOBEL[None, :])
        py = self.per_channel_conv3d(states, kernels.Y_SOBEL[None, :])
        pz = self.per_channel_conv3d(states, kernels.Z_SOBEL_DOWN[None, :])
        lap = self.per_channel_conv3d(states, kernels.LAP_KERN_27[None, :])
        
        # * get perception tensors
        px = px[..., None]
        py = py[..., None]
        pz = pz[..., None]
        pxyz = torch.cat([px, py, pz], 5)
        bs, hc, sx, sy, sz, p3 = pxyz.shape
        pxyz = pxyz.reshape([bs, hc, sx*sy*sz, p3])
        pxyz = pxyz.unsqueeze(-1)
        
        # * get quat values
        bs, _, sx, sy, sz = ax.shape
        ax = ax.reshape([bs, sx*sy*sz])
        ay = ay.reshape([bs, sx*sy*sz])
        az = az.reshape([bs, sx*sy*sz])
        R_mats = utils.eul2rot(ax, ay, az)
        
        # * rotate perception tensors
        rxyz = torch.zeros_like(pxyz)
        for i in range(hc):
            rxyz[:, i] = torch.matmul(R_mats, pxyz[:, i])
        rxyz = rxyz.reshape([bs, hc, sx, sy, sz, p3])
        
        # * extract rotated perception tensors
        rx = rxyz[:, :, :, :, :, 0]
        ry = rxyz[:, :, :, :, :, 1]
        rz = rxyz[:, :, :, :, :, 2]
        return torch.cat([states, rx, ry, rz, lap], 1)