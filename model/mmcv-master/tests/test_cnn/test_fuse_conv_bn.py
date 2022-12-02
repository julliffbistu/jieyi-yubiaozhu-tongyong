import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, fuse_conv_SyncBN


def test_fuse_conv_SyncBN():
    inputs = torch.rand((1, 3, 5, 5))
    modules = nn.ModuleList()
    modules.append(nn.BatchNorm2d(3))
    modules.append(ConvModule(3, 5, 3, norm_cfg=dict(type='SyncBN')))
    modules.append(ConvModule(5, 5, 3, norm_cfg=dict(type='SyncBN')))
    modules = nn.Sequential(*modules)
    fused_modules = fuse_conv_SyncBN(modules)
    assert torch.equal(modules(inputs), fused_modules(inputs))
