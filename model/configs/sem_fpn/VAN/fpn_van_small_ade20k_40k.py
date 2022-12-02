_base_ = './fpn_van_base_ade20k_40k.py'

# model settings
model = dict(
    pretrained='./seg_pretrained/model_best.pth.tar',
    backbone=dict(
        type='van_small',
        style='pytorch'),
    neck=dict(in_channels=[64, 128, 320, 512]))
