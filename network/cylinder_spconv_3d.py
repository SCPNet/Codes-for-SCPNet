# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

from torch import nn
import torch

REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class cylinder_asym(nn.Module):
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size, val_grid=None, voting_num=4, use_tta=False):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        # train_pt_fea_ten: [batch_size, N1, 7]
        # train_vox_ten: [batch_size, N1, 3]

        if use_tta:
            batch_size *= voting_num

        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)  # [batch_size, 20, 256, 256, 32]
        
        if use_tta:
            fused_predict = spatial_features[0, :]
            for idx in range(1, voting_num, 1):
                aug_predict = spatial_features[idx, :]
                aug_predict = torch.flip(aug_predict, dims=[2])
                fused_predict += aug_predict
            return torch.unsqueeze(fused_predict, 0)
        else: 
            return spatial_features
