import os
import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.data import DGLDataset

def mse_loss(input, target):
    mse = 0
    for i in range(input.size(0)):
        mse += torch.sum(torch.pow(input[i, :] - target[i, :], 2))
    mse = mse / input.size(0)
    return mse

def kl_div(mu, log_var, nSize = 1):
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kld = kld / nSize
    return kld

def object_orientation_loss(input, target):
    cos_error = 0
    for i in range(input.size(0)):
        _input = input[i, :]
        _target = target[i, :]
        _input = _input/torch.norm(_input)
        _target = _target/torch.norm(_target)
        cos_error += 1 - torch.dot(_input, _target)
    cos_error = cos_error/input.size(0)
    return cos_error

def object_size_loss(in_xyz, tar_xyz):
    """[summary]

    Args:
        in_xyz ([tensor]): input length, height, width
        tar_xyz ([tensor]): output length, height, width
    """
    l = mse_loss(in_xyz, tar_xyz)
    return l

def object_location_loss(in_loc, tar_loc):
    """[summary]

    Args:
        in_loc ([tensor]): input xyz location
        tar_loc ([tensor]): output xyz location
    """
    l = mse_loss(in_loc, tar_loc)
    return l

def object_category_loss(input, target):
    """[summary]

    Args:
        in ([tensor]): input distribution
        target ([tensor]): target distribution. One hot representation
    """
    target_label = torch.argmax(target, dim = 1)
    loss_func = nn.CrossEntropyLoss()
    output = loss_func(input, target_label)
    return output

def compute_furniture_node_loss(pred_g, target_g):
    def decompose_feature(feature):
        """Decompose furniture features to each component
        Args:
            feature (tensor): feature vector of a furniture, 1040D
        output:
            category (tensor): 7D vector
            shape_featrue(tensor): 1024D vector
            placement_center: 3D vector
            orientation(tensor): 3D vector
            object_dimension(tensor): 3D vector
        """
        category = feature[:, 0:7]
        shape_feature = feature[:, 7:1024+7]
        placement_center = feature[:, 1031:1031+3]
        orientation = feature[:, 1034:1034+3]
        object_dimension = feature[:, 1037:1040]
        return category, shape_feature, placement_center, orientation, object_dimension
    
    pred_feature = pred_g.nodes['furniture'].data['feature']
    target_feature = target_g.nodes['furniture'].data['feature']
    
    pred_category, pred_shape, pred_center, pred_size, pred_orient = decompose_feature(pred_feature)
    tar_category, tar_shape, tar_center, tar_size, tar_orient = decompose_feature(target_feature)
    
    # category loss
    loss_category = object_category_loss(pred_category, tar_category)
    
    # shape feature loss
    loss_shape = mse_loss(pred_shape, tar_shape)
    
    # detection loss
    loss_orient = object_orientation_loss(pred_orient, tar_orient)
    loss_size = object_size_loss(pred_size, tar_size)
    loss_center = object_location_loss(pred_center, tar_center)
    
    loss = loss_category + loss_shape + loss_orient + loss_size + loss_center
    return loss
    
def compute_edge_loss(pred_g, target_g):
    loss_ff = mse_loss(pred_g.edges['ff'].data['feature'],
                  target_g.edges['ff'].data['feature'])
    loss_rf = mse_loss(pred_g.edges['rf'].data['feature'],
                       target_g.edges['rf'].data['feature'])
    loss_rr = mse_loss(pred_g.edges['rr'].data['feature'],
                       target_g.edges['rr'].data['feature'])
    
    loss = loss_ff + loss_rf + loss_rr
    
    return loss

def compute_kl_div(mu, log_var):
    loss = kl_div(mu, log_var, nSize = mu.size(0))
    return loss
    