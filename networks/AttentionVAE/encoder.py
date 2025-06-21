import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import dgl.function as fn
import numpy as np
import torch as  th
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import configs
from datasets import _3d_front
from .gat import AttentionLayer
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        ntypes = ['room', 'furniture']
        etypes = ['ff', 'rr', 'rf']

        r_nfeature = 14
        f_nfeature = 1040

        # layer 1
        nodes_in_dims = {'room': r_nfeature, 'furniture': f_nfeature}
        nodes_out_dims = {'room': r_nfeature, 'furniture': 256}
        edges_in_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        edges_out_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        self.gatlayer1 = AttentionLayer(ntypes, etypes, nodes_in_dims, edges_in_dims, nodes_out_dims, edges_out_dims)
        
        # layer 2
        nodes_in_dims = {'room': r_nfeature, 'furniture': 256}
        nodes_out_dims = {'room': r_nfeature, 'furniture': 128}
        edges_in_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        edges_out_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        self.gatlayer2 = AttentionLayer(ntypes, etypes, nodes_in_dims, edges_in_dims, nodes_out_dims, edges_out_dims)
        
        # layer 3
        nodes_in_dims = {'room': r_nfeature, 'furniture': 128}
        nodes_out_dims = {'room': r_nfeature, 'furniture': 64}
        edges_in_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        edges_out_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        self.gatlayer3 = AttentionLayer(ntypes, etypes, nodes_in_dims, edges_in_dims, nodes_out_dims, edges_out_dims)
        
        # gaussian mean and std weights
        self.wMean = nn.Linear(nodes_out_dims['furniture'], nodes_out_dims['furniture'])
        self.wLogVar = nn.Linear(nodes_out_dims['furniture'], nodes_out_dims['furniture'])
        
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.tanh_scale = torch.tensor(1)
        self.reset_params()
        
    def reset_params(self):
        nn.init.xavier_normal_(self.wMean.weight)
        nn.init.xavier_normal_(self.wLogVar.weight)
        
    def dump_model_info(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Total number of parameters: {}'.format(params))

    def readout(self, g):
        # use tanh to limit the value range
        mu = self.tanh_scale * self.tanh(self.wMean(g.nodes['furniture'].data['feature']))
        log_var = self.tanh_scale * self.tanh(self.wLogVar(g.nodes['furniture'].data['feature']))
        return mu, log_var

    def forward(self, g):
        g1 = self.gatlayer1(g)
        g2 = self.gatlayer2(g1)
        g3 = self.gatlayer3(g2)
        
        mu, log_var = self.readout(g3)
        
        return mu, log_var

def main():
    configs.update(configs.from_yaml(filename='configs/3d_front.yaml'))
    _3dfront = _3d_front._3DFrontDataset(configs.io.dir_scene_graph)
    full_gs, room_gs = _3dfront[0]
    
    # nodes_in_dims = {'room': 14, 'furniture': 1040}
    # nodes_out_dims = {'room': 14, 'furniture': 256}
    # edges_in_dims = {'ff': 3, 'rr': 4, 'rf': 5}
    # edges_out_dims = {'ff': 3, 'rr': 4, 'rf': 5}
    # ntypes = ['room', 'furniture']
    # etypes = ['ff', 'rr', 'rf']
    # model = AttentionLayer(ntypes, etypes, nodes_in_dims, edges_in_dims, nodes_out_dims, edges_out_dims)
    # model = model.float()
    # # check the number of params
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # for name, parms in model.named_parameters():
    #     print(name)

    encoder = Encoder()
    mu, sigma = encoder(full_gs)

if __name__ == "__main__":
    main()