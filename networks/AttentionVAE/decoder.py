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
from .encoder import Encoder

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()    
        ntypes = ['room', 'furniture']
        etypes = ['ff', 'rr', 'rf']

        r_nfeature = 14
        f_nfeature = 1040

        # layer 1
        nodes_in_dims = {'room': r_nfeature, 'furniture': 64}
        nodes_out_dims = {'room': r_nfeature, 'furniture': 128}
        edges_in_dims = {'ff': 128, 'rr': 4, 'rf': 64 + r_nfeature}
        edges_out_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        self.gatlayer1 = AttentionLayer(ntypes, etypes, nodes_in_dims, edges_in_dims, nodes_out_dims, edges_out_dims)
        
        # layer 2
        nodes_in_dims = {'room': r_nfeature, 'furniture': 128}
        nodes_out_dims = {'room': r_nfeature, 'furniture': 256}
        edges_in_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        edges_out_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        self.gatlayer2 = AttentionLayer(ntypes, etypes, nodes_in_dims, edges_in_dims, nodes_out_dims, edges_out_dims)
        
        # layer 3
        nodes_in_dims = {'room': r_nfeature, 'furniture': 256}
        nodes_out_dims = {'room': r_nfeature, 'furniture': f_nfeature}
        edges_in_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        edges_out_dims = {'ff': 3, 'rr': 4, 'rf': 5}
        self.gatlayer3 = AttentionLayer(ntypes, etypes, nodes_in_dims, edges_in_dims, nodes_out_dims, edges_out_dims)
        
    def reparameterization(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        # std_z = torch.from_numpy(np.random.normal(0, 1, size = sigma.size())).float()
        std_z = torch.randn(sigma.size())
        return mu + sigma * Variable(std_z, requires_grad=False).to('cuda')    
    
    def dump_model_info(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Total number of parameters: {}'.format(params))

    def initialize_full_graph(self, g, mu, log_var):
        furniture_features = self.reparameterization(mu, log_var)
        g.nodes['furniture'].data['feature']=furniture_features
        
        # ff
        srcs, dsts = g.edges(etype='ff')
        edge_features = None
        for src, dst in zip(srcs, dsts):
            feature = torch.cat((furniture_features[src:src+1, :], furniture_features[dst:dst+1, :]), dim = 1)
            if edge_features is None:
                edge_features = feature
            else:
                edge_features = torch.cat((edge_features, feature), dim = 0)
        g.edges['ff'].data['feature'] = edge_features
        
        # rf
        srcs, dsts = g.edges(etype='rf')
        room_features = g.nodes['room'].data['feature']
        edge_features = None
        for src, dst in zip(srcs, dsts):
            feature = torch.cat((room_features[src:src+1, :], furniture_features[dst:dst+1, :]), dim = 1)
            if edge_features is None:
                edge_features = feature
            else:
                edge_features = torch.cat((edge_features, feature), dim = 0)
        g.edges['rf'].data['feature'] = edge_features
        
        return g
        
    def forward(self, g, mu, log_var):
        g = self.initialize_full_graph(g, mu, log_var) 
        g1 = self.gatlayer1(g)
        g2 = self.gatlayer2(g1)
        g3 = self.gatlayer3(g2)
        
        return g3    

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
    mu, log_var = encoder(full_gs)

    decoder = Decoder()
    decoder(room_gs, mu, log_var)

if __name__ == "__main__":
    main()