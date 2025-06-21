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
from .decoder import Decoder

class FLR_GNN_Train(nn.Module):
    def __init__(self):
        super(FLR_GNN_Train, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, full_g, room_g):
        with full_g.local_scope():
            mu, log_var = self.encoder(full_g)
            pred_g = self.decoder(room_g, mu, log_var)
        
            return pred_g, mu, log_var
    
if __name__ == "__main__":
    main()