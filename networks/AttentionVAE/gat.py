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

class AttentionLayer(nn.Module):
    def __init__(self, ntypes, etypes, nodes_in_dims, edges_in_dims, nodes_out_dims, edges_out_dims):
        """[summary]
        Args:
            g (Hetero grpah): Input hetero graph
            nodes_in_dims (dict with keys: 'room' and 'furniture'): A list of input dimensions for nodes
            edges_in_dims (dict with keys: 'ff', 'rr', and 'rf'): A list of input dimensions for edges
            nodes_out_dims (dict with keys: 'room' and 'furniture'): A list of output dimensions for nodes
            edges_out_dims (dict with keys: 'ff', 'rr', and 'rf'): A list of output dimensions for edges
        """
        super(AttentionLayer, self).__init__()
        self.ntypes = ntypes
        self.etypes = etypes
        
        # Initialize node weights
        self.W_rnode = nn.Linear(nodes_in_dims['room'], nodes_out_dims['room'], bias = False)
        self.W_fnode = nn.Linear(nodes_in_dims['furniture'], nodes_out_dims['furniture'], bias = False)

        # Initialize edge weights
        self.W_ffedge = nn.Linear(edges_in_dims['ff'], edges_out_dims['ff'], bias = False)
        self.W_rredge = nn.Linear(edges_in_dims['rr'], edges_out_dims['rr'], bias = False)
        self.W_rfedge = nn.Linear(edges_in_dims['rf'], edges_out_dims['rf'], bias = False)
    
        # Initialize edge attention vectors
        self.W_att_ffedge = nn.Linear(nodes_out_dims['furniture'] + \
                                      nodes_out_dims['furniture'] + \
                                      edges_out_dims['ff'], 1, bias = False)
        self.W_att_rredge = nn.Linear(nodes_out_dims['room'] + \
                                      nodes_out_dims['room'] + \
                                      edges_out_dims['rr'], 1, bias = False)
        self.W_att_rfedge = nn.Linear(nodes_out_dims['room'] + \
                                      nodes_out_dims['furniture'] + \
                                      edges_out_dims['rf'], 1, bias = False)
        
        # Weights that convert node feature dimension when computing edge updates
        self.W_ffedge_convert = nn.Linear(nodes_out_dims['furniture'], edges_out_dims['ff'], bias = False)
        self.W_rredge_convert = nn.Linear(nodes_out_dims['room'], edges_out_dims['rr'], bias = False)
        self.W_rfedge_convert = nn.Linear(nodes_out_dims['furniture'], edges_out_dims['rf'], bias = False)
        
        # Weights that convert edge feature dimension when computing edge updates
        self.W_ffnode_convert = nn.Linear(edges_out_dims['ff'], nodes_out_dims['furniture'], bias = False)
        self.W_rrnode_convert = nn.Linear(edges_out_dims['rr'], nodes_out_dims['room'], bias = False)
        self.W_rfnode_convert = nn.Linear(edges_out_dims['rf'], nodes_out_dims['furniture'], bias = False)
        
        self.leaky_relu = nn.LeakyReLU()
    
        self.reset_params()
    
    def reset_params(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W_rnode.weight, gain = gain)
        nn.init.xavier_normal_(self.W_fnode.weight, gain = gain)
        nn.init.xavier_normal_(self.W_ffedge.weight, gain = gain)
        nn.init.xavier_normal_(self.W_rredge.weight, gain = gain)
        nn.init.xavier_normal_(self.W_rfedge.weight, gain = gain)
        nn.init.xavier_normal_(self.W_att_ffedge.weight, gain = gain)
        nn.init.xavier_normal_(self.W_att_rredge.weight, gain = gain)
        nn.init.xavier_normal_(self.W_att_rfedge.weight, gain = gain)
        nn.init.xavier_normal_(self.W_ffedge_convert.weight, gain = gain)
        nn.init.xavier_normal_(self.W_rredge_convert.weight, gain = gain)
        nn.init.xavier_normal_(self.W_rfedge_convert.weight, gain = gain)
        nn.init.xavier_normal_(self.W_ffnode_convert.weight, gain = gain)
        nn.init.xavier_normal_(self.W_rrnode_convert.weight, gain = gain)
        nn.init.xavier_normal_(self.W_rfnode_convert.weight, gain = gain)
        
        
    def edge_attention(self, edges):
        canonical_etype = edges.canonical_etype        
        if canonical_etype[1] == 'ff':
            msg_src = self.W_fnode(edges.src['feature'])
            msg_dst = self.W_fnode(edges.dst['feature'])
            msg_edge = self.W_ffedge(edges.data['feature']) 
            msg = torch.cat([msg_src, msg_dst, msg_edge], dim = 1)
            score = self.leaky_relu(self.W_att_ffedge(msg))
        if canonical_etype[1] == 'rr':
            msg_edge = self.W_rredge(edges.data['feature'])
            msg_src = self.W_rnode(edges.src['feature'])
            msg_dst = self.W_rnode(edges.dst['feature']) 
            msg = torch.cat([msg_src, msg_dst, msg_edge], dim = 1)
            score = self.leaky_relu(self.W_att_rredge(msg))
        if canonical_etype[1] == 'rf':
            msg_edge = self.W_rfedge(edges.data['feature']) 
            msg_src = self.W_rnode(edges.src['feature'])
            msg_dst = self.W_fnode(edges.dst['feature'])
            msg = torch.cat([msg_src, msg_dst, msg_edge], dim = 1)
            score = self.leaky_relu(self.W_att_rfedge(msg))

        return {'score': score}
        
    def edge_normalized_attention(self, edges):
        score = edges.data['score']/edges.src['sum_scores']
        return {'score': score}
        
    def update_edge_feature(self, edges):
        canonical_etype = edges.canonical_etype
        score = edges.data['score']
        if canonical_etype[1] == 'ff':
            feature = score * (self.W_ffedge_convert(self.W_fnode(edges.dst['feature'])) + \
                               self.W_ffedge(edges.data['feature']))
        if canonical_etype[1] == 'rr':
            feature = score * (self.W_rredge_convert(self.W_rnode(edges.dst['feature'])) + \
                               self.W_rredge(edges.data['feature']))
        if canonical_etype[1] == 'rf':
            feature = score * (self.W_rfedge_convert(self.W_fnode(edges.dst['feature'])) + \
                               self.W_rfedge(edges.data['feature']))
        return {'feature': feature}
        
    def node_test(self, nodes):
        # nodes.data['feature'] = nodes.data['feature'] * 2
        
        zx = nodes.data['sum_scores']
        print(nodes.data['sum_scores'])
        
        return {'feature': nodes.data['feature']}
        
    def message_func_cumulate_scores(self, edges):
        return {'score': edges.data['score']} # put results to mailbox['score']
    
    def reduce_func_compute_total_scores(self, nodes):      
        if 'sum_scores' not in nodes.data:
            sum_scores = torch.sum(nodes.mailbox['score'], dim = 1)
        else:
            sum_scores = nodes.data['sum_scores'] + torch.sum(nodes.mailbox['score'], dim = 1)
        return {'sum_scores': sum_scores}
                
    def message_func_cumulate_node_updates(self, edges):
        canonical_etype = edges.canonical_etype
        
        if canonical_etype[1] == 'ff':
            update = self.W_ffnode_convert(edges.data['feature'])
            return {'update': update}
        if canonical_etype[1] == 'rr':
            update = self.W_rrnode_convert(edges.data['feature'])
            return {'update': update}
        if canonical_etype[1] == 'rf':
            update = self.W_rfnode_convert(edges.data['feature'])
            return {'update': update}
        
    def reduce_func_compute_total_updates(self, nodes):
        if 'sum_updates' not in nodes.data:
            sum_updates = torch.sum(nodes.mailbox['update'], dim = 1)
        else:
            sum_updates = nodes.data['sum_updates'] + torch.sum(nodes.mailbox['update'], dim = 1)
        return {'sum_updates': sum_updates}
        
    def update_node_feature(self, nodes):
        ntype = nodes.ntype
        
        if ntype=='furniture':
            new_feature = self.W_fnode(nodes.data['feature']) + self.leaky_relu(nodes.data['sum_updates'])
        if ntype=='room':
            new_feature = self.W_rnode(nodes.data['feature']) + self.leaky_relu(nodes.data['sum_updates'])

        return {'feature': new_feature}
        
    def dummy_reduce_func(self, nodes):
        nodes.data['feature'] = nodes.data['feature'] * 9999999
        return {'new_feature': nodes.data['feature'], 'dummy': nodes.data['feature']}
    
    def forward(self, g):
        # Compute attention to each edge
        g['ff'].apply_edges(self.edge_attention)
        g['rr'].apply_edges(self.edge_attention)
        g['rf'].apply_edges(self.edge_attention)
        
        # Compute the sum of attention scores for each node
        # Each type of edge will send its score to its dst node.
        # On the node side, the node will sum up scores from all
        # types of edges and save to data['sum_scores']
        g['ff'].update_all(self.message_func_cumulate_scores, self.reduce_func_compute_total_scores)
        g['rf'].update_all(self.message_func_cumulate_scores, self.reduce_func_compute_total_scores)
        g['rr'].update_all(self.message_func_cumulate_scores, self.reduce_func_compute_total_scores)
        
        # Compute normalized attention
        # The dst node of each edge saved the sum of scores from
        # all edges pointing to the node.
        # Normalization is done be dividing score of each edge by that sum.
        g['ff'].apply_edges(self.edge_normalized_attention)
        g['rf'].apply_edges(self.edge_normalized_attention)
        g['rr'].apply_edges(self.edge_normalized_attention)
                
        # Update edge feature
        g['ff'].apply_edges(self.update_edge_feature)
        g['rr'].apply_edges(self.update_edge_feature)
        g['rf'].apply_edges(self.update_edge_feature)
        
        # Update nodes feature
        g['ff'].update_all(self.message_func_cumulate_node_updates, self.reduce_func_compute_total_updates)
        g['rf'].update_all(self.message_func_cumulate_node_updates, self.reduce_func_compute_total_updates)
        g['rr'].update_all(self.message_func_cumulate_node_updates, self.reduce_func_compute_total_updates)
        g.apply_nodes(self.update_node_feature, ntype='furniture')
        g.apply_nodes(self.update_node_feature, ntype='room')
        
        # remove temporary variables        
        g.nodes['furniture'].data.pop('sum_scores', None)
        g.nodes['room'].data.pop('sum_scores', None)
        g.nodes['furniture'].data.pop('sum_updates', None)
        g.nodes['room'].data.pop('sum_updates', None)
        
        # g.apply_nodes(self.node_test, ntype='furniture')
        
        return g