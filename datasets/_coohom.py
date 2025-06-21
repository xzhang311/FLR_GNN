import dgl
from dgl.data import DGLDataset
import torch
import os
import pickle
import numpy as np
import sys, pdb
sys.path.insert(1, os.path.join(sys.path[0], '..'))
print(sys.path)
from utils import recursive_glob_full_path
from config import configs
import json
import open3d as o3d
import itertools

class _CoohamDataset(DGLDataset):
    def __init__(self, dir_root, mode='Normal'):
        self.dir_root = dir_root
        self.mode = mode
        self.category_to_id = self.get_category_to_id_mapping()
        super().__init__(name = 'Cooham')
    
    def id_to_idx(self, nodes):
        id_idx_dict = {}
        idx_id_dict = {}
        count = 0
        for id in nodes:
            id_idx_dict[id] = count
            idx_id_dict[count] = id
            count = count + 1
            
        return id_idx_dict, idx_id_dict

    def get_category_to_id_mapping(self):
        f = open("/mnt/ebs_xizhn/Projects/FurnitureLayout/PreprocessCoohom/miscs/pruned_cat_id.json")
        data = f.read()
        obj = json.loads(data)
        new_dict = {}
        for index, key in enumerate(obj.keys()):
            new_dict[int(key)] = index
        f.close()
        self.category_names = obj
        return new_dict

    
    def get_edges_src_dst_ids(self, edges, room_id_idx_dict, furniture_id_idx_dict):
        srcs = []
        dsts = []
        for src in edges:
            for dst in edges[src]:
                # if src == dst:
                #     continue

                if src in room_id_idx_dict:
                    srcs.append(int(room_id_idx_dict[src]))
                elif src in furniture_id_idx_dict:
                    srcs.append(int(furniture_id_idx_dict[src]))
                elif src not in furniture_id_idx_dict:
                    continue

                if dst in room_id_idx_dict:
                    dsts.append(int(room_id_idx_dict[dst]))
                elif dst in furniture_id_idx_dict:
                    dsts.append(int(furniture_id_idx_dict[dst]))

                elif dst not in furniture_id_idx_dict:
                    srcs.pop()
                    continue
                    
        return srcs, dsts
    
    def print_grpah_info(self, g):
        print('Node types: {}'.format(g.ntypes))
        print('Edge types: {}'.format(g.etypes))
        for ntype in g.ntypes:
            print('Num of node type {} is {}'.format(ntype, g.num_nodes(ntype)))      
        for etype in g.etypes:
            print('Num of edge type {} is {}'.format(etype, g.num_edges(etype)))
            print('Edge nodes: {}'.format(g.edges(etype=etype)))
    
    def assign_node_features(self, g, ntype, scene):
        node_type_dict = {'floor': 0, 'win': 1, 'door': 2, 'wall': 3}
        node_type = 'room_nodes' if ntype == 'room' else 'furniture_nodes'
        
        num_nodes = g.number_of_nodes(ntype)
        
        if ntype == 'room':
            rnode_features = {}
            rnode_features['room_type'] = []
            rnode_features['bbox'] = []
            rnode_features['normal'] = []
            rnode_features['area'] = []
            rnode_features['node_type'] = []
            for node_idx in scene['nodes'][node_type]:
                node = scene['nodes'][node_type][node_idx]
                rnode_features['bbox'].append(node['bbox'])
                rnode_features['normal'].append(node['normal'])
                rnode_features['area'].append(node['area'])
                rnode_features['room_type'].append(scene['room_type'])
                rnode_features['node_type'].append(node_type_dict[node['type']])
            g.nodes[ntype].data['bbox'] = torch.tensor(np.asarray(rnode_features['bbox']).astype(np.float32))
            g.nodes[ntype].data['normal'] = torch.tensor(np.asarray(rnode_features['normal']).astype(np.float32))
            g.nodes[ntype].data['area'] = torch.tensor(np.asarray(rnode_features['area']).astype(np.float32))
            g.nodes[ntype].data['room_type'] = torch.tensor(np.asarray(rnode_features['room_type']).astype(np.float32))
            g.nodes[ntype].data['node_type'] = torch.tensor(np.asarray(rnode_features['node_type']).astype(np.float32))
            
            feature = np.concatenate([np.asarray(rnode_features['room_type']).reshape(num_nodes, -1),
                                      np.asarray(rnode_features['bbox']).reshape(num_nodes, -1), 
                                      np.asarray(rnode_features['normal']).reshape(num_nodes, -1),
                                      np.asarray(rnode_features['area']).reshape(num_nodes, -1)
                                    ], axis = 1)
            feature = np.nan_to_num(feature, nan = 0.0, posinf = 9999.0, neginf = -9999.0)        
        if ntype == 'furniture':
            up = np.asarray((0, 1, 0))
            fnode_features = {}
            fnode_features['super_category'] = [] # super-category, 7 classes
            fnode_features['shape_feature'] = []
            fnode_features['bbox_rotated_translated'] = []
            fnode_features['placement_center'] = []
            fnode_features['placement_orientation'] = []
            fnode_features['object_dimension'] = [] # Array representing dimension in order of x, y, z
            for node_idx in scene['nodes'][node_type]:
                node = scene['nodes'][node_type][node_idx]
                fnode_features['super_category'].append(self.to_onehot(self.category_to_id[node['model_info_index']['super-category']], len(self.category_names)))
                fnode_features['shape_feature'].append(node['shape_feature'])
                fnode_features['bbox_rotated_translated'].append(node['bbox_rotated_translated'])
                fnode_features['placement_center'].append(node['placement_center'])
                fnode_features['placement_orientation'].append(node['placement_orientation'].flatten())

                ori_2 = np.cross(node['placement_orientation'].flatten(), up)
                if np.linalg.norm(ori_2) == 0.0:
                    return None

                fnode_features['object_dimension'].append(node['object_dimension'])   
            g.nodes[ntype].data['super_category'] = torch.tensor(np.asarray(fnode_features['super_category']).astype(np.float32))
            g.nodes[ntype].data['shape_feature'] = torch.tensor(np.asarray(fnode_features['shape_feature']).astype(np.float32))
            g.nodes[ntype].data['bbox_rotated_translated'] = torch.tensor(np.asarray(fnode_features['bbox_rotated_translated']).astype(np.float32))
            g.nodes[ntype].data['placement_center'] = torch.tensor(np.asarray(fnode_features['placement_center']).astype(np.float32))            
            g.nodes[ntype].data['placement_orientation'] = torch.tensor(np.asarray(fnode_features['placement_orientation']).astype(np.float32))
            g.nodes[ntype].data['object_dimension'] = torch.tensor(np.asarray(fnode_features['object_dimension']).astype(np.float32)) 

            feature = np.concatenate([np.asarray(fnode_features['shape_feature']).reshape(num_nodes, -1),
                                      np.asarray(fnode_features['placement_center']).reshape(num_nodes, -1),
                                      np.asarray(fnode_features['placement_orientation']).reshape(num_nodes, -1),
                                      np.asarray(fnode_features['object_dimension']).reshape(num_nodes, -1)], axis = 1)
            feature = np.nan_to_num(feature, nan = 0.0, posinf = 9999.0, neginf = -9999.0)

        g.nodes[ntype].data['feature'] = torch.tensor(feature.astype(np.float32))
        
        return g
    
    def assign_edge_features(self, g, etype, scene, room_idx_id_dict, furniture_idx_id_dict):
        edge_type = 'furniture_to_furniture' if etype == 'ff' else None
        edge_type = 'room_to_furniture' if etype == 'rf' else edge_type
        edge_type = 'room_to_room' if etype == 'rr' else edge_type
        
        num_edges = g.number_of_edges(etype)
        
        if etype=='ff':
            center_to_center_dist = []
            bbox_to_bbox_dist = []
            orientation =[]
            srcs = g.edges(etype='ff')[0]
            dsts = g.edges(etype='ff')[1]
            
            for src, dst in zip(srcs, dsts):
                src_id = furniture_idx_id_dict[int(src)]
                dst_id = furniture_idx_id_dict[int(dst)]
                edge = scene['edges']['furniture_to_furniture'][src_id][dst_id]
                center_to_center_dist.append(edge['center_to_center_dist'])
                bbox_to_bbox_dist.append(edge['bbox_to_bbox_dist'])
                orientation.append(edge['orientation'])
            g.edges['ff'].data['center_to_center_dist'] = torch.tensor(np.asarray(center_to_center_dist).astype(np.float32))
            g.edges['ff'].data['bbox_to_bbox_dist'] = torch.tensor(np.asarray(bbox_to_bbox_dist).astype(np.float32))
            g.edges['ff'].data['orientation'] = torch.tensor(np.asarray(orientation).astype(np.float32))
            
            feature = np.concatenate([np.asarray(center_to_center_dist).reshape(num_edges, -1),
                                      np.asarray(bbox_to_bbox_dist).reshape(num_edges, -1),
                                      np.asarray(orientation).reshape(num_edges, -1)], axis = 1)
            feature = np.nan_to_num(feature, nan = 0.0, posinf = 9999.0, neginf = -9999.0)
            g.edges['ff'].data['feature'] = torch.tensor(feature.astype(np.float32))
            
        if etype=='rr':
            center_to_center_dist = []
            longest_dist = []
            shortest_dist = []
            orientation =[]
            srcs = g.edges(etype='rr')[0]
            dsts = g.edges(etype='rr')[1]
            
            for src, dst in zip(srcs, dsts):
                src_id = room_idx_id_dict[int(src)]
                dst_id = room_idx_id_dict[int(dst)]
                edge =  scene['edges']['room_to_room'][src_id][dst_id]
                center_to_center_dist.append(edge['center_to_center'])
                orientation.append(edge['orientation'])
                longest_dist.append(edge['longest_dist'])
                shortest_dist.append(edge['shortest_dist'])
            g.edges['rr'].data['center_to_center_dist'] = torch.tensor(np.asarray(center_to_center_dist).astype(np.float32))
            g.edges['rr'].data['orientation'] = torch.tensor(np.asarray(orientation).astype(np.float32))
            g.edges['rr'].data['longest_dist'] = torch.tensor(np.asarray(longest_dist).astype(np.float32))
            g.edges['rr'].data['shortest_dist'] = torch.tensor(np.asarray(shortest_dist).astype(np.float32))
            
            feature = np.concatenate([np.asarray(center_to_center_dist).reshape(num_edges, -1), 
                                      np.asarray(orientation).reshape(num_edges, -1),
                                      np.asarray(longest_dist).reshape(num_edges, -1),
                                      np.asarray(shortest_dist).reshape(num_edges, -1)], axis = 1)
            feature = np.nan_to_num(feature, nan = 0.0, posinf = 9999.0, neginf = -9999.0)
            g.edges['rr'].data['feature'] = torch.tensor(feature.astype(np.float32))
            
        if etype=='rf':
            center_to_wall_dist = []
            center_to_wall_center = []
            bbox_to_wall_center = []
            bbox_to_wall_dist = []
            orientation = []
            srcs = g.edges(etype='rf')[0]
            dsts = g.edges(etype='rf')[1]
            
            for src, dst in zip(srcs, dsts):
                src_id = room_idx_id_dict[int(src)]
                dst_id = furniture_idx_id_dict[int(dst)]
                edge =  scene['edges']['room_to_furniture'][src_id][dst_id]
                center_to_wall_dist.append(edge['center_to_wall_dist'])
                center_to_wall_center.append(edge['center_to_wall_center'])
                bbox_to_wall_center.append(edge['bbox_to_wall_center'])
                bbox_to_wall_dist.append(edge['bbox_to_wall_dist'])
                orientation.append(edge['orientation'])
            g.edges['rf'].data['center_to_wall_dist'] = torch.tensor(np.asarray(center_to_wall_dist).flatten().astype(np.float32))
            g.edges['rf'].data['center_to_wall_center'] = torch.tensor(np.asarray(center_to_wall_center).flatten().astype(np.float32))
            g.edges['rf'].data['bbox_to_wall_center'] = torch.tensor(np.asarray(bbox_to_wall_center).flatten().astype(np.float32))
            g.edges['rf'].data['bbox_to_wall_dist'] = torch.tensor(np.asarray(bbox_to_wall_dist).flatten().astype(np.float32))
            g.edges['rf'].data['orientation'] = torch.tensor(np.asarray(orientation).astype(np.float32))
        
            feature = np.concatenate([np.asarray(center_to_wall_dist).reshape(num_edges, -1),
                                      np.asarray(center_to_wall_center).reshape(num_edges, -1),
                                      np.asarray(bbox_to_wall_center).reshape(num_edges, -1),
                                      np.asarray(bbox_to_wall_dist).reshape(num_edges, -1),
                                      np.asarray(orientation).reshape(num_edges, -1)], axis = 1)
            feature = np.nan_to_num(feature, nan = 0.0, posinf = 9999.0, neginf = -9999.0)
            g.edges['rf'].data['feature'] = torch.tensor(feature.astype(np.float32))
        
        return g
    
    def to_onehot(self, value, size):
        a = np.zeros(size)
        a[value] = 1
        return a
    
    def process(self):
        paths, ids = recursive_glob_full_path(self.dir_root)
        self.paths = paths
        self.ids = ids

    def recentering_scene(self, scene):
        room_nodes = scene['nodes']['room_nodes']
        furniture_nodes = scene['nodes']['furniture_nodes']
        all_bboxs = []
        for room_node in room_nodes:
            all_bboxs.extend(room_nodes[room_node]['bbox'])
        all_bboxs = np.asarray(all_bboxs)
        bbox_min = np.min(all_bboxs, axis=0)
        bbox_max = np.max(all_bboxs, axis=0)
        center = (bbox_min + bbox_max) / 2
        translate = -1 * center
        translate[1] = 0

        rotation_angle_index = torch.randint(low=0, high=4, size=(1,))
        orientation_modes = [np.pi, np.pi / 2, np.pi * 2, -np.pi / 2]

        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, orientation_modes[rotation_angle_index], 0])

        # centering room nodes
        for room_node in room_nodes:
            room_nodes[room_node]['bbox'] += translate
            room_nodes[room_node]['bbox'] = np.matmul(rotation_matrix, room_nodes[room_node]['bbox'].T).T

        pruned_furniture_nodes = {}
        for fnode in furniture_nodes:
            category = furniture_nodes[fnode]['model_info_index']['super-category']
            if str(category) in self.category_names:
                pruned_furniture_nodes[fnode] = furniture_nodes[fnode]
            else:
                continue
            if 'bbox_aligned' in furniture_nodes[fnode]:
                furniture_nodes[fnode]['bbox_aligned'] += translate
                furniture_nodes[fnode]['bbox_aligned'] = np.matmul(rotation_matrix, furniture_nodes[fnode]['bbox_aligned'].T).T

            if 'bbox_rotated' in furniture_nodes[fnode]:
                furniture_nodes[fnode]['bbox_rotated'] += translate
                furniture_nodes[fnode]['bbox_rotated'] = np.matmul(rotation_matrix, furniture_nodes[fnode]['bbox_rotated'].T).T

            if 'bbox_rotated_translated' in furniture_nodes[fnode]:
                furniture_nodes[fnode]['bbox_rotated_translated'] += translate
                furniture_nodes[fnode]['bbox_rotated_translated'] = np.matmul(rotation_matrix, furniture_nodes[fnode]['bbox_rotated_translated'].T).T

            if 'translation_vec' in furniture_nodes[fnode]:
                furniture_nodes[fnode]['translation_vec'] = np.asarray(furniture_nodes[fnode]['translation_vec']) + translate
                furniture_nodes[fnode]['translation_vec'] = np.matmul(rotation_matrix, furniture_nodes[fnode]['translation_vec']['bbox'].T).T

            if 'placement_center' in furniture_nodes[fnode]:
                furniture_nodes[fnode]['placement_center'] = np.asarray(furniture_nodes[fnode]['placement_center']) + translate

                if 'placement_orientation' in furniture_nodes[fnode]:
                    direction_vector = furniture_nodes[fnode]['placement_orientation']
                    base = furniture_nodes[fnode]['placement_center']
                    tip = furniture_nodes[fnode]['placement_center'] + direction_vector

                    base = np.matmul(rotation_matrix, base.T).T
                    tip = np.matmul(rotation_matrix, tip.T).T

                    direction_vector = tip - base

                    furniture_nodes[fnode]['placement_orientation'] = direction_vector

                furniture_nodes[fnode]['placement_center'] = np.matmul(rotation_matrix, furniture_nodes[fnode]['placement_center'].T).T

        scene['nodes']['room_nodes'] = room_nodes
        scene['nodes']['furniture_nodes'] = pruned_furniture_nodes

        return scene

    def __getitem__(self, i):
        path = self.paths[i]
        id = self.ids[i]
        
        with open(path, 'rb') as f:
            scene = pickle.load(f)

        scene = self.recentering_scene(scene)
        room_type = self.to_onehot(3, 4)
        
        if 'living' in id.lower():
            room_type = self.to_onehot(0, 4)
        if 'bedroom' in id.lower():
            room_type = self.to_onehot(1, 4)
        if 'library' in id.lower():
            room_type = self.to_onehot(2, 4)
            
        scene['room_type'] = room_type
        
        # Currently, in scene, we assume the nodes are fully connected.
        room_id_idx_dict, room_idx_id_dict = self.id_to_idx(scene['nodes']['room_nodes'])
        furniture_id_idx_dict, furniture_idx_id_dict = self.id_to_idx(scene['nodes']['furniture_nodes'])
    
        ff_src, ff_dst = self.get_edges_src_dst_ids(scene['edges']['furniture_to_furniture'], room_id_idx_dict, furniture_id_idx_dict)
        rr_src, rr_dst = self.get_edges_src_dst_ids(scene['edges']['room_to_room'], room_id_idx_dict, furniture_id_idx_dict)
        rf_src, rf_dst = self.get_edges_src_dst_ids(scene['edges']['room_to_furniture'], room_id_idx_dict, furniture_id_idx_dict)
        
        # full graph
        graph_data = {
            ('furniture', 'ff', 'furniture'): (torch.tensor(ff_src, dtype=torch.int32), torch.tensor(ff_dst, dtype=torch.int32)),
            ('room', 'rr', 'room'): (torch.tensor(rr_src, dtype=torch.int32), torch.tensor(rr_dst, dtype=torch.int32)),
            ('room', 'rf', 'furniture'): (torch.tensor(rf_src, dtype=torch.int32), torch.tensor(rf_dst, dtype=torch.int32))
        }
        g = dgl.heterograph(graph_data)  
        g = self.assign_node_features(g, 'room', scene) # N x 14 features

        try:
            g = self.assign_node_features(g, 'furniture', scene)  # N x 1040 features
        except ValueError:
            return None

        if g == None:
            return None

        g = self.assign_edge_features(g, 'ff', scene, room_idx_id_dict, furniture_idx_id_dict) # N x 3
        g = self.assign_edge_features(g, 'rr', scene, room_idx_id_dict, furniture_idx_id_dict) # N x 4
        g = self.assign_edge_features(g, 'rf', scene, room_idx_id_dict, furniture_idx_id_dict) # N x 5
        full_graph = g

        # room graph only
        graph_data = {
            ('furniture', 'ff', 'furniture'): (torch.tensor(ff_src, dtype=torch.int32), torch.tensor(ff_dst, dtype=torch.int32)),
            ('room', 'rr', 'room'): (torch.tensor(rr_src, dtype=torch.int32), torch.tensor(rr_dst, dtype=torch.int32)),
            ('room', 'rf', 'furniture'): (torch.tensor(rf_src, dtype=torch.int32), torch.tensor(rf_dst, dtype=torch.int32))
        }

        g1 = dgl.heterograph(graph_data)  
        g1 = self.assign_node_features(g1, 'room', scene) # N x 14 features 
        g1 = self.assign_edge_features(g1, 'rr', scene, room_idx_id_dict, furniture_idx_id_dict) # N x 4
        room_graph = g1
        return full_graph, room_graph, id
    
    def __len__(self):
        return len(self.paths)
 
if __name__ == "__main__":
    configs.update(configs.from_yaml(filename='configs/3d_front.yaml'))
    _3dfront = _3DFrontDataset(configs.io.dir_scene_graph)
    full_graph, room_graph = _3dfront[0]
    zx = 0