import numpy as np
from datasets._coohom import _CoohamDataset
import os
import dgl
from config import configs

def gen_a_room_mesh(g, idx):
    if 'room' not in g.ntypes:
        return 
    
    rnodes = g.nodes['room']
    node_type_dict = {'0': 'floor', '1': 'win', '2': 'door', '3': 'wall'}
    
    vertices = []
    faces = []
    
    for bbox, node_type in zip(rnodes.data['bbox'], rnodes.data['node_type']):
        # {'floor': 0, 'win': 1, 'door': 2, 'wall': 3}
        node_type = np.asarray(node_type)
        if node_type == 0:
            continue
        if node_type == 3:
            bbox = np.asarray(bbox)
            p0 = bbox[0]
            p1 = np.asarray((bbox[0, 0], bbox[1, 1], bbox[0, 2]))
            p2 = bbox[1]
            p3 = np.asarray((bbox[1, 0], bbox[0, 1], bbox[1, 2]))
            # adding one side
            vertices.append(p0)
            vertices.append(p1)
            vertices.append(p2)
            vertices.append(p3)
            idx = len(vertices) - 3
            faces.append(np.asarray((int(idx), int(idx+1), int(idx+2), int(idx+3))))
            # adding another side
            vertices.append(p3)
            vertices.append(p2)
            vertices.append(p1)
            vertices.append(p0)
            idx = len(vertices) - 3
            faces.append(np.asarray((int(idx), int(idx+1), int(idx+2), int(idx+3))))
            
        if node_type == 1 or node_type == 2: 
            bbox = np.asarray(bbox)
            p0 = bbox[0]
            p1 = np.asarray((bbox[0, 0], bbox[1, 1], bbox[0, 2]))
            p2 = bbox[1]
            p3 = np.asarray((bbox[1, 0], bbox[0, 1], bbox[1, 2]))
            
            vec1 = p2 - p0
            vec2 = np.asarray([0, 1, 0])
            dst_vec = np.cross(vec1, vec2)
            dst_vec = dst_vec / np.linalg.norm(dst_vec)
            
            dst_vec = dst_vec * 0.08
            
            p0a = p0 + dst_vec
            p1a = p1 + dst_vec
            p2a = p2 + dst_vec
            p3a = p3 + dst_vec
            
            p0b = p0 - dst_vec
            p1b = p1 - dst_vec
            p2b = p2 - dst_vec
            p3b = p3 - dst_vec
            
            vertices.append(p0a)
            vertices.append(p1a)
            vertices.append(p2a)
            vertices.append(p3a)
            
            idx = len(vertices) - 3
            faces.append(np.asarray((int(idx), int(idx+1), int(idx+2), int(idx+3))))
            
            vertices.append(p0b)
            vertices.append(p1b)
            vertices.append(p2b)
            vertices.append(p3b)
    
            idx = len(vertices) - 3
            faces.append(np.asarray((int(idx), int(idx+1), int(idx+2), int(idx+3))))
    
    text = ''
    for vertex in vertices:
        text = text + 'v ' + str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n'
        
    for face in faces:
        text = text  + 'f ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + ' ' + str(face[3]) + '\n'    
    
    os.makedirs('/mnt/ebs_xizhn5/Coohom/Data/rooms_mesh', exist_ok=True)
    out_path = os.path.join('/mnt/ebs_xizhn5/Coohom/Data/rooms_mesh', str(idx) + '.obj')
    
    with open(out_path, 'w') as f:
        f.write(text)
        
    zx = 0
    
if __name__ == "__main__":
    configs.update(configs.from_yaml(filename='configs/coohom.yaml'))
    coohom = _CoohamDataset(configs.io.dir_scene_graph, mode='Test')    
    for i in range(len(coohom)):
        try:
            full_gs, room_gs, id = coohom[i]
            gen_a_room_mesh(full_gs, i)
        except:
            continue