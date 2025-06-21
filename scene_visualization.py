import numpy as np
import dgl
import cv2
import os
import random
from datasets._coohom import _CoohamDataset
from config import configs

def tensor_to_numpy(input):
    if input.device.type=='cpu':
        input = input.cpu().detach().numpy()
    else:
        input = input.gpu().detach().numpy()
    return input
        
def get_bbox_of_scene(rnodes):
    bboxs = rnodes.data['bbox']
    
    if bboxs.device.type == 'cpu':
        bboxs = bboxs.cpu().detach().numpy()
    else:
        bboxs = bboxs.gpu().detach().numpy()
    
    bboxs = bboxs.reshape(-1, 3)
    
    minx, miny, minz = np.min(bboxs, axis = 0)
    maxx, maxy, maxz = np.max(bboxs, axis = 0)
    
    rst_bbox = {}
    rst_bbox['dim_xyz'] = np.asarray((maxx - minx, maxy - miny, maxz - minz))
    rst_bbox['min_xyz'] = np.asarray((minx, miny, minz))
    rst_bbox['max_xyz'] = np.asarray((maxx, maxy, maxz))
    rst_bbox['center'] = np.asarray(((maxx + minx)/2, (maxy + miny)/2, (maxz + minz)/2))
    
    return rst_bbox

def get_transformation(bbox, img_w, img_h, margin = 10):
    """[summary]

    Args:
        bbox (dict): bounding box infomation of a room. Assume Y direction is vertical direction
        img_h (scalar): [description]
        img_w (scalar): [description]
    """
    img_center = np.array((img_w/2, 0, img_h/2))
    translate = img_center - bbox['center']
    translate[1] = 0 # No translation in vertical direction
    scale = np.min(np.array(((img_w - margin * 2)/bbox['dim_xyz'][0], (img_h - margin * 2)/bbox['dim_xyz'][2])))

    return translate, scale

def draw_room(canvas, rnodes, room_center, translate, scale):
    def transform_bbox(bbox, room_center, translate, scale):
        origin_trans = np.mean(bbox, axis = 0)
        bbox = (bbox - room_center) * scale + room_center
        bbox = bbox + translate
        return bbox.astype(np.int32)
    
    def draw_bbox(canvas, bbox, color, thickness=1):
        p0 = (bbox[0, 0], bbox[0, 2])
        p1 = (bbox[1, 0], bbox[1, 2])
        canvas = cv2.line(canvas, p0, p1, color, thickness)
        return canvas

    node_type_dict = {'0': 'floor', '1': 'win', '2': 'door', '3': 'wall'}
    colors = {'wall': [255, 255, 255], 'win': [255, 0, 0], 'door': [ 0, 255, 0]}
    thicknesses = {'wall': 3, 'win': 9, 'door': 9}
    
    bboxs = rnodes.data['bbox']
    node_types = rnodes.data['node_type']
    
    for bbox, node_type in zip(bboxs, node_types):
        bbox = tensor_to_numpy(bbox)
        center1 = np.mean(bbox, axis = 0)
        print('center1: {}'.format(center1))
        node_type = int(tensor_to_numpy(node_type))

        node_type = node_type_dict[str(node_type)]
        
        if node_type == 'floor':
            continue
        
        bbox = transform_bbox(bbox, room_center, translate, scale)
        center2 = np.mean(bbox, axis = 0)
        print('center2: {}'.format(center2))
        
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        
        
        canvas = draw_bbox(canvas, bbox, color, thicknesses[node_type])
        
    return canvas
        
def draw_furniture(canvas, fnodes, room_center, translate, scale):
    def transform_corners(bbox, room_center, translate, scale):
        origin_trans = np.mean(bbox, axis = 0)
        bbox = (bbox - room_center) * scale + room_center
        bbox = bbox + translate
        return bbox.astype(np.int32)
    
    def get_corners(center, ori, size):
        corners = []
        up = np.asarray((0, 1, 0))
        ori_2 = np.cross(ori, up)
        
        x, y, z = size[0]/2, size[1]/2, size[2]/2

        corners.append(center + z * ori + x * ori_2)
        corners.append(center + z * ori + -x * ori_2)
        corners.append(center + -z * ori + -x * ori_2)
        corners.append(center + -z * ori + x * ori_2)
        
        corners = np.asarray(corners)
        
        return corners
    
    def draw_bbox(canvas, corners, ori, color, thickness=1):
        ori = np.asarray((ori[0], ori[2]))
        ori = ori/np.linalg.norm(ori)
        p0 = (corners[0, 0], corners[0, 2])
        p1 = (corners[1, 0], corners[1, 2])
        p2 = (corners[2, 0], corners[2, 2])
        p3 = (corners[3, 0], corners[3, 2])
        canvas = cv2.line(canvas, p0, p1, color, thickness)
        canvas = cv2.line(canvas, p1, p2, color, thickness)
        canvas = cv2.line(canvas, p2, p3, color, thickness)
        canvas = cv2.line(canvas, p3, p0, color, thickness)
        
        vec1 = np.asarray(p0) - np.asarray(p1)
        vec2 = np.asarray(p1) - np.asarray(p2)
        
        vec1_normed = vec1/np.linalg.norm(vec1)
        vec2_normed = vec2/np.linalg.norm(vec2)
        
        diff1 = np.abs(np.dot(vec1_normed, ori))
        diff2 = np.abs(np.dot(vec2_normed, ori))
        
        if diff1 > diff2:
            l = np.linalg.norm(vec1)
        else:
            l = np.linalg.norm(vec2)
        
        center = (np.asarray(p0) + np.asarray(p1) + np.asarray(p2) + np.asarray(p3))/4
        tmp = center + ori * l * 0.7
        canvas = cv2.line(canvas, tuple(center.astype(np.int16)), tuple(tmp.astype(np.int16)), color, thickness)
        
        return canvas
    
    # {'id': 0, 'category': 'Cabinet/Shelf/Desk'},
    # {'id': 1, 'category': 'Bed'},
    # {'id': 2, 'category': 'Chair'},
    # {'id': 3, 'category': 'Table'},
    # {'id': 4, 'category': 'Sofa'},
    # {'id': 5, 'category': 'Pier/Stool'},
    # {'id': 6, 'category': 'Lighting'},
    
    colors = {'0': [255, 255, 0],
              '1': [255, 0, 255],
              '2': [0, 255, 255],
              '3': [255, 128, 128],
              '4': [128, 255, 128],
              '5': [ 128, 128, 255],
              '6': [128, 128, 128],
              '7': [64, 255, 255]}
    
    categorys = fnodes.data['super_category']
    centers = fnodes.data['placement_center']
    orientations = fnodes.data['placement_orientation']
    sizes = fnodes.data['object_dimension']
    
    categorys = np.argmax(tensor_to_numpy(categorys), axis = 0)
    centers = tensor_to_numpy(centers)
    orientations = tensor_to_numpy(orientations)
    sizes = tensor_to_numpy(sizes)
        
    for center, ori, size, category in zip(centers, orientations, sizes, categorys):
        if category > 7:
            category = 7
        try:
            corners = get_corners(center, ori, size)
            corners = transform_corners(corners, room_center, translate, scale)
            canvas = draw_bbox(canvas, corners, ori, colors[str(category)], thickness = 9)
        except:
            zx = 0
    return canvas

def visualize_a_graph(g, id):
    """[summary]

    Args:
        g (dgl heterogeneous graph): A dgl heterogeneous graph 
    """
    id = id.split('***')[0]
    out_path = os.path.join('/mnt/ebs_xizhn5/Coohom/Data/rooms/visualization', str(id) + '.png')
    
    if os.path.isfile(out_path):
        return
    
    intensity = 100
    canvas_color = [[intensity, 0, 0], #living room
                    [0, intensity, 0], # bedroom
                    [0, 0, intensity], # library
                    [intensity, intensity, 0]] # other
    
    canvas = np.zeros((configs.params.img_height, configs.params.img_width, 3))
    room_center = None
    if 'room' in g.ntypes:
        rnodes = g.nodes['room']
        room_type = np.argmax(tensor_to_numpy(rnodes.data['feature'][:, 0]))
        canvas = canvas + canvas_color[int(room_type)]
        rbbox = get_bbox_of_scene(rnodes)
        room_center = rbbox['center']
        translate, scale = get_transformation(rbbox, img_w = configs.params.img_width, img_h = configs.params.img_height)
        canvas = draw_room(canvas, rnodes, room_center, translate, scale)
    if 'furniture' in g.ntypes:
        fnodes = g.nodes['furniture']
        canvas = draw_furniture(canvas, fnodes, room_center, translate, scale)

    # convert to RGB
    canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_path, canvas)
    print(id)

if __name__ == "__main__":
    configs.update(configs.from_yaml(filename='configs/coohom.yaml'))
    coohom = _CoohamDataset(configs.io.dir_scene_graph, mode='Test')
    
    for i in range(len(coohom)):
        try:
            full_gs, room_gs, id = coohom[i]
            visualize_a_graph(full_gs, id)
        except:
            continue