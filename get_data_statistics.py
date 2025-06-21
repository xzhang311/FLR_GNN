import numpy as np
from datasets.coohom import Coohom
import os
import dgl
from config import configs

def get_node_counts(g):
    n_fnodes = g.number_of_nodes('furniture')
    n_rnodes = g.number_of_nodes('room')
    rnodes = g.nodes['room']
    fnodes = g.nodes['furniture']
    
    nwins = 0
    ndoors = 0
    nwalls = 0
    
    for node_type in rnodes.data['node_type']:
        if node_type == 1:
            nwins += 1
        if node_type == 2:
            ndoors += 1
        if node_type == 3:
            nwalls += 1
            
    return n_fnodes, nwins, ndoors, nwalls

if __name__ == "__main__":
    configs.update(configs.from_yaml(filename='configs/coohom.yaml'))
    coohom = Coohom(configs.io.dir_scene_graph, mode='Test')
    
    n_fnodes = []
    nwins = []
    ndoors = [] 
    nwalls = []
    
    for i in range(len(coohom)):
        try:
            full_gs, room_gs = coohom[i]
            n_fnodes_, nwins_, ndoors_, nwalls_ = get_node_counts(full_gs)
            
            n_fnodes.append(n_fnodes_)
            nwins.append(nwins_)
            ndoors.append(ndoors_)
            nwalls.append(nwalls_)
        except:
            continue
    
    fsum = np.sum(np.asarray(n_fnodes))
    winsum = np.sum(np.asarray(nwins))
    dsum = np.sum(np.asarray(ndoors))
    wallsum = np.sum(np.asarray(nwalls))
    
    fmean = np.mean(np.asarray(n_fnodes))
    winmean = np.mean(np.asarray(nwins))
    dmean = np.mean(np.asarray(ndoors))
    wallmean = np.mean(np.asarray(nwalls))
    
    fstd = np.std(np.asarray(n_fnodes))
    winstd = np.std(np.asarray(nwins))
    dstd = np.std(np.asarray(ndoors))
    wallstd = np.std(np.asarray(nwalls))
    
    print('Sum furniture: {}   wins: {}    doors: {}   walls: {}'.format(fsum, winsum, dsum, wallsum))
    print('Mean furnirutre: {}   wins: {}    doors: {}   walls: {}'.format(fmean, winmean, dmean, wallmean))    
    print('Std furnirutre: {}   wins: {}    doors: {}   walls: {}'.format(fstd, winstd, dstd, wallstd))    
