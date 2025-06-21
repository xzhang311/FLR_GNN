#!/usr/bin/env python3
"""Train L-CNN
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-identifier]
"""
import datetime
import os
import numpy as np
import cv2
import torch
import dgl
import random
from docopt import docopt
from config import configs
from trainer import Trainer
from datasets._3d_front import _3DFrontDataset
from datasets._3d_front import _3DFrontDataset
from dgl.dataloading import GraphDataLoader
from networks.AttentionVAE.flr_gnn_train import FLR_GNN_Train
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

def setup_random_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dgl.random.seed(0)

def get_device(args):
    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", len(args["--devices"].split(',')), "GPU(s)!")
        
        if len(args["--devices"].split(',')) == 1:
            print("Running code on GPU: {}".format(args["--devices"]))
            device_name = device_name + ':' + args["--devices"]    
    else:
        print("CUDA is not available")
    device = torch.device(device_name)
    return device

def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    # name += "-%s" % git_hash()
    name += "-%s" % identifier
    outdir = os.path.join(os.path.expanduser(configs.io.logdir), name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    configs.io.resume_from = outdir
    configs.to_yaml(os.path.join(outdir, "config.yaml"))
    os.system(f"git diff HEAD > {outdir}/gitdiff.patch")
    return outdir

def get_dataloader(dataset):
    """[summary]

    Args:
        dataset (DGLDataset): An instance of dataset defined by a class inherited from DGLDataset 
    """
    num_examples = len(dataset)
    num_train = int(num_examples * configs.training_params.train_ratio)
    
    # Use random sampler for training data
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    # Use sequential sampler for testing data
    test_sampler = SequentialSampler(torch.arange(num_train, num_examples))
    
    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=configs.training_params.batch_size, drop_last=False, num_workers = configs.params.nworkers)
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=configs.training_params.batch_size, drop_last=False, num_workers = configs.params.nworkers)
    
    return train_dataloader, test_dataloader
  
def get_optimizer(configs, model):
    if configs.training_params.optimizer == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=configs.training_params.lr,
            weight_decay=configs.training_params.weight_decay,
            amsgrad=configs.training_params.amsgrad,
        )
    elif configs.training_params.optimizer == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=configs.training_params.lr,
            weight_decay=configs.training_params.weight_decay,
            momentum=configs.training_params.momentum,
        )
    else:
        raise NotImplementedError
    
    return optim
  
def main():
    args = docopt(__doc__)  
    config_file = args["<yaml-config>"] or "config/lsun.yaml"
    configs.update(configs.from_yaml(filename = config_file))

    setup_random_seeds()
    
    resume_from = configs.io.resume_from
    dataset = _3DFrontDataset(configs.io.dir_scene_graph)
    train_loader, test_loader = get_dataloader(dataset)
    device = get_device(args)
    model = FLR_GNN_Train()
    model = torch.nn.DataParallel(model, [int(i) for i in args["--devices"].split(',')])
    model = model.to(device)
    
    optimizer = get_optimizer(configs, model)
    outdir = resume_from or get_outdir(args["--identifier"])
    print("outdir:", outdir)
    
    trainer = Trainer(model, device, train_loader, test_loader, optimizer, outdir, configs)
    trainer.run()
    
if __name__ == "__main__":
    main()