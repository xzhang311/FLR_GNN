import os
import os.path as osp
import numpy as np
import cv2
import torch
import dgl
import random
import time
import shutil
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
from docopt import docopt
from config import configs
from loss import compute_kl_div, compute_edge_loss, compute_furniture_node_loss

class Trainer(object):
    def __init__(self, model, device, train_loader, test_loader, optimizer, out, configs):
        self.device = device
        self.model = model
        self.optim = optimizer
        self.configs = configs
        self.train_loader = train_loader
        self.val_loader = test_loader
        self.loss_labels = ["Total"] + ["KL_div"] + ["Edge_loss"] + ["Node_loss"]
        self.metrics = np.zeros([len(self.loss_labels)])
        self.ave_metrics = None
        self.iteration = 0
        self.batch_size = configs.training_params.batch_size
        self.mean_loss = self.best_mean_loss = 1e1000
        self.out = out
        self.epoch = 0
        
    def run_tensorboard(self):
        board_out = osp.join(self.out, "tensorboard")
        if not osp.exists(board_out):
            os.makedirs(board_out)
        self.writer = SummaryWriter(board_out)
        self.pred_img_writer = []
        for i in range(configs.training_params.n_val_imgs):
            self.pred_img_writer.append(SummaryWriter(os.path.join(board_out, 'Imgs', str(i))))

    def _write_metrics(self, size, prefix):
        for label, metric in zip(self.loss_labels, self.metrics):
            self.writer.add_scalar(
                    f"{prefix}/{label}", metric / size, self.iteration
                )
        prt_str = (
                f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                + "| ".join(map("{:.5f}".format, self.metrics / size))
            )
        pprint(prt_str, " " * 7)
        
    def compute_total_loss(self, pred_g, gt_g, mu, log_var, is_train = True):
        # kl_loss_scale = 0.001
        # edge_loss_scale = 0.000000001
        # node_loss_scale = 0.0001
        
        kl_loss_scale = 1
        edge_loss_scale = 1
        node_loss_scale = 1
        
        kl_loss = compute_kl_div(mu, log_var) * kl_loss_scale
        edge_loss = compute_edge_loss(pred_g, gt_g) * edge_loss_scale
        node_loss = compute_furniture_node_loss(pred_g, gt_g) * node_loss_scale
        total_loss = kl_loss + edge_loss + node_loss
        
        print()
        if is_train:
            print(
            "| ".join(
                ["progress "]
                + list(map("{:7}".format, self.loss_labels))
                + ["speed"])
            )
        
        for j, name in enumerate(self.loss_labels):
            if name == 'Total':
                self.metrics[j] += total_loss
            if name == 'KL_div':
                self.metrics[j] += kl_loss
            if name == 'Edge_loss':
                self.metrics[j] += edge_loss
            if name == 'Node_loss':
                self.metrics[j] += node_loss
        
        return total_loss
    
    def train_epoch(self):
        time = timer()
        self.model.train()
        for full_graph, room_graph in self.train_loader:
            # move data to gpu if provided
            full_graph = full_graph.to(self.device)
            room_graph = room_graph.to(self.device)
            
            self.optim.zero_grad()
            self.metrics[...] = 0
            pred_g, mu, log_var = self.model(full_graph, room_graph)
            total_loss = self.compute_total_loss(pred_g, full_graph, mu, log_var)
            total_loss.backward()
            self.optim.step()
            
            if self.ave_metrics is None:
                self.ave_metrics = self.metrics
            else:
                self.ave_metrics = self.ave_metrics * 0.9 + self.metrics * 0.1
                
            self._write_metrics(size = 1, prefix = "train")
            
            if self.iteration % 4 == 0:
                tprint(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, self.ave_metrics))
                    + f"| {4 * self.batch_size / (timer() - time):04.1f} "
                )
                time = timer()
            self.iteration += 1
                
    def validate_epoch(self):
        total_loss = 0
        
        viz = osp.join(self.out, "viz", f"{self.iteration * configs.training_params.batch_size:09d}")
        npz = osp.join(self.out, "npz", f"{self.iteration * configs.training_params.batch_size:09d}")
        osp.exists(viz) or os.makedirs(viz)
        osp.exists(npz) or os.makedirs(npz)
        
        self.model.eval()
        with torch.no_grad():
            for full_graph, room_graph in self.val_loader:
                # move data to gpu if provided
                full_graph = full_graph.to(self.device)
                room_graph = room_graph.to(self.device)
                
                pred_g, mu, log_var = self.model(full_graph, room_graph)
                total_loss += self.compute_total_loss(pred_g, full_graph, mu, log_var, is_train=False)
            
            self._write_metrics(size = len(self.val_loader), prefix = "validation")  
            
            torch.save(
                {
                    "iteration": self.iteration,
                    "arch": self.model.__class__.__name__,
                    "optim_state_dict": self.optim.state_dict(),
                    "model_state_dict": self.model.state_dict(),
                    "best_mean_loss": self.best_mean_loss,
                },
                osp.join(self.out, "checkpoint_latest.pth"),
            )
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth"),
                osp.join(npz, "checkpoint.pth"),
            )
            if self.mean_loss < self.best_mean_loss:
                self.best_mean_loss = self.mean_loss
                shutil.copy(
                    osp.join(self.out, "checkpoint_latest.pth"),
                    osp.join(self.out, "checkpoint_best.pth"),
                )
    
    def run(self):
        self.run_tensorboard()
        for self.epoch in range(configs.training_params.n_epochs):
            # check if need to decay lr
            if self.epoch % configs.training_params.lr_decay_epoch:
                for i_group in range(len(self.optim.param_groups)):
                        self.optim.param_groups[i_group]["lr"] /= 10

            self.train_epoch()
            self.validate_epoch()
            
def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")

def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)