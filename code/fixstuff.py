
from utils import *

import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph,get_laplacian,to_dense_adj,subgraph
import numpy as np
import random
import torch
from scipy.stats import linregress
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

import os

from torch_geometric.datasets import TUDataset, Planetoid, GNNBenchmarkDataset, Coauthor, WebKB, WikipediaNetwork
from torch_geometric.loader import DataLoader
import math
import torch

"""
# example change edge
datasetname = 'Cora'

dataset = Planetoid(root = 'testedge', name = datasetname, pre_transform=add_edge_attr)
print(dataset[0])

trans = T.Compose([T.NormalizeFeatures(),edge_trim_w])
dataset2 = Planetoid(root = 'testedge', name = datasetname, transform = trans, pre_transform=add_edge_attr)
print(dataset2[0])

"""











"""
# fix webkb
device = torch.device('cpu')
dataset_name = 'chameleon'
params = {}
params['num_layers'] = 2
params['hidden_dim'] = 64
params['hidden_dim_post'] = 32
params['lr'] = 0.01
params['batch_size'] = 1
dataset = WikipediaNetwork(root = 'fix', name = dataset_name, pre_transform=func1)
#print(dataset[0])

train_d, dataloaders = get_dataloaders(params, dataset_name, func1, False, folder_loc = 'fix')
model,optimizer = get_model_and_opt('gat', params, train_d,device)

test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
print(test_result)
#dataset = WebKB(root = 'fix', name = dataset_name, pre_transform=func1)
print(dataset[0])
print(dataset[0].train_mask)
"""