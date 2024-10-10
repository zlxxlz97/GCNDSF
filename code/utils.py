import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph,get_laplacian,to_dense_adj,subgraph
import numpy as np
import random
import torch
from scipy.stats import linregress
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

import os

from torch_geometric.datasets import TUDataset, Planetoid, GNNBenchmarkDataset, Coauthor, WebKB, Amazon, WikipediaNetwork
from torch_geometric.loader import DataLoader
import math
import torch

from models import *
from models2 import *


def ei_decomposition(m, asscending = True):
    u,v = np.linalg.eigh(m)
    idx = u.argsort()   
    u.sort()
    v = v[:,idx]
    return u,v


def return_one_subgraph_spectrum(edge_index):

    lap_sym = get_laplacian(edge_index, normalization='sym')
    L = to_dense_adj(lap_sym[0], edge_attr=lap_sym[1])
    A = to_dense_adj(edge_index)
    L = L.cpu().detach().numpy()
    A = A.cpu().detach().numpy()

    ua, va = ei_decomposition(A)
    ul, vl = ei_decomposition(L)
    return ua,ul


# preprocessing
def func1(d):
    print('replacing nan values to 0 in data.x')
    if torch.isnan(d.x).any():
        d.x = torch.nan_to_num(d.x)

    # first row normalize features .x
    normed_x =  d.x / d.x.sum(dim=-1).unsqueeze(-1)
    
        
    # computing features
    FA,FL = spectra_1(d, 1, 200)
    FA = torch.FloatTensor(FA)
    FL = torch.FloatTensor(FL)
    print('replacing nan values to 0 in fa')
    if torch.isnan(FA).any():
        FA = torch.nan_to_num(FA)
    print('replacing nan values to 0 in fl')
    if torch.isnan(FL).any():
        FL = torch.nan_to_num(FL)
    FA = FA / FA.sum(dim=-1).unsqueeze(-1)
    FL = FL / FL.sum(dim=-1).unsqueeze(-1)

    d.x = torch.cat([normed_x, FA, FL],1)

    print('to be safe replacing nan values to 0 in data.x again')
    if torch.isnan(d.x).any():
        d.x = torch.nan_to_num(d.x)
    return d

def func2(d):
    print('replacing nan values to 0 in data.x')
    if torch.isnan(d.x).any():
        d.x = torch.nan_to_num(d.x)
    # first row normalize features .x
    normed_x =  d.x / d.x.sum(dim=-1).unsqueeze(-1)
    # computing features
    FA,FL = spectra_1(d, 2, 200)
    FA = torch.FloatTensor(FA)
    FL = torch.FloatTensor(FL)
    print('replacing nan values to 0 in fa')
    if torch.isnan(FA).any():
        FA = torch.nan_to_num(FA)
    print('replacing nan values to 0 in fl')
    if torch.isnan(FL).any():
        FL = torch.nan_to_num(FL)
    FA = FA / FA.sum(dim=-1).unsqueeze(-1)
    FL = FL / FL.sum(dim=-1).unsqueeze(-1)

    d.x = torch.cat([normed_x, FA, FL],1)
    print('to be safe replacing nan values to 0 in data.x again')
    if torch.isnan(d.x).any():
        d.x = torch.nan_to_num(d.x)
    return d


def func3(d):
    print('replacing nan values to 0 in data.x')
    if torch.isnan(d.x).any():
        d.x = torch.nan_to_num(d.x)
    # first row normalize features .x
    normed_x =  d.x / d.x.sum(dim=-1).unsqueeze(-1)
    # computing features
    FA,FL = spectra_2(d, 1, 200, 10,10)
    FA = torch.FloatTensor(FA)
    FL = torch.FloatTensor(FL)
    print('replacing nan values to 0 in fa')
    if torch.isnan(FA).any():
        FA = torch.nan_to_num(FA)
    print('replacing nan values to 0 in fl')
    if torch.isnan(FL).any():
        FL = torch.nan_to_num(FL)
    FA = FA / FA.sum(dim=-1).unsqueeze(-1)
    FL = FL / FL.sum(dim=-1).unsqueeze(-1)

    d.x = torch.cat([normed_x, FA, FL],1)
    print('to be safe replacing nan values to 0 in data.x again')
    if torch.isnan(d.x).any():
        d.x = torch.nan_to_num(d.x)
    return d

def func4(d):
    print('replacing nan values to 0 in data.x')
    if torch.isnan(d.x).any():
        d.x = torch.nan_to_num(d.x)
    # first row normalize features .x
    normed_x =  d.x / d.x.sum(dim=-1).unsqueeze(-1)
    # computing features
    FA,FL = spectra_2(d, 2, 200, 10,10)
    FA = torch.FloatTensor(FA)
    FL = torch.FloatTensor(FL)
    print('replacing nan values to 0 in fa')
    if torch.isnan(FA).any():
        FA = torch.nan_to_num(FA)
    print('replacing nan values to 0 in fl')
    if torch.isnan(FL).any():
        FL = torch.nan_to_num(FL)
    FA = FA / FA.sum(dim=-1).unsqueeze(-1)
    FL = FL / FL.sum(dim=-1).unsqueeze(-1)

    d.x = torch.cat([normed_x, FA, FL],1)
    print('to be safe replacing nan values to 0 in data.x again')
    if torch.isnan(d.x).any():
        d.x = torch.nan_to_num(d.x)

    return d

def f11(d):
    cropped_x = d.x[:,:-12]
    d.x = cropped_x
    return d

def f12(d):
    cropped_x = d.x[:,:-7]
    d.x = cropped_x
    return d

def f13(d):
    cropped_x = d.x[:,:-12]
    cropped_l_spec =  d.x[:,-7:]
    d.x = torch.cat([cropped_x,cropped_l_spec],1)
    return d


def f21(d):
    cropped_x = d.x[:,:-20]
    d.x = cropped_x
    return d

def f22(d):
    cropped_x = d.x[:,:-10]
    d.x = cropped_x
    return d

def f23(d):
    cropped_x = d.x[:,:-10]
    cropped_l_spec =  d.x[:,-10:]
    d.x = torch.cat([cropped_x,cropped_l_spec],1)
    return d



# return 5+7 discriptors of A spectra and L spectra
def spectra_1(data,k_hop = 2, max_nodes = 200):
    UA,UL = get_spectrum(data, k_hop, max_nodes)
    FA,FL = process_features_1(UA,UL)
    return FA, FL

def spectra_2(data,k_hop = 2, max_nodes = 200, length_A = 10, length_B = 10):
    UA,UL = get_spectrum(data, k_hop, max_nodes)
    FA,FL = process_features_2(UA, UL, max_nodes, length_A, length_B)
    return FA, FL

def get_spectrum(data,k_hop = 2, max_nodes = 200):
    UA = []
    UL = []
    for i in range(data.x.shape[0]):
        k_subgraph = k_hop_subgraph(i , k_hop, data.edge_index, flow = "target_to_source", relabel_nodes = True)
        #k_subgraph = k_hop_subgraph(i , k_hop, data.edge_index,relabel_nodes = True)
        #assert k_subgraph[0].shape[0]>1, "exists isolated point"
        if k_subgraph[0].shape[0]<=1:
            #print("isolated node spoted")
            #print(data.contains_isolated_nodes())
            #print(i)
            UA.append(np.array([0,0]))
            UL.append(np.array([0,0]))
            continue

        if k_subgraph[0].shape[0]>max_nodes:

            pos = k_subgraph[2].item()
            temp_list = k_subgraph[0].tolist()

            cent = temp_list[pos]
            if pos == k_subgraph[0].shape[0]-1:
                except_temp_list = temp_list[:-1]
            else:
                except_temp_list = temp_list[:pos]+temp_list[pos+1:]   

            random.shuffle(except_temp_list)
            new_list = except_temp_list[:max_nodes-1]+[cent]
            new_list.sort()
            new_nodes_tensor =  torch.LongTensor(new_list)
            
            _,old_edge_index,_,_ = k_hop_subgraph(i , k_hop, data.edge_index, flow = "target_to_source", relabel_nodes = False)
            new_subgraph_edge_index,_ = subgraph(new_nodes_tensor, old_edge_index,relabel_nodes=True)
        else:
            new_subgraph_edge_index = k_subgraph[1]

        ua,ul = return_one_subgraph_spectrum(new_subgraph_edge_index)
        ua = np.real_if_close(ua,100)
        ua = np.around(ua,4)
        ul = np.real_if_close(ul,100)
        ul = np.around(ul,4)
        UA.append(ua)
        UL.append(ul)
    
    return UA,UL



def process_features_1(UA,UL):
    FA = get_features_A(UA)
    FL = get_features_L(UL)
    return FA,FL

def process_features_2(UA,UL,max_nodes = 200, length_A = 10, length_B = 10):
    FA = get_dist_A(UA, max_nodes, length_A)
    FL = get_dist_L(UL, length_B)
    return FA,FL

# return empirical probability
def get_dist_A(m, max_nodes = 200, length = 10):
    # lies in [-max, max]
    # UA in -200, 200
    DA = []
    delta = 2*max_nodes/length
    for i in range(len(m)):

        spec = m[i].flatten()  # numpy array ascending

        # isolated node return all 0
        if (len(spec)==2) and (not np.any(spec)):
            temp = np.zeros(length)
            DA.append(temp)
            continue
        
        d_spec = [0]*length
        sp_list = list(spec)
        sp_list = [math.floor((s-(-max_nodes))/delta) for s in sp_list]

        for s in sp_list:
            if s==length:
                d_spec[s-1] = d_spec[s-1]+1
            else:
                d_spec[s] = d_spec[s]+1
        
        DA.append(np.array(d_spec))
            
    return np.stack(DA, axis=0)

def get_dist_L(m, length = 10):
    # lies in [0,2]
    DL = []
    delta = 2/length
    for i in range(len(m)):

        spec = m[i].flatten()  # numpy array ascending

        # isolated node return all 0
        if (len(spec)==2) and (not np.any(spec)):
            temp = np.zeros(length)
            DL.append(temp)
            continue
        
        d_spec = [0]*length
        sp_list = list(spec)
        sp_list = [math.floor(s/delta) for s in sp_list]

        for s in sp_list:
            if s==length:
                d_spec[s-1] = d_spec[s-1]+1
            else:
                d_spec[s] = d_spec[s]+1
        
        DL.append(np.array(d_spec))
            
    return np.stack(DL, axis=0)


# features of A: rad,second_largest_ev,lower_slope, upper_slope ,energy
def get_features_A(m):
    FA = []
    for i in range(len(m)):

        spec = m[i].flatten()  # numpy array ascending

        if (len(spec)==2) and (not np.any(spec)):
            # isolated node
            temp = np.array([0,0,0,0,0])
            FA.append(temp)
            continue

        rad = max([abs(spec[0]),abs(spec[-1])])
        if abs(spec[0])>abs(spec[-1]):
            second_largest_ev = max([abs(spec[1]),abs(spec[-1])])
        else:
            second_largest_ev = max([abs(spec[0]),abs(spec[-2])])

        # find the first >0
        p0 = -1
        for p in range(spec.size-1):
            if spec[p]<=1e-4 and spec[p+1]>1e-4:
                p0 = p+1
                break
        if p0>0 and spec.size-p0>=2:
            upper_slope, _, _, _, _ = linregress(np.arange(1, spec.size-p0+1), spec[p0:])
        else:
            upper_slope = 0

        if p0<0:
            # shouldn't happen
            lower_slope=0
        elif p0-1>0:
            lower_slope, _, _, _, _ = linregress(np.arange(1, p0+1), spec[:p0])
        else:
            lower_slope = 0
        

        energy = sum(np.square(spec))
        temp = np.array([rad,second_largest_ev,lower_slope, upper_slope ,energy])
        FA.append(temp)
    return np.stack(FA, axis=0)

def get_features_L(m):
    FL = []
    for i in range(len(m)):
        spec = m[i].flatten()  # numpy array ascending

        if (len(spec)==2) and (not np.any(spec)):
            # isolated node
            temp = np.array([0,0,0,0,0,0,0])
            FL.append(temp)
            continue

        # first find index for first>0, first = 1, first>1, first =2
        # which i named p1,p2,p3,p4
        
        p1,p2,p3,p4 = -1,-1,-1,-1
        for p in range(spec.size-1):
            if spec[p]<=1e-4 and spec[p+1]>1e-4:
                p1 = p+1
            if spec[p]<1-1e-4 and abs(spec[p+1]-1)<=1e-4:
                p2 = p+1
            if spec[p]<=1+1e-4 and spec[p+1]-1>1e-4:
                p3 = p+1
            if spec[p]<2-1e-4 and abs(spec[p+1]-2)<=1e-4:
                p4 = p+1
        
        # multiplicity of 0: p1 is the multiplicity of 0
        #assert p1 >0 , "p1 is wrong"
        if p1>0:
            mul_0 = p1
        else:
            mul_0 = -1

        # multiplicity of 1:
        if p2>0:
            if p3>0:
                mul_1 = p3-p2
            else:
                mul_1 = spec.size-p2
        else:
            mul_1 = 0

        # multiplicity of 2:
        if p4>0:
            mul_2 = spec.size-p4
        else:
            mul_2 = 0

        # debugging
        #print(p1,p2,p3,p4)
        #print(spec)
        
        if len(spec)==2:
            temp = np.array([1,0,1,0,0,2,4])
            FL.append(temp)
            continue
        
        # now at least three nodes
        # first>0, first = 1, first>1, first =2
        # compute upper slope
        if p3>0:
            # exists something>1 p3,...,p4
            if p4>0:
                if p4-p3>0:
                    upper_slope, _, _, _, _ = linregress(np.arange(1, p4-p3+2),spec[p3:p4+1])
                else:
                    upper_slope = 0
            else:
                # 1<last 1 <2
                if p3==spec.size-1:
                    upper_slope = 0
                else:
                    # p3,...,last
                    upper_slope, _, _, _, _ = linregress(np.arange(1, spec.size-p3+1), spec[p3:])
        else:
            # last 1 <1
            upper_slope = 0

        # compute lower slope
        if spec[-1]<1:
            lower_slope, _, _, _, _ = linregress(np.arange(1, spec.size),spec[1:])
        else:
            # last 1 >=1
            if p2>0:
                if p2>1:
                    lower_slope, _, _, _, _ = linregress(np.arange(1, p2+1),spec[1:p2+1])
                else:
                    lower_slope = 0
            else:
                # no==1
                if p3-1>1:
                    lower_slope, _, _, _, _ = linregress(np.arange(1, p3),spec[1:p3])
                else:
                    lower_slope = 0
        

        sum_all = sum(spec)
        energy = sum(np.square(spec))
        temp = np.array([mul_0,mul_1,mul_2,lower_slope, upper_slope,sum_all,energy])
        FL.append(temp)
    return np.stack(FL, axis=0)

    



def set_all_seed(num, device):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    if device.type == 'cuda':
      torch.cuda.manual_seed(num)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

def train_one_epoch(model, data_loader, optimizer, dataset_name, device):
    model.train()
    
    train_loss_epoch = 0
    train_acc_epoch = 0
    correct = 0
    card = 0

    for ind, batch_data in enumerate(data_loader):
        optimizer.zero_grad()
        batch_data = batch_data.to(device)

        try:
            out = model(batch_data)
        except RuntimeError as exception:
            if 'out of memory' in str(exception):
                print('clearing cache')
                torch.cuda.empty_cache()
                out = model(batch_data)
                    
        #out = model(batch_data)

        #print(torch.isnan(batch_data.x).any())
        if dataset_name in ['Cora','Citeseer','Pubmed','Cornell','Texas','Wisconsin','CS','Physics','Computers','Photo','ogbn-arxiv','chameleon', 'squirrel']:
            #print('debugging')
            #print(out[batch_data.train_mask].shape)
            #print( batch_data.y[batch_data.train_mask].flatten().shape)
            loss = F.nll_loss(out[batch_data.train_mask], batch_data.y[batch_data.train_mask].flatten())
            #print('debug1')
            #print(out[batch_data.train_mask])

            pred_out = out.argmax(dim=1)[batch_data.train_mask].detach().cpu()
            lab = batch_data.y[batch_data.train_mask].detach().cpu().flatten()
            correct_batch = ( pred_out== lab).sum()
            #acc = int(correct) / int(batch_data.train_mask.sum())
            correct += correct_batch
            card += int(batch_data.train_mask.sum())
        else:
            loss = F.nll_loss(out, batch_data.y)
            correct_batch = (out.argmax(dim=1) == batch_data.y.flatten()).sum()
            #acc = int(correct) / int(batch_data.train_mask.sum())
            correct += correct_batch
            
            #card += int(batch_data.shape[0])
            card += batch_data.y.shape[0]


        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
        
    assert (not torch.isnan(batch_data.x).any()) ,"not updating"
    train_acc_epoch = correct/card

    return train_loss_epoch, train_acc_epoch


def test(model, data_loader, dataset_name, mode , device):
    model.eval()
    loss_epoch = 0
    acc_epoch = 0
    correct = 0
    card = 0
    with torch.no_grad():
        for ind, batch_data in enumerate(data_loader):

            batch_data = batch_data.to(device)

            
            try:
                out = model(batch_data)
            except RuntimeError as exception:
                if 'out of memory' in str(exception):
                    print('clearing cache')
                    torch.cuda.empty_cache()
                    out = model(batch_data)
                    

            # assert not nan

            if dataset_name in ['Cora','Citeseer','Pubmed','Cornell','Texas','Wisconsin','CS','Physics','Computers','Photo','ogbn-arxiv','chameleon', 'squirrel']:
                if mode == 'test':
                    loss = F.nll_loss(out[batch_data.test_mask], batch_data.y[batch_data.test_mask].flatten())
                    a = out.argmax(dim=1)[batch_data.test_mask].detach().cpu()
                    b = batch_data.y[batch_data.test_mask].detach().cpu().flatten()
                    correct_batch = ( a==b ).sum()
                    correct += correct_batch
                    card += int(batch_data.test_mask.sum())
                elif mode == 'val':
                    loss = F.nll_loss(out[batch_data.val_mask], batch_data.y[batch_data.val_mask].flatten())
                    a = out.argmax(dim=1)[batch_data.val_mask].detach().cpu()
                    b = batch_data.y[batch_data.val_mask].detach().cpu().flatten()
                    correct_batch = ( a==b ).sum()
                    correct += correct_batch
                    card += int(batch_data.val_mask.sum())
                else:
                    print(" something wrong ")

            else:
                loss = F.nll_loss(out, batch_data.y)
                correct_batch = (out.argmax(dim=1) == batch_data.y.flatten()).sum()
                #acc = int(correct) / int(batch_data.train_mask.sum())
                correct += correct_batch
                # change
                card += batch_data.y.shape[0]

            loss_epoch += loss.item()
        
        acc_epoch = correct/card
        return loss_epoch, acc_epoch



def train_all_epochs(epoch_num, model, dataloaders, optimizer, dataset_name, device, pat = 50):
    best_val = 0
    best_test = 0

    last_loss = math.inf
    cont_decrease = 0
    

    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    

    for epoch in range(epoch_num):
        l_train,acc_train = train_one_epoch(model, train_loader, optimizer, dataset_name, device)
        loss_val, acc_val = test(model, val_loader, dataset_name, 'val', device)
        # debugging
        #print(f'epoch {epoch} accu {acc_train} loss {l_train} lossval{loss_val} accu val {acc_val}')
        # no improvement
        if loss_val>last_loss:
            cont_decrease = cont_decrease+1
            if cont_decrease>=pat:
                print('early stopping at epoch '+ str(epoch))
                return best_test
        else:
            cont_decrease = 0    
        last_loss = loss_val

        if best_val<acc_val:
            best_val = acc_val
            loss_test, acc_test = test(model, test_loader, dataset_name, 'test', device)
            best_test = acc_test
            #print(f'epoch {epoch}  bestval{best_val} accu_test {acc_test} {loss_test} test {best_test}')
            
    return best_test

def get_model_and_opt(model_name, params, dataset, device, edge_dim = None):
    if model_name == 'gcn':
        model = GCN(dataset, params['num_layers'], params['hidden_dim'], params['hidden_dim_post']).to(device)
    if model_name == 'gat':
        model = GAT(dataset, params['num_layers'], params['hidden_dim'], params['hidden_dim_post']).to(device)
    if model_name == 'graphsage':
        model = GraphSAGE(dataset, params['num_layers'], params['hidden_dim'], params['hidden_dim_post']).to(device)
    if model_name == 'gin':
        model = GIN(dataset, params['num_layers'], params['hidden_dim'], params['hidden_dim_post']).to(device)
    if model_name == 'egin':
        model = eGIN(dataset, params['num_layers'], params['hidden_dim'], params['hidden_dim_post'],edge_dim).to(device)
    if model_name == 'egat':
        model = eGAT(dataset, params['num_layers'], params['hidden_dim'], params['hidden_dim_post'],edge_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), params['lr'], weight_decay=5e-4)
    return model,optimizer


def change_params(p, d_name):
    # change p['batch_size']
    if d_name in ['Cora','Citeseer','Pubmed','CS','Physics','ogbn-arxiv','Cornell','Texas','Wisconsin','Computers','Photo','chameleon', 'squirrel']:
        p['batch_size'] = 1
    else:
        p['batch_size'] = 8
    return p

def split_arxiv(d):
    dummy = PygNodePropPredDataset(name='ogbn-arxiv', root='/dum')
    split_idx = dummy.get_idx_split() 
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    d.train_mask = train_idx
    d.val_mask = val_idx
    d.test_mask = test_idx
    return d

def split_webkb(d,seed_num):
    train_m = d.train_mask[:,seed_num]
    val_m = d.val_mask[:,seed_num]
    test_m = d.test_mask[:,seed_num]

    d.train_mask = train_m
    d.val_mask = val_m
    d.test_mask = test_m
    return d
    

def get_dataloaders(p, d_name, pretrans, if_trans = False, trans = None, folder_loc = './drive/MyDrive/datasets1',seed = 0):
    """  # random split standard deviation too large
    if d_name in ['Cornell','Texas','Wisconsin']:
        num_node_dict ={'Cornell':183,'Texas':183,'Wisconsin':251} 
        split_num = int(num_node_dict[d_name]*0.2)
        split_trans = T.RandomNodeSplit(split = 'train_rest', num_val=split_num, num_test=split_num)
        if if_trans:
            composed_trans = T.Compose([trans, split_trans])
            dataset = WebKB(folder_loc, name = d_name, transform = composed_trans, pre_transform=func1)
            dataset.shuffle()  
        else:
            dataset = WebKB(folder_loc, name = d_name, transform = split_trans, pre_transform=func1)
            dataset.shuffle()  

        dataloaders = {}
        dataloaders['train'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=True)
        dataloaders['val'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)
        dataloaders['test'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)
    """
    if d_name in ['Cornell','Texas','Wisconsin']:
        
        split_trans = lambda da:split_webkb(da,seed)
        if if_trans:
            composed_trans = T.Compose([trans, split_trans])
            dataset = WebKB(folder_loc, name = d_name, transform = composed_trans, pre_transform=func1)
            dataset.shuffle()  
        else:
            dataset = WebKB(folder_loc, name = d_name, transform = split_trans, pre_transform=func1)
            dataset.shuffle()  

        dataloaders = {}
        dataloaders['train'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=True)
        dataloaders['val'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)
        dataloaders['test'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)

    if d_name in ['chameleon', 'squirrel']:
        
        split_trans = lambda da:split_webkb(da,seed)
        if if_trans:
            composed_trans = T.Compose([trans, split_trans])
            dataset = WikipediaNetwork(folder_loc, name = d_name, transform = composed_trans, pre_transform=func1)
            dataset.shuffle()  
        else:
            dataset = WikipediaNetwork(folder_loc, name = d_name, transform = split_trans, pre_transform=func1)
            dataset.shuffle()  

        dataloaders = {}
        dataloaders['train'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=True)
        dataloaders['val'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)
        dataloaders['test'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)


    if d_name in ['Computers','Photo']:
        split_trans = T.RandomNodeSplit(split = 'test_rest')
        if if_trans:
            composed_trans = T.Compose([trans, split_trans])
            dataset = Amazon(folder_loc, name = d_name, transform = composed_trans, pre_transform=func1)
            dataset.shuffle()  
        else:
            dataset = Amazon(folder_loc, name = d_name, transform = split_trans, pre_transform=func1)
            dataset.shuffle()  
        dataloaders = {}
        dataloaders['train'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=True)
        dataloaders['val'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)
        dataloaders['test'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)


    if d_name in ['Cora','Citeseer','Pubmed']:
        #if d_name=='Citeseer':
        #    pretrans = T.Compose([T.RemoveIsolatedNodes(),pretrans])
        if if_trans:        
            dataset = Planetoid(folder_loc, name = d_name, transform = trans, pre_transform=pretrans)
        else:      
            dataset = Planetoid(folder_loc, name = d_name, pre_transform=pretrans)
        dataset.shuffle()  
        dataloaders = {}
        dataloaders['train'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=True)
        dataloaders['val'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)
        dataloaders['test'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)
    
    if d_name in ['ogbn-arxiv']:
        if if_trans:
            transforms = T.Compose([T.RemoveIsolatedNodes(),pretrans])
            pose_trans = T.Compose([split_arxiv,trans])
            dataset = PygNodePropPredDataset(name=d_name, root=folder_loc, transform = pose_trans, pre_transform=transforms)
        else:
            transforms = T.Compose([T.RemoveIsolatedNodes(),pretrans])
            dataset = PygNodePropPredDataset(name=d_name, root=folder_loc, transform = split_arxiv, pre_transform=transforms)
        dataset.shuffle() 
        
    
        dataloaders = {}
        dataloaders['train'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=True)
        
        dataloaders['val'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)
        dataloaders['test'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)

    
    if d_name in ['PATTERN','CLUSTER']:
        if if_trans:
            train_dataset = GNNBenchmarkDataset(folder_loc, name = d_name, split = 'train', transform = trans, pre_transform=pretrans)
            val_dataset = GNNBenchmarkDataset(folder_loc, name = d_name, split = 'val', transform = trans, pre_transform=pretrans)
            test_dataset = GNNBenchmarkDataset(folder_loc, name = d_name, split = 'test', transform = trans, pre_transform=pretrans)
        else:
            train_dataset = GNNBenchmarkDataset(folder_loc, name = d_name, split = 'train', pre_transform=pretrans)
            val_dataset = GNNBenchmarkDataset(folder_loc, name = d_name, split = 'val', pre_transform=pretrans)
            test_dataset = GNNBenchmarkDataset(folder_loc, name = d_name, split = 'test', pre_transform=pretrans)
        train_dataset.shuffle()
        val_dataset.shuffle()
        test_dataset.shuffle()
        dataset = train_dataset
        dataloaders = {}
        dataloaders['train'] = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True)
        dataloaders['val'] = DataLoader(val_dataset, batch_size=p['batch_size'], shuffle=False)
        dataloaders['test'] = DataLoader(test_dataset, batch_size=p['batch_size'], shuffle=False)

    
    if d_name in ['CS','Physics']:
        # split datasets add masks according to pitfall paper
        split_trans = T.RandomNodeSplit(split = 'test_rest')
        if if_trans:
            composed_trans = T.Compose([trans, split_trans])
            dataset = Coauthor(folder_loc, name = d_name, transform = composed_trans, pre_transform=func1)
            dataset.shuffle()  

        else:
            dataset = Coauthor(folder_loc, name = d_name, transform = split_trans, pre_transform=func1)
            dataset.shuffle()  


        dataloaders = {}
        dataloaders['train'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=True)
        dataloaders['val'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)
        dataloaders['test'] = DataLoader(dataset, batch_size=p['batch_size'], shuffle=False)
    
    return dataset, dataloaders




def add_edge_attr(data, khop = 2, max_nodes = 50, length = 10):

    

    # first row normalize features .x
    normed_x =  data.x / data.x.sum(dim=-1).unsqueeze(-1)

    print('replacing nan values to 0 in data.x')
    if torch.isnan(normed_x).any():
        normed_x = torch.nan_to_num(normed_x)
    data.x = normed_x

    edge_dim = 12+2*length
    st = data.edge_index[0].tolist()
    en = data.edge_index[1].tolist()
    edge_a = np.zeros((data.num_nodes,data.num_nodes,edge_dim))

    for index in range(len(st)):
        u = st[index]
        v = en[index]

        k_subgraph = k_hop_subgraph([u,v] , khop, data.edge_index, flow = "target_to_source", relabel_nodes = True)

        assert k_subgraph[0].shape[0]>1, "something wrong"
        
        
        # isolated.
        if k_subgraph[0].shape[0]==2:
            #print("isolated node spoted")
            #print(data.contains_isolated_nodes())
            #print(i)
            ua = np.array([-1,1])
            ul = np.array([0,2])
        else:
            if k_subgraph[0].shape[0]>max_nodes:
                # changed to keep two nodes
                pos1 = k_subgraph[2][0].item()
                pos2 = k_subgraph[2][1].item()

                temp_list = k_subgraph[0].tolist()


                cent1 = temp_list[pos1]
                cent2 = temp_list[pos2]
    
                if pos1>pos2:
                    pos1,pos2 = pos2,pos1


                del temp_list[pos2]
                del temp_list[pos1]

                random.shuffle(temp_list)
                new_list = temp_list[:max_nodes-2]+[cent1,cent2]
                new_list.sort()
                new_nodes_tensor =  torch.LongTensor(new_list)
                
                _,old_edge_index,_,_ = k_hop_subgraph([u,v] , khop, data.edge_index, flow = "target_to_source", relabel_nodes = False)
                new_subgraph_edge_index,_ = subgraph(new_nodes_tensor, old_edge_index,relabel_nodes=True)
            else:
                new_subgraph_edge_index = k_subgraph[1]
            
            ua,ul = return_one_subgraph_spectrum(new_subgraph_edge_index)
            ua = np.real_if_close(ua,100)
            ua = np.around(ua,4)
            ul = np.real_if_close(ul,100)
            ul = np.around(ul,4)

        # 2d array with one element
        FA = get_features_A([ua])[0]
        FL = get_features_L([ul])[0]
        DA = get_dist_A([ua], max_nodes,length)[0]
        DL = get_dist_L([ul],length)[0]
        # dim: 5,7,length,length
        edge_a[u,v] = np.concatenate((FA, FL, DA, DL), axis=None)

    print('to be safe replacing nan values to 0 in edge attribute')
    if torch.isnan(edge_a).any():
        edge_a = torch.nan_to_num(edge_a)

    data.edge_attr = edge_a

    return data




def edge_trim_1(data):
    
    full_edge_att = data.edge_attr
    corped_edge_att = full_edge_att[:,:,:12]

    data.edge_attr = corped_edge_att
    return data


def edge_trim_2(data):
    
    full_edge_att = data.edge_attr
    corped_edge_att = full_edge_att[:,:,12:]

    data.edge_attr = corped_edge_att
    return data


