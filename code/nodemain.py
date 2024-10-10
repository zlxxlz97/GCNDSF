import torch
from utils import *
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5120"
#torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

print(f'using device {device}')

seeds = [0,1,2,3,4,5,6,7,8,9]
#seeds = [0,1,2]

num_hops = [1]
#dataset_names= ['Cora','Citeseer','Pubmed','Cornell','Texas','Wisconsin','CS','Physics','Computers','Photo','PATTERN','CLUSTER','ogbn-arxiv']
#dataset_names= ['CS','Physics','Computers','Photo','PATTERN','CLUSTER','ogbn-arxiv']
#dataset_names= ['PATTERN','CLUSTER']
#dataset_names = ['ogbn-arxiv']
#dataset_names = ['Citeseer']
#dataset_names = ['Cora']
#dataset_names = ['Cora','Pubmed']
dataset_names= ['Cornell','Texas','Wisconsin','chameleon', 'squirrel']
model_name_list = ['gin','gcn','graphsage','gat']


params = {}
params['num_layers'] = 2
params['hidden_dim'] = 64
params['hidden_dim_post'] = 32
params['lr'] = 0.01

folder_loc1 = 'pre1'
folder_loc2 = 'pre2'
folder_loc3 = 'pre3'
folder_loc4 = 'pre4'

import sys
#sys.setrecursionlimit(1500)
for dataset_name in dataset_names:
    for model_name in model_name_list:
        print('-'*20)
        print(f"Running for dataset: {dataset_name} and model {model_name}.")
        results_table = np.zeros((len(seeds),16))

        for seed_num_ind in range(len(seeds)):
            seed_num = seeds[seed_num_ind]
            set_all_seed(seed_num,device)
            print(f'running for seed {seed_num}')
                    
            # change params 
            params = change_params(params, dataset_name)

            train_d, dataloaders = get_dataloaders(params, dataset_name, func1, False, folder_loc = folder_loc1, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,0] = test_result
            
            train_d, dataloaders = get_dataloaders(params, dataset_name, func1,True, f11, folder_loc = folder_loc1, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,1] = test_result

            train_d, dataloaders = get_dataloaders(params, dataset_name, func1,True, f12, folder_loc = folder_loc1, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,2] = test_result

            train_d, dataloaders = get_dataloaders(params, dataset_name, func1,True, f13, folder_loc = folder_loc1, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,3] = test_result

            ##########################


            train_d, dataloaders = get_dataloaders(params, dataset_name, func2, False, folder_loc = folder_loc2, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,4] = test_result
            
            train_d, dataloaders = get_dataloaders(params, dataset_name, func2,True, f11, folder_loc = folder_loc2, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,5] = test_result

            train_d, dataloaders = get_dataloaders(params, dataset_name, func2,True, f12, folder_loc = folder_loc2, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,6] = test_result

            train_d, dataloaders = get_dataloaders(params, dataset_name, func2,True, f13, folder_loc = folder_loc2, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,7] = test_result


            ##########################


            train_d, dataloaders = get_dataloaders(params, dataset_name, func3, False, folder_loc = folder_loc3, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,8] = test_result
            
            train_d, dataloaders = get_dataloaders(params, dataset_name, func3,True, f21, folder_loc = folder_loc3, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,9] = test_result

            train_d, dataloaders = get_dataloaders(params, dataset_name, func3,True, f22, folder_loc = folder_loc3, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,10] = test_result

            train_d, dataloaders = get_dataloaders(params, dataset_name, func3,True, f23, folder_loc = folder_loc3, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,11] = test_result

            ##########################


            train_d, dataloaders = get_dataloaders(params, dataset_name, func4, False, folder_loc = folder_loc4, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,12] = test_result
            
            train_d, dataloaders = get_dataloaders(params, dataset_name, func4,True, f21, folder_loc = folder_loc4, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,13] = test_result

            train_d, dataloaders = get_dataloaders(params, dataset_name, func4,True, f22, folder_loc = folder_loc4, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,14] = test_result

            train_d, dataloaders = get_dataloaders(params, dataset_name, func4,True, f23, folder_loc = folder_loc4, seed = seed_num)
            model,optimizer = get_model_and_opt(model_name, params, train_d,device)
            test_result = train_all_epochs(400,model,dataloaders,optimizer, dataset_name, device)
            results_table[seed_num_ind,15] = test_result

        print(results_table)
        averaged_result = np.mean(results_table,axis=0)
        exp_std = np.std(results_table,axis=0)
        print(averaged_result)
        print(exp_std)