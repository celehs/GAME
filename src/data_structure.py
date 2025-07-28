# pylint: skip-file

"""
Title: data_structure.py
Author: Han Tong
Date: 2025-07-01
Python Version: Python 3.11.3
Description: define the class of each node, the dataset structures, and the initialize function of them
"""

import numpy as np
from torch.utils.data.sampler import Sampler
import warnings
import torch
from config import get_config
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import  *


class item_node:
    '''
    Six sets are useful for every code: 
    1. same_par: siblings [across inst. or within inst. or none]
    2. same_gra: cousins + siblings [across inst. or within inst. or none]
    3. P_local: positive loinc code of other lab or local lab [across inst. or none]
    4. N_local: negative loinc code of other lab or local lab [across inst. or none]
    5. rel: related codes [within inst. or none]
    6. sim_No_hie: similar_no_hie_pairs [within inst. or none]
    '''
    def __init__(self, name=""):
        self.name = name
        self.same_par = set()    # siblings
        self.same_gra = set()    # siblings and cousins    
        self.P_local = set()     # for local code
        self.N_local = set()
        self.rel = set()  
        self.sim_no_hie = set()
        

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MySampler(Sampler):
    
    def __init__(self, unique_names):
        self.unique_names = unique_names

    def __iter__(self):
        '''
        sample unique names and return corresponding index of them
        this method makes sense because we want the same code have more consistent results 
        '''
        config = get_config()
        unique_names = self.unique_names
        unique_name_batch_index = np.random.choice(range(config['num_union']), size=config['batch_size'], replace=False)
        
        if str(config['latent']).lower() != 'false':
            indices_new = [i for i, name in enumerate(pd.Series(name_new, name='V1')) if name_new is not None and name in unique_name_batch]
        self.indices = indices
        if str(config['latent']).lower() != 'false':
            self.indices_new = indices_new
            return indices, indices_new
        else:
            return indices
    
    def __len__(self):
        return int(np.ceil(len(self.unique_names) / get_config()['batch_size']))


def origin_loss_set(unique_name, pos_LTOL, neg_LTOL, hie_loinc_rxn_phe, 
                Train_REL_pairs, Train_sim_no_hie_pairs, rel_index, sim_no_hie_index):
    '''
    create a term for each node, including name, one_one, same_par, same_gra, rel
    '''
    my_objects = []
    config = get_config()

    for i in tqdm(range(len(unique_name))):
        obj = item_node(name=unique_name[i])

        if grepl('^LOCAL\\|LAB:|^Other lab:|^LOCAL\\|PX:|^CCAM:', obj.name):
            obj.P_local = find_other_local(obj.name, pos_LTOL, unique_name)
            obj.N_local = find_other_local(obj.name, neg_LTOL, unique_name)

        if grepl('^PheCode:|^LOINC:|^CCAM:|^RXNORM:', obj.name):
            obj.same_par = find_same_par_gra(hie_loinc_rxn_phe, unique_name, name_id = i, PARENT=1)
            obj.same_gra = find_same_par_gra(hie_loinc_rxn_phe, unique_name, name_id = i, PARENT=2)

        if i in rel_index:
            obj.rel = find_rel(i, Train_REL_pairs, unique_name)
            
        if i in sim_no_hie_index:
            obj.sim_no_hie = find_rel(i, Train_sim_no_hie_pairs, unique_name)
        
        if i in pos_sppmi_index:
            obj.pos_ppmi = pos_sppmi[1][pos_sppmi[0]==i]
            obj.neg_ppmi = neg_sppmi[1][neg_sppmi[0]==i]
        
        my_objects.append(obj)

    return my_objects



class item_node_temp:
    '''
    2 sets are useful for every code: 
    1. sampled set1: Positive set
    2. sampled set2: Negative set
    '''
    def __init__(self, name=""):
        self.name = name
        self.sampled_set1 = set()
        self.sampled_set2 = set()

        
def origin_term_temp(name_temp, set1, set2, DIFF=True, max_len=10):
    '''
    create a term for each node temp, including name, sampled_set1, sampled_set2
    '''
    my_objects = []

    for i in range(len(name_temp)):
        obj =item_node_temp(name = name_temp[i])
        obj.sampled_set1 = sample(set1[i], max_len)
        obj.sampled_set2 = nega_sample(set(set1[i]), set(set2[i]), max_len, DIFF)
        my_objects.append(obj)

    return my_objects   


class CustomExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr):
        self.min_lr = min_lr
        super(CustomExponentialLR, self).__init__(optimizer, gamma)

    def get_lr(self):
        lrs = []
        for group in self.optimizer.param_groups:
            lr = group['lr'] * self.gamma ** self.last_epoch
            lr = max(lr, self.min_lr)
            lrs.append(lr)
        return lrs