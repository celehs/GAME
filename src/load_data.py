# pylint: skip-file

"""
Title: load_data.py
Author: Han Tong
Date: 2025-07-01
Python Version: Python 3.11.3
Description: Load all data we need in this file

"""
import numpy as np
import pandas as pd
import pickle
import warnings
import torch
import re
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import set_config, get_config
config = get_config()
os.chdir(f'{config["path"]}/src')

warnings.filterwarnings('ignore')
import torch_geometric as tg
from torch.utils.data import DataLoader
from utils import *
from data_structure import *

# load the inst row index
data = np.load(f'{config["input_dir"]}/name_desc/inst_row.npz')
keys = data.files  # This will give you a list of all keys in the .npz file
config['inst_row'] = [data[key] for key in keys]

# the original embedding we need
sppmi_list = torch.load(f'{config["input_dir"]}/emb/inst_emb.pth')
# sppmi_list = torch.load(f'{config["input_dir"]}/emb/inst_emb_01.pth')
sap_emb = torch.load(f'{config["input_dir"]}/emb/sap_emb.pth')
coder_emb = torch.load(f'{config["input_dir"]}/emb/coder_emb.pth')
bge_emb = torch.load(f'{config["input_dir"]}/emb/bge_emb.pth')
openai_emb = torch.load(f'{config["input_dir"]}/emb/openai_emb.pth')

# the name of latent nodes
unique_name = pd.read_csv(f'{config["input_dir"]}/name_desc/unique_name_desc.csv')
unique_name = unique_name.iloc[:,0].values

# the hierarchy of loinc, rxnorm and phecode we need.
hie_loinc_rxn_phe = pd.read_csv(f'{config["input_dir"]}/Hierarchy/hie_train_0806.csv')

# Local code map to codified (gpt4o)
P_LTOL = pd.read_csv(f'{config["input_dir"]}/Local_Code_Mapping/LOCAL_pos_0823.csv')
N_LTOL = pd.read_csv(f'{config["input_dir"]}/Local_Code_Mapping/LOCAL_neg_0823.csv')

# Related pairs we have
REL_pairs = pd.read_csv(f'{config["input_dir"]}/similar_related_pairs/related_pairs_0806.csv')

# no-hie similar pairs we have
SIM_no_hie_pairs = pd.read_csv(f'{config["input_dir"]}/similar_related_pairs/similar_nohie_pairs_0806.csv')

# train hie-similar pairs we have
test_sim_pairs = pd.read_csv(f'{config["input_dir"]}/similar_related_pairs/similar_pairs_hie_test_0806.csv')

# # initialize the train, validation and test set of related pairs
# train_rel_pairs, val_rel_pairs, test_rel_pairs = split_train_set(unique_name, REL_pairs=REL_pairs, scale=[0.7,0.3])
# test_rel_pairs = pd.concat([val_rel_pairs, test_rel_pairs])

# with open(f"{config['input_dir']}/similar_related_pairs/rel_pairs_0806.pkl", 'wb') as f:
#     pickle.dump([train_rel_pairs, test_rel_pairs], f)
# rel_edges = np.row_stack([match(train_rel_pairs.iloc[:,0].values, unique_name), match(train_rel_pairs.iloc[:,1].values, unique_name)])
# np.save(f"{config['input_dir']}/edges/edges_rel.npy", rel_edges)

with open(f"{config['input_dir']}/similar_related_pairs/rel_pairs_0806.pkl", 'rb') as f:
    train_rel_pairs, test_rel_pairs = pickle.load(f)

# # initialize the train, validation and test set of similar no hie pairs
# train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs = split_train_set(unique_name, REL_pairs=SIM_no_hie_pairs, scale=[0.7,0.3])
# test_sim_no_hie_pairs = pd.concat([val_sim_no_hie_pairs, test_sim_no_hie_pairs])
# with open(f"{config['input_dir']}/similar_related_pairs/sim_no_hie_pairs_0806.pkl", 'wb') as f:
#     pickle.dump([train_sim_no_hie_pairs, test_sim_no_hie_pairs], f)
# sim_edges = np.row_stack([match(train_sim_no_hie_pairs.iloc[:,0].values, unique_name), match(train_sim_no_hie_pairs.iloc[:,1].values, unique_name)])
# np.save(f"{config['input_dir']}/edges/edges_sim_no_hie.npy", sim_edges)

train_sim_no_hie_pairs, test_sim_no_hie_pairs = np.load(f"{config['input_dir']}/similar_related_pairs/sim_no_hie_pairs_0806.pkl", allow_pickle=True)

# edges = torch.tensor(np.load(f"{config['input_dir']}/edges/edges.npy", allow_pickle=True))
edges_map = torch.tensor(np.load(f"{config['input_dir']}/edges/edges_map.npy", allow_pickle=True))
edges_hie = torch.tensor(np.load(f"{config['input_dir']}/edges/edges_hie.npy", allow_pickle=True))
same_desc_edge = torch.tensor(np.load(f"{config['input_dir']}/edges/edges_same_desc.npy", allow_pickle=True))

edges_rel = torch.tensor(np.load(f"{config['input_dir']}/edges/edges_rel.npy", allow_pickle=True))
edges_sim = torch.tensor(np.load(f"{config['input_dir']}/edges/edges_sim_no_hie.npy", allow_pickle=True))

pos_sppmi = torch.tensor(np.load(f"{config['input_dir']}/edges/pos_sppmi_0730.npy", allow_pickle=True))
neg_sppmi = torch.tensor(np.load(f"{config['input_dir']}/edges/neg_sppmi_0729.npy", allow_pickle=True))

all_pos_sppmi = pos_sppmi
ALL_sim_val_pairs =  pd.concat([test_sim_no_hie_pairs, test_sim_pairs], ignore_index=True)

rel_index = get_index(train_rel_pairs, unique_name)
sim_no_hie_index = get_index(train_sim_no_hie_pairs, unique_name)

# # generate my_objects using hie_train
# # Origin_term can take 10 mins. We can load my_objects that have been generated
# my_objects_new = origin_loss_set(unique_name, P_LTOL, N_LTOL, hie_loinc_rxn_phe, 
#                           train_rel_pairs, train_sim_no_hie_pairs, pos_sppmi, neg_sppmi)

# np.save(f"{config['input_dir']}/edges/my_objects_0527.npy", my_objects_new)
my_objects = np.load(f"{config['input_dir']}/edges/my_objects_0527.npy", allow_pickle=True)