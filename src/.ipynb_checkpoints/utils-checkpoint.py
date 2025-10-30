# pylint: skip-file

""""
Title: Utils.py
Author: Han Tong
Date: 2025-10-08
Python Version: Python 3.11.3
Description: All useful functions we use
"""

import logging
import re
from collections import OrderedDict
from itertools import islice
from config import get_config
from scipy.stats import spearmanr
import os
import numpy as np
import pandas as pd
import itertools
import torch
import csv
import warnings
from scipy.linalg import svd
import torch.nn.functional as F
warnings.filterwarnings("ignore")
import time
import logging
import random

from openai import OpenAI 
from torch_geometric.utils import to_undirected
from torch_geometric.utils import add_self_loops
from scipy.stats import spearmanr
import itertools
import io
import sys


config = get_config()
    
def logging_config(config, start_time,
                   level=logging.INFO,
                   console_level=logging.INFO,
                   no_console=True):
    folder = f"{config['path']}/output/{start_time}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, start_time + ".log")
    print("All logs will be saved to %s" %logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def split_into_batches(lst, batch_size):
    '''
    split my_objects into batches
    '''
    random.shuffle(lst)
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def get_values(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.values
    return data


def grep_index(pattern, name_all):
    """
    find pattern in name_all
    name_all can be a string or strings
    """
    if isinstance(name_all, str):
        return [0] if re.match(pattern, name_all) else []
    else:
        return np.where([re.match(pattern, name) for name in name_all])[0]

def grepl(pattern, name_all):
    """
    whether can we find pattern in name_all or not
    name_all is a string
    """
    return bool(re.match(pattern, name_all))



def id_map(codes, name_all):
    """
    Fast mapping of codes to indices in name_all.
    Missing codes → -1
    """
    codes = np.asarray(codes)
    name_all = np.asarray(name_all)

    # fast map with pandas
    mapping = pd.Series(np.arange(len(name_all)), index=name_all)
    mapped = mapping.reindex(codes, copy=False).to_numpy()

    # replace NaN with -1
    mapped = np.where(np.isnan(mapped), -1, mapped).astype(int)
    return mapped
    
        
def match(a, b, rm_None=True):
    # Create an array to store the indices
    indices = np.array([np.where(b == x)[0][0] if x in b else np.nan for x in a])
    # Filter out None values, which represent elements not found in b
    indices = indices[~np.isnan(indices)]
    return indices


def unique_slice(s, num_codes=20):
    """
    return the first num_codes unique codes
    """
    return list(islice(OrderedDict.fromkeys(s), num_codes))
    

def ret_type(codes, type_list):
    """
    codes should be a list []
    """
    return(type_list.iloc[id_map(codes, type_list.iloc[:,0].values),2].values)



def find_one_one(code, now_index, name_all):
    """
    help codes find the index of one-one mapping codes across institutions except for leaf LOINC codes
    """
    return np.setdiff1d(id_map([code], name_all), now_index)



def find_same_par_gra(hie_loinc_rxn_phe, name_all, name_id, PARENT=1):
    """
    help LOINC, RXNORM, PheCode find their siblings
    hie_loinc_rxn_phe has 3 cols. self, parent, grandpa
    if PARENT = 1, find siblings; is PARENT = 2, find  cousins and siblings
    """

    
    name = name_all[name_id]
    parent_values = hie_loinc_rxn_phe.iloc[:, PARENT].values
    
    name_map = id_map([name], hie_loinc_rxn_phe.iloc[:, 0].values)
    
    if len(name_map)==0:
        return []

    now_id = name_map[0]
    sib_me_id_in_hie = np.where(parent_values == parent_values[now_id])[0]
    
    sib_id_in_hie = list(set(sib_me_id_in_hie) - {now_id})
    sib_names = hie_loinc_rxn_phe.iloc[sib_id_in_hie, 0].values
    
    sib_id_in_name = id_map(sib_names, name_all)
   
    return list(set(sib_id_in_name))

    
    
def sample(codes, max_len = 10):
    """
    sample max_len codes from codes
    """
    if len(codes) < max_len:
        return codes
    else:
        if isinstance(codes, np.ndarray):
            codes = codes.tolist()
        elif isinstance(codes, (set, dict)):
            codes = sorted(codes)
        return codes
    

def sample_cols(data, p):
    """
    sample p fraction columns from data
    """
    num_columns = data.shape[1]
    sampled_columns = np.random.choice(num_columns, int(p * num_columns), replace=False)
    sampled_data = data[:, sampled_columns]
    return sampled_data
    
    
def nega_sample(set1, set2, max_len = 10, DIFF=True):
    """
    Negative sampling function, you can negative sample set1 from set2\set1(if DIFF)
    """
    num_samples = len(set1)
    if DIFF:
        sample_range = set2 - set1
    else:
        sample_range = set2
    return set(np.random.choice(list(sample_range), min(num_samples, len(sample_range), max_len), replace=False))

    
def TABLE_TO_INDEX(TABLE, name_all, all_col=False):
    """
    Map values in TABLE to their indices in name_all using id_map.

    Parameters
   """ 
    if not isinstance(TABLE, pd.DataFrame):
        TABLE = pd.DataFrame(TABLE)

    if all_col:
        mapped = TABLE.apply(lambda col: id_map(col.values, name_all))
        return mapped.to_numpy()
    else:
        mapped = np.row_stack([
            id_map(TABLE.iloc[:, 0].values, name_all),
            id_map(TABLE.iloc[:, 1].values, name_all)
        ])
        return mapped

        
def find_rel(name_id, TABLE):
    """
    find related pairs of code in REL_pairs table
    """
    related_index = np.unique(np.union1d(TABLE[1][TABLE[0]==name_id], TABLE[0][TABLE[1]==name_id]))
    return related_index

    

def find_same_type(name_id, type_list):
    """
    find codes from same type
    """
    type_list = type_list[['type']].values
    type_now = type_list[name_id] 
    index = np.where(type_list==type_now)[0]
    return index

    
def unique_func(row, size=20):
    """
    return unique codes in row
    keep the order of codes
    """
    unique_elements = pd.unique(row)
    unique_elements = unique_elements[:size]
    return unique_elements


def now_time():
    """
    return time now
    """
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    return formatted_time



def save_hyperparameters(config, start_time):
    """
    save hyperparameters in config
    """
    with open(f"{config['path']}/output/{start_time}/hyper_par.csv", "a", newline="") as csvfile:
        fieldnames = list(config.keys())
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerow(config.values())



def write_file_sub(MGB_AUC1, name, start_time, config):
    """
    Write header (column names) for AUC files.
    For the first write only.
    """
    save_path = f"{config['path']}/output/{start_time}/{name}.csv"

    # Only write header if file does not exist yet
    if not os.path.exists(save_path):
        row_names = MGB_AUC1.index
        header = ["EPOCH", "BATCH"]
        for rn in row_names:
            header.extend([rn, "num"])
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def write_file_sub2(MGB_AUC1, name, Epoch, Batch, start_time, config):
    """
    Write AUC values (no column names, just data rows).
    """
    save_path = f"{config['path']}/output/{start_time}/{name}.csv"

    # Prepare data in same order as header
    row_names = MGB_AUC1.index
    values = [Epoch, Batch]
    for rn in row_names:
        values.extend([MGB_AUC1.loc[rn, "auc"], MGB_AUC1.loc[rn, "num"]])

    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(values)


def write_file(Epoch, Batch, config, start_time, loss=None, pre=None, 
               SIM_AUC=None, SIM_AUC_CUI=None, REL_AUC=None, REL_AUC_CUI=None):
    """
    Write all output files (loss, pre, and AUCs).
    """
    output_dir = f"{config['path']}/output/{start_time}"
    begin = not os.path.exists(output_dir)
    if begin:
        os.mkdir(output_dir)

        # LOSS header
        lossfile = open(f"{output_dir}/ALL_LOSS.csv", "w", newline="")
        losswriter = csv.writer(lossfile)
        if config.get('path_origin') == 'align_NA':
            losswriter.writerow(["EPOCH", "BATCH", "align_loss"])
        elif config.get('path_origin') is None:
            losswriter.writerow(["EPOCH", "BATCH", 
                                 "P_LOSS_hie", "N_LOSS_hie", 
                                 "P_LOSS_OTOL", "N_LOSS_OTOL", 
                                 "P_LOSS_SIM_NO_HIE", "N_LOSS_SIM_NO_HIE"])
        else:
            losswriter.writerow(["EPOCH", "BATCH", "P_REL", "N_REL", "P_sppmi", "N_sppmi"])
        lossfile.close()

        # PRE header
        if pre is not None:
            with open(f"{output_dir}/ALL_PRE.csv", "w", newline="") as prefile:
                prewriter = csv.writer(prefile)
                prewriter.writerow(["EPOCH", "BATCH", "TOP1", "TOP5", "TOP10", "TOP20"])

        # AUC headers
        if REL_AUC is not None:
            write_file_sub(REL_AUC, "REL_AUC", start_time, config)
        if SIM_AUC is not None:
            write_file_sub(SIM_AUC, "SIM_AUC", start_time, config)
        if REL_AUC_CUI is not None:
            write_file_sub(REL_AUC_CUI, "REL_AUC_CUI", start_time, config)
        if SIM_AUC_CUI is not None:
            write_file_sub(SIM_AUC_CUI, "SIM_AUC_CUI", start_time, config)

        # save hyperparameters
        save_hyperparameters(config, start_time)

    # LOSS append
    if loss is not None:
        with open(f"{output_dir}/ALL_LOSS.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([Epoch, Batch] + list(loss))

    # PRE append
    if pre is not None:
        with open(f"{output_dir}/ALL_PRE.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([Epoch, Batch] + list(pre))

    # AUC append
    if REL_AUC is not None:
        write_file_sub2(REL_AUC, "REL_AUC", Epoch, Batch, start_time, config)
    if SIM_AUC is not None:
        write_file_sub2(SIM_AUC, "SIM_AUC", Epoch, Batch, start_time, config)
    if REL_AUC_CUI is not None:
        write_file_sub2(REL_AUC_CUI, "REL_AUC_CUI", Epoch, Batch, start_time, config)
    if SIM_AUC_CUI is not None:
        write_file_sub2(SIM_AUC_CUI, "SIM_AUC_CUI", Epoch, Batch, start_time, config)
        

def mask_inst(pairs, dict_MGB, shift):
    
    mask = pairs["code1"].isin(dict_MGB) & pairs["code2"].isin(dict_MGB)
    pairs_MGB = pairs[mask]
    pairs_MGB.index = range(pairs_MGB.shape[0])
    temp = np.array(id_map(pairs_MGB["code1"], dict_MGB))
    temp2 = np.array(id_map(pairs_MGB["code2"], dict_MGB))
    pairs_MGB["Var1"] = temp + shift
    pairs_MGB["Var2"] = temp2 + shift
    return(pairs_MGB)

def split_train_set(unique_name, REL_pairs, scale=[0.5, 0.3], seed=42):
    # Set a seed for reproducibility
    random_state = seed
    
    # split train pairs
    num_rows = int(scale[0] * len(REL_pairs)) # add scale ratio of pairs into adj matrix (as train set)
    train_pairs = REL_pairs.sample(n=num_rows, random_state=random_state)
    remaining = REL_pairs[~REL_pairs.index.isin(train_pairs.index)]

    # Split remaining data into validation and test sets
    num_validation = int(scale[1] * len(REL_pairs))
    val_pairs = remaining.sample(n=num_validation, random_state=random_state)
    test_pairs = remaining[~remaining.index.isin(val_pairs.index)]
    
    return train_pairs, val_pairs, test_pairs
    

def remove_duplicate_edge(edge_index):
    edge_index = edge_index.t()
    edge_index =torch.sort(edge_index, dim=1)[0]
    unique_edge_index, inverse_indices = torch.unique(edge_index, return_inverse=True, dim=0)
    unique_edge_index = unique_edge_index.t()
    return unique_edge_index


def get_index(train_rel_pairs, name_all):
    # Convert to set for faster membership checking
    code1_set = set(train_rel_pairs["code1"].tolist())
    code2_set = set(train_rel_pairs["code2"].tolist())

    # Ensure name_all is a list for proper membership checking
    if isinstance(name_all, pd.Series):
        name_all = name_all.tolist()
    elif isinstance(name_all, np.ndarray):
        name_all = name_all.tolist()
    # else assume it"s already a list

    rel_index_1 = np.where([name in code1_set for name in name_all])[0]
    rel_index_2 = np.where([name in code2_set for name in name_all])[0]
    rel_index = np.unique(np.union1d(rel_index_1, rel_index_2))

    return rel_index

def process_predictions(Pre, LEVEL, other_name, item_dict):
    n = len(Pre)
    right_top1 = np.zeros(n)
    right_top5 = np.zeros(n)
    right_top10 = np.zeros(n)
    right_top20 = np.zeros(n)
    ans = 0
    for i in range(n):
        predict_top1 = Pre[i, 0]
        predict_top5 = Pre[i, 0:5]
        predict_top10 = Pre[i, :10]
        predict_top20 = Pre[i, :]
        
        ans += 1
        level_data = item_dict[other_name[i]]

        right_top1[i] = int(predict_top1 in level_data)
        right_top5[i] = int(any(np.isin(predict_top5, level_data)))
        right_top10[i] = int(any(np.isin(predict_top10, level_data)))
        right_top20[i] = int(any(np.isin(predict_top20, level_data)))

    return right_top1, right_top5, right_top10, right_top20, ans

def solve_Procrustes(X1, X2):
    """
    \Omega = argmin ||X1 - X2\Omega||_F
    X1: local code
    X2: LOINC code sapbert emb

    In this function, we can get the best \Omega, so that we can get the new Loinc and the new TOP20
    """
    temp = X2.T @ X1
    u, s, vt = svd(temp)
    return u @ vt


        
def my_item(x):
    if x is None:
        return None
    elif not isinstance(x, torch.Tensor):
         x = torch.tensor(x)
    return x.item()


def weight_auc(RELA_MGB_AUC):
    return sum(RELA_MGB_AUC[0].num.values[~np.isnan(RELA_MGB_AUC[0].auc.values)] * RELA_MGB_AUC[0].auc.values[~np.isnan(RELA_MGB_AUC[0].auc.values)])/sum(RELA_MGB_AUC[0].num.values[~np.isnan(RELA_MGB_AUC[0].auc.values)])


def func_type(RELA_MGB_AUC):
    return RELA_MGB_AUC[0].iloc[~np.isnan(RELA_MGB_AUC[0].auc.values),]


def sum_(SIMI_MGB):
    return(sum(func_type(SIMI_MGB).iloc[:,0].values *  func_type(SIMI_MGB).iloc[:,1].values)/sum(func_type(SIMI_MGB).iloc[:,1].values))



def my_diff(cos1, true_rank):
    ans = [np.mean(compute_spearman(cos1[i,:], true_rank)) for i in range(cos1.shape[0])]
    return ans



def combine_2_array(array1, array2):

    array1 = torch.tensor(array1)
    array2 = torch.tensor(array2)

    # Repeat each element of array1 len(array2) times
    array1_rep = array1.repeat_interleave(len(array2))

    # Repeat array2 len(array1) times
    array2_rep = array2.repeat(len(array1))

    # Combine the arrays
    combined = torch.stack((array1_rep, array2_rep), dim=1)
    return combined



def compute_spearman(a, b):
    # Check if a and b are tensors
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()

    coef, p = spearmanr(a, b)
    return coef, p



def feature_selection_every_epoch(emb_all, loc, epoch, name_list = ['SapBERT','CODER','BGE','OPENAI', 'MGB SPPMI','VA SPPMI','UPMC SPPMI','BCH SPPMI','Duke SPPMI', 'MIMIC SPPMI', 'Bor SPPMI', 'GAME'], code_list = ["PheCode:296.2", "PheCode:290.11", "PheCode:250.1", "PheCode:250.2", "PheCode:555.1", "PheCode:555.2", "PheCode:428.1", 'PheCode:714.1'], api_key=None, config=None, NEG_INST=None):
    # load data
    emb_all = [pd.DataFrame(emb.cpu().numpy()) for emb in emb_all]
    name_desc = pd.read_csv(f'{config["input_dir"]}/name_desc/unique_name_desc_LP.csv')
    name_all = name_desc.iloc[:,0]
    unique_code = pd.DataFrame(name_all).drop_duplicates().values
    unique_code = np.concatenate(unique_code)
    unique_desc = name_desc.drop_duplicates(subset=name_desc.columns[0]).iloc[:,1]
    unique_desc = unique_desc.values

    name_desc = pd.DataFrame(np.column_stack([unique_code, unique_desc]))
    name_desc.columns = ['sign_id', 'desc']

    # do not contain the replicated desc in same type; for LOINC:LP and their childs, only use once
    name_desc['type_desc'] = pd.Series(ret_type(name_desc['sign_id'])) + ':' + name_desc['desc']
    unique_combinations = name_desc[['type_desc']].drop_duplicates()
    mapping_list = pd.DataFrame(columns=['name', 'indices'])

    # Populate the mapping list
    for _, row in unique_combinations.iterrows():
        type_desc = row['type_desc']
        indices = name_desc[name_desc['type_desc'] == type_desc].index.tolist()
        new_row = pd.DataFrame({'name': [type_desc], 'indices': [indices]})
        mapping_list = pd.concat([mapping_list, new_row], ignore_index=True)

    corr_list = []
    result_list = []
    for code in code_list:
        add_param = f'{loc}'  # Dynamically change the add parameter
        result, corr = all_fea_select(emb_all, code, name_desc, add=add_param, 
                                 name_list = name_list,
                                feat_max=100, mapping_list=mapping_list, api_key=api_key, NEG_INST=NEG_INST)
        corr_list.append(corr)
        result_list.append(result)
    
    now_corr = np.mean(corr_list) 
    return now_corr


def feature_selection(emb_all, code, name_all):
    cos_sims = []
    code_indice = np.where(name_all == code)[0]
    emb_all_tensor = torch.tensor(emb_all.values.astype(np.float32))
    
    emb_code_row = emb_all.iloc[code_indice, :].values
    emb_code_tensor = torch.tensor(emb_code_row.astype(np.float32))
    cos_sim = torch.matmul(emb_all_tensor, emb_code_tensor.t()) / torch.norm(emb_code_tensor)**2
    return cos_sim


def all_fea_select(
    emb_all, code, name_desc, name_list, feat_max=100, add=None, 
    negative=None, edges=None, neg=None, mapping_list=None, api_key=None, 
    seed=42, NEG_INST=None
):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    desc_code = name_desc.iloc[np.where(name_desc.iloc[:, 0].values == code)[0], 1].values
    name_all = name_desc.iloc[:, 0].values
    code_indice = np.where(name_all == code)[0]

    if edges is not None:
        index_edges = np.union1d(
            name_all[edges[1][np.where(pd.Series(edges[0]).isin(code_indice))[0]]],
            name_all[edges[0][np.where(pd.Series(edges[1]).isin(code_indice))[0]]]
        )
        index_edges_series = pd.Series(index_edges)

    if neg is not None:
        index_neg = np.union1d(
            name_all[neg[1][np.where(pd.Series(neg[0]).isin(code_indice))[0]]],
            name_all[neg[0][np.where(pd.Series(neg[1]).isin(code_indice))[0]]]
        )
        index_neg_series = pd.Series(index_neg)

    index_all = []
    cos_all = []
    neg_index_all = []  # <- store 100 negative indices per method

    for emb in emb_all:
        cos = feature_selection(emb, code, name_all)
        cos = cos.reshape(-1)
        cos_all.append(cos)

        if mapping_list is not None:
            cos_list = []
            ind_all_list = []
            for mapping in mapping_list['indices'].values:
                tmp = [cos[i] for i in mapping]
                max_value = max(tmp)
                max_index = np.argmax(tmp)
                cos_list.append(max_value)
                ind_all_list.append(mapping[max_index])
            cos = cos_list

        top_values, top_indices = torch.topk(torch.tensor(cos), k=feat_max, largest=True, sorted=True)

        if mapping_list is not None:
            top_indices = [ind_all_list[i] for i in top_indices.numpy()]
        else:
            top_indices = top_indices.numpy().tolist()

        index_all.append(top_indices)

        # ---- Negative Sampling (random 100) ----
        total_index = list(range(len(name_all)))
        unselected = list(set(total_index) - set(top_indices))
        if NEG_INST is not None:
            unselected = list(np.intersect1d(unselected, config['inst_row'][NEG_INST]))
        neg_indices = random.sample(unselected, 100) if len(unselected) >= 100 else unselected
        neg_index_all.append(neg_indices)

    # ---- Build final DataFrame ----
    pos_index = np.concatenate(index_all)

    # ---- Shared 100 Negatives ----
    total_index = list(range(len(name_all)))
    all_pos_indices = set(pos_index)
    unselected = list(set(total_index) - all_pos_indices)
    neg_index = random.sample(unselected, 100) if len(unselected) >= 100 else unselected

    all_index = np.concatenate([pos_index, neg_index])

    unique_desc = name_desc.iloc[:, 1]
    data = {
        'index': all_index,
        'name': name_all[all_index],
        'desc': unique_desc.iloc[all_index].values,
    }

    # Scores
    for i, (name, cos_sim) in enumerate(zip(name_list, cos_all)):
        data[f'{name}_cos'] = [cos_sim[idx].item() for idx in all_index]

    data_df = pd.DataFrame(data)
    data_df['selected_by'] = ''

    # Positives
    for i, name in enumerate(name_list):
        pos_inds = index_all[i]
        data_df.loc[data_df['index'].isin(pos_inds), 'selected_by'] += f'{name}&'

    # Negatives
    data_df.loc[data_df['index'].isin(neg_index), 'selected_by'] += 'NEG&'

    if edges is not None:
        data_df['GPT3.5_TRUE'] = pd.Series(name_all[data_df['index']]).isin(index_edges_series)
    if neg is not None:
        data_df['GPT3.5_FALSE'] = pd.Series(name_all[data_df['index']]).isin(index_neg_series)

    data_df['selected_by'] = data_df['selected_by'].str.rstrip('&')

    # === GPT4 evaluation ===
    if os.path.exists(f"{config['path']}/supp_code/feature_selection/input/GPT4_{code.replace(':', '_')}.csv"):
        score_ = pd.read_csv(f"{config['path']}/supp_code/feature_selection/input/GPT4_{code.replace(':', '_')}.csv")
    else:
        score_ = None

    pos = data_df.drop_duplicates(subset=data_df.columns[0]).iloc[:, 1:]
    pos['gpt4'] = write_score(score_, pos)

    ask_gpt4(pos, name=code, desc=desc_code, add=add, api_key=api_key)

    pos = pd.read_csv(f"{config['path']}/supp_code/feature_selection/score_all/GPT4_ans_{code}_{add}.csv", index_col=0)
    pos_new = change_string_to_float(pos)
    pos_new.to_csv(f"{config['path']}/supp_code/feature_selection/score_all/GPT4_ans_{code}_{add}.csv", index=None)

    update_score(
        f"{config['path']}/supp_code/feature_selection/score_all/GPT4_ans_{code}_{add}.csv",
        f"{config['path']}/supp_code/feature_selection/input/GPT4_{code.replace(':','_')}.csv"
    )

    # Evaluate
    output_str, output_tmp = print_all(pos_new, name_list, name_all)
    with open(f"{config['path']}/output/{add}/GPT4_rank_ans_{code}.txt", 'a') as f:
        f.write('\n' + output_str)
    print(output_str)

    return pos_new, output_tmp




def write_score(pos_score, pos):
    if pos_score is None:
        desired_values = np.nan
    else:
        index_results = [np.where(pos_score.iloc[:, 0].values == x)[0] for x in pos.iloc[:, 1]]
        desired_values = [pos_score.iloc[idx[0], 1] if len(idx) > 0 else np.nan for idx in index_results]
    return desired_values


def find_indices_by_type(pos, selected_type='sap'):
    pos = pos.reset_index()
    # Check if 'selected_by' is a column in the dataframe
    if 'selected_by' not in pos.columns:
        raise ValueError("The DataFrame does not contain a 'selected_by' column")

    # Create a boolean mask where the type is in the 'selected_by' column after splitting by '&'
    mask = pos['selected_by'].apply(lambda x: selected_type in str(x).split('&'))

    # Get the indices where the mask is True
    indices = pos.index[mask].tolist()

    return indices


def concordance_index(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    assert len(a) == len(b)

    n = len(a)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            if b[i] == b[j]:
                continue  # No order between same outcomes
            if b[i] > b[j]:
                pred_diff = a[i] - a[j]
            else:
                pred_diff = a[j] - a[i]

            if pred_diff > 0:
                concordant += 1
            elif pred_diff < 0:
                discordant += 1
            else:
                tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return np.nan  # Undefined if all b are equal

    return (concordant + 0.5 * tied) / total

def read_and_compute(code_list, name_list,note='0519'):
    C_index_list = []
    for code in code_list:
        C_index_list2 = []
        colnames = [name + '_cos' for name in name_list]
        score_file = pd.read_csv(f"{config['path']}/supp_code/feature_selection/score_all/GPT4_ans_{code}_{note}.csv")
        score = score_file[colnames]
        for i in range(len(name_list)):
            C_index = concordance_index(score_file['gpt4'], score.iloc[:,i])
            C_index_list2.append(C_index)
        C_index_list.append(C_index_list2)
    return pd.DataFrame(C_index_list, index=code_list, columns=name_list)
    



def print_all(pos, name_list, unique_name):
    output = io.StringIO()
    for i in range(len(name_list)):
        # index = np.where(pos[pos['gpt4'].notna()]['name'].isin(unique_name[config['inst_row'][0]]))[0]
        output_tmp = compute_spearman(np.array(pos[pos['gpt4'].notna()][f'{name_list[i]}_cos']), np.array(pos[pos['gpt4'].notna()]['gpt4']))
        output_tmp = np.round(output_tmp[0],3)
        # output.write(f'{name_list[i]}: {output_tmp} ({len(index)})\t')
        output.write(f'{name_list[i]}: {output_tmp}\t')

    return output.getvalue(), output_tmp


def change_string_to_float(pos_all):
    pos_all_new = pos_all.copy()
    pos_all_new['gpt4'] = pos_all_new['gpt4'].apply(lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else 0.0)
    return pos_all_new


def update_score(path, path_origin=None):
    score_new = pd.read_csv(path)
    score_new = score_new[['desc', 'gpt4']]

    if os.path.exists(path_origin):
        score_ = pd.read_csv(path_origin)
        score_.columns = score_new.columns
        score_new = pd.concat([score_, score_new], axis=0)
        
    score_new.drop_duplicates(subset=score_new.columns[0], keep='first', inplace=True)
    score_new = score_new[pd.to_numeric(score_new.iloc[:,1], errors='coerce').notnull()]
    score_new.to_csv(path_origin, index=None)

    
def ask_gpt4(data, name='PheCode:714.1', desc='Rheumatoid Arthritis', add=None, api_key=None):
    # model_engine = "gpt-4"
    # model_engine = 'gpt-4-turbo-preview'
    # model_engine = "gpt-3.5-turbo"
    model_engine = 'gpt-4o-mini'
    client = OpenAI(api_key=api_key)
    pd.DataFrame(data.columns).transpose().to_csv(f"{config['path']}/supp_code/feature_selection/score_all/GPT4_ans_{name}_{add}.csv", header=None) 
    data.reset_index()
    data = np.asarray(data)
    temp = 0 # run the code if broke
    ans = []
    
    def gen_prompt(i):
        prompt = (f"Is the following medical code related to {desc}?\n\n"
                  f"{data[i,1]}\n\n"
                  f"Please provide a score between 0 and 1 indicating the likelihood that this code is related to {desc}. "
                  f"For instance, a score of 1 indicates complete relevance, whereas a score of 0 indicates no relevance. "
                  f"Provide your score as a decimal (e.g., 0.13, 0.75, etc.) without any additional information or explanations.")
        return prompt
    
    range_ask = np.where(np.isnan(data[:,-1].astype(float)))[0]
    for i in range(temp, data.shape[0]):
        if i not in range_ask:
            pd.DataFrame(data=data[i,:].reshape(-1,1)).transpose().to_csv(f"{config['path']}/supp_code/feature_selection/score_all/GPT4_ans_{name}_{add}.csv", header=False, mode="a")
            ans.append(data[i,-1])
            continue
        response = client.chat.completions.create(
            model=model_engine,
            temperature=0.0,
            messages=[
                {"role": "system",
                "content": f"As a knowledgeable assistant supporting a healthcare professional, your task is to help determine whether various medical codes are related to {desc} based on their provided details. The professional values concise yet precise responses."},
                {"role": "user",
                 "content": gen_prompt(i)}]
        )
        print(gen_prompt(i))
        message = response.choices[0].message.content
        print(message)
        ans.append(message)
        re = np.vstack([data[i,:-1].reshape(-1,1), ans[i - temp]])
        print(i, re)
        df = pd.DataFrame(data=re)
        df = df.transpose()
        df.to_csv(f"{config['path']}/supp_code/feature_selection/score_all/GPT4_ans_{name}_{add}.csv", header=False, mode="a")
        

def sample_and_combine_edges(edge_all_sim, edge_all_rel, config):
    num_edges_to_sample = round(edge_all_rel.size(1) * config['drop_p'])
    permuted_indices = torch.randperm(edge_all_rel.size(1))
    sampled_indices = permuted_indices[:num_edges_to_sample]
    sampled_edges = edge_all_rel[:, sampled_indices]
    edge_index = torch.cat((edge_all_sim, sampled_edges), dim=1)
    return edge_index