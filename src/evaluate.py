# pylint: skip-file

"""
Title: evaluate.py
Author: Han Tong
Date: 2025-07-01
Python Version: Python 3.11.3
Description: evaluate AUC and accuracy functions can be seen here
"""

import numpy as np
import pandas as pd
import warnings
from config import get_config
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from utils import *
import logging
config = get_config()
CHECK_ALL = config['CHECK_ALL']

# ---------------------Predict functions----------------------
def predict_fun(emb_other, emb_loinc, other_name, loinc_name, item_dict):
    ans_ = ans = 0
    similarity = np.dot(emb_other, emb_loinc.T)
    Pre = np.argsort(-similarity, axis=1)
    m = emb_loinc.shape[0]
    n = emb_other.shape[0]

    right_top1_ = right_top1 = np.zeros(n)
    right_top5_ = right_top5 = np.zeros(n)
    right_top10_ = right_top10 = np.zeros(n)
    right_top20_ = right_top20 = np.zeros(n)
    
    Pre = np.array(loinc_name)[Pre]
    Pre0 = Pre
    df = pd.DataFrame(Pre)
    result = df.apply(lambda x: pd.Series(x.unique()), axis=1).to_numpy()
    Pre = result[:, 0:20]
    right_top1, right_top5, right_top10, right_top20, ans = process_predictions(Pre, 1, other_name, item_dict)
    return np.vstack((right_top1, right_top5, right_top10, right_top20)).T


def output(new_emb, name_all):
    
    OtherToLoincLabel = np.load(f"{config['input_dir']}/edges/OtherToLoinc_new.npy",
                                allow_pickle=True).item()
    existed_row = np.where(np.in1d(name_all, list(OtherToLoincLabel.keys())))[0]
    index1 = [i for i, x in enumerate(name_all) if re.search("LOINC:", x)] 
    emb_loinc = new_emb[index1, :]
    emb_other = new_emb[existed_row, :]
    
    N1 = len(OtherToLoincLabel)
    pre = predict_fun(emb_other, emb_loinc, get_values(name_all[existed_row]), get_values(name_all[index1]), OtherToLoincLabel)
    return np.array([np.sum(pre[:, i])/N1 for i in range(4)])*100


#----------------------- evaluate AUC functions------------------
def get_NULLpairs(pairs, dict_MGB):
    '''
    add NULL pairs for every positive pairs, so we can compute AUC further
    '''
    mask = pairs['code1'].isin(dict_MGB) & pairs['code2'].isin(dict_MGB)
    pairs_MGB = pairs[mask]
    pairs_MGB.index = range(pairs_MGB.shape[0])
    temp = np.array(id_map(pairs_MGB['code1'].values, dict_MGB))
    temp2 = np.array(id_map(pairs_MGB['code2'].values, dict_MGB))
    pairs_MGB['Var1'] = temp
    pairs_MGB['Var2'] = temp2

    group = pd.DataFrame({
        'g1': np.array(ret_type(dict_MGB))[temp], 
        'g2': np.array(ret_type(dict_MGB))[temp2]
    })

    groupdf = pd.DataFrame({
        'g1': group.apply(min, axis=1), 
        'g2': group.apply(max, axis=1)
    })

    group = groupdf.drop_duplicates()
    pairs_MGB = pairs_MGB.copy()
    pairs_MGB['nullcode1'] = np.nan
    pairs_MGB['nullcode2'] = np.nan

    for i in range(group.shape[0]):
        idx = groupdf[(groupdf['g1'] == group.iloc[i, 0]) & (groupdf['g2'] == group.iloc[i, 1])].index
        posi = pairs_MGB.loc[np.array(idx), ['code1', 'code2']]
        posi = pd.DataFrame({
            'code1': np.concatenate([posi['code1'].values, posi['code2'].values]), 
            'code2': np.concatenate([posi['code2'].values, posi['code1'].values])
        })

        # generate negative samples for code1 and code2 separately from corresponding groups
        nega1 = pd.DataFrame({
            'code1': np.random.choice(dict_MGB[np.where(np.array(ret_type(dict_MGB)) == group.iloc[i, 0])[0]], posi.shape[0] * 2, replace=True)
        })

        nega2 = pd.DataFrame({
            'code2': np.random.choice(dict_MGB[np.where(np.array(ret_type(dict_MGB)) == group.iloc[i, 1])[0]], posi.shape[0] * 2, replace=True)
        })

        # combine negative samples for code1 and code2
        nega = pd.concat([nega1, nega2], axis=1)

        nega = nega[nega['code1'] != nega['code2']]
        nega = pd.DataFrame({
            'code1': nega.apply(min, axis=1), 
            'code2': nega.apply(max, axis=1)
        })
        nega = nega.drop_duplicates()
        if nega.shape[0] != 0:
            nega = pd.concat([nega, posi]).drop_duplicates(keep=False)
        if nega.shape[0] < len(idx):
            print(f"Warning: Not enough negative samples! g1 = {group.iloc[i, 0]}, g2 = {group.iloc[i, 1]}")
            pairs_MGB.loc[idx[0:nega.shape[0]], 'nullcode1'] = nega['code1'].values
            pairs_MGB.loc[idx[0:nega.shape[0]], 'nullcode2'] = nega['code2'].values
        else:
            pairs_MGB.loc[idx, 'nullcode1'] = nega['code1'].values[0:len(idx)]
            pairs_MGB.loc[idx, 'nullcode2'] = nega['code2'].values[0:len(idx)]

    return pairs_MGB


def get_AUC_sub(emb_MGB, now_sub, nmax = 1000):
    emb_MGB_cpu = emb_MGB.cpu()
    posi = now_sub.iloc[:, 5:7]
    nega = now_sub.assign(id1=id_map(now_sub['nullcode1'], emb_MGB.index), 
                        id2=id_map(now_sub['nullcode2'], emb_MGB.index))[['id1', 'id2']].dropna()
    n1 = posi.shape[0]; n0 = nega.shape[0]; n = min(n0, n1)
    if n < 10:
        return({'auc':None, 'num':0})
    else:
        y = y = np.array([[0] * min(n, nmax) , [1] * min(n, nmax)]).ravel() 
        p0 = (emb_MGB_cpu[nega.iloc[:, 0].values, :] * emb_MGB_cpu[nega.iloc[:, 1].values, :]).sum(axis=1)
        p1 = (emb_MGB_cpu[posi.iloc[:, 0].values, :] * emb_MGB_cpu[posi.iloc[:, 1].values, :]).sum(axis=1)
        p = pd.concat([pd.Series(p0).sample(min(n, nmax)), pd.Series(p1).sample(min(n, nmax))])
        roc0 = roc_auc_score(y, p)
        return({'auc':roc0, 'num':n})


def get_AUC_emb(emb_MGB, pairs_MGB, nmax = 1000):
    group = pairs_MGB[['type', 'group', 'source']].drop_duplicates()
    Group = group[['type', 'group']].drop_duplicates().sort_values(by=['group', 'type'], ascending=[True, False])
    AUClist = list()
    i = 0
    if CHECK_ALL:
        print(Group.iloc[i]['type'])
    subgroup = group[(group['type'] == Group.iloc[i]['type']) & (group['group'] == Group.iloc[i]['group'])]
    subpairs = pairs_MGB[(pairs_MGB['type'] == Group.iloc[i]['type']) & (pairs_MGB['group'] == Group.iloc[i]['group'])]
    AUCpart = [get_AUC_sub(emb_MGB, subpairs[subpairs['source'] == subgroup.iloc[j]['source']], nmax) for j in range(subgroup.shape[0])]
    AUCpart = pd.DataFrame(AUCpart, columns=['auc', 'num'], index=subgroup['source']).sort_values(by='num', ascending=False)
    AUClist.append(AUCpart)
    return AUClist


def get_total_AUC(emb, name_all, pairs):
    emb_latent = emb.detach()
    emb_latent.index = name_all
    pairs = get_NULLpairs(pairs, get_values(emb_latent.index))
    AUC = get_AUC_emb(emb_latent, pairs)
    return AUC


def test(x, name_all, config, related_pairs=None, similar_pairs=None, drug_side_pairs=None, PRE=True, AUC=True, AUC_type=True):
    # x_low_dim is used to do similar jobs; x_rel is used to do related jobs
    results = []
    if config['path_origin'] is 'align_NA':
        x_low_dim = x
    else:
        rmax=config['rmax']
        x_low_dim = x[:,:rmax]
        
    def compute_auc(data, pairs):
        if AUC_type:
            pairs['source'] = [''.join(sorted([x, y])) for x, y in zip(ret_type(pairs['code1']), ret_type(pairs['code2']))]
        return get_total_AUC(data, name_all, pairs)

    # Predict accuracy
    if PRE:
        my_z = x_low_dim.detach().cpu().numpy()
        pre = output(my_z, name_all)
        logging.info(f'Accuracy    TOP1 {pre[0]:.2f}%    TOP5 {pre[1]:.2f}%    TOP10 {pre[2]:.2f}%    TOP20 {pre[3]:.2f}%')
        results.append(pre)
        
    # related and similar AUC
    if AUC:
        if similar_pairs is not None:
            similar_pairs = similar_pairs.iloc[np.where(similar_pairs.iloc[:,0].isin(name_all))[0],:]
            SIMI = compute_auc(x_low_dim, similar_pairs)
            results.append([SIMI])
        if related_pairs is not None:
            related_pairs = related_pairs.iloc[np.where(related_pairs.iloc[:,0].isin(name_all))[0],:]
            RELA = compute_auc(x, related_pairs)
            results.append([RELA])
        if drug_side_pairs is not None:
            DRUG_SIDE = compute_auc(x, drug_side_pairs)
            results.append([DRUG_SIDE])

    return tuple(results)