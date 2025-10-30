# pylint: skip-file

"""
Title: main.py
Author: Han Tong
Date: 2025-10-08
Python Version: Python 3.11.3
Description: main file of our attention model
"""


import torch
from torch_geometric.utils import to_undirected
from torch_geometric.utils import add_self_loops
import warnings
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import logging
warnings.filterwarnings('ignore')
import gc
import argparse
import random
import numpy as np   
from config import set_config
import argparse
import pdb
from utils import sample_and_combine_edges, now_time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
warnings.filterwarnings('ignore')
start_time = now_time()
START_time = time.time()

def update_config_from_args():
    from config import set_config, get_config
    
    config = get_config()

    parser = argparse.ArgumentParser(description="GAMEw Training Script")
    parser.add_argument("--drop_out", type=float, default=0.5,
                        help="Parameter drop_out prob. Default: 0.5.") 
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Parameter learning rate. Default: 1e-4.")  
    parser.add_argument("--min_lr", type=float, default=5e-7,
                        help="Parameter learning rate. Default: 5e-7.")
    parser.add_argument("--AA", type=float, default=1.0,
                        help="Parameter AA. Default: 1.0")
    parser.add_argument("--BB", type=float, default=5.0,
                        help="Parameter BB. Default: 5.0")
    parser.add_argument("--lambd", type=float, default=0.5,
                        help="Parameter lambd. Default: 0.5.")
    parser.add_argument("--scale_hie", type=int, default=1,
                        help="Parameter scale_hie. Default: 1.")
    parser.add_argument("--scale_sppmi", type=float, default=0.1,
                        help="Parameter scale_sppmi. Default: 0.1.")
    parser.add_argument("--scale_OTOL", type=int, default=50,
                        help="Parameter scale local lab to LOINC. Default: 50.")
    parser.add_argument("--scale_REL", type=int, default=5,
                        help="Parameter scale_REL. Default: 5.")   
    parser.add_argument("--scale_align", type=int, default=1,
                        help="Parameter scale_REL. Default: 1.")  
    parser.add_argument("--rmax", type=int, default=256,
                        help="Parameter r_max we use for similarity. Default: 256.") 
    parser.add_argument("--hidden_features", type=int, default=768,
                        help="Parameter dimension we use for all. Default: 768.")
    parser.add_argument("--path", type=str, default=config['path'],
                        help="Specify the path parameter.")
    parser.add_argument("--input_dir", type=str, default=config['input_dir'],
                        help="Specify the path parameter for input data.")
    parser.add_argument("--path_origin", type=str, default=config['path_origin'],
                        help='Get aligned sppmi if embedding path_origin is align_NA. Else if is None, train the similar step. Else if is not None(the similar embedding), train the related step. Train from the initial model and embedding path_origin is not None. Default: None.')
    parser.add_argument("--align_path", type=str, default=None,
                        help='Pretrained aligned sppmi embedding path. Default: None.')
    parser.add_argument("--epochs", type=int, default=3,
                    help='Total Epochs. Default: 3.')
    parser.add_argument("--CHECK_ALL", type=bool, default=False,
                    help='whether to check attention or not. Default: False.')
    parser.add_argument("--DEVICE", type=str, default='cuda:0',
                    help='Use GPU or CPU. Default: cuda:0.')
    parser.add_argument("--num_inst", type=int, default=7,
                    help='The number of institutions. Default: 7.')


    args = parser.parse_args()
    config['num_inst'] = args.num_inst
    config['base_lr'] = args.lr
    config['min_lr'] = args.min_lr
    config['drop_p'] = args.drop_out   
    config['AA'] = args.AA
    config['BB'] = args.BB
    config['lambd'] = args.lambd
    config['scale_hie'] = args.scale_hie
    config['scale_sppmi'] = args.scale_sppmi
    config['scale_OTOL'] = args.scale_OTOL
    config['scale_REL'] = args.scale_REL
    config['scale_align'] = args.scale_align
    config['rmax'] = args.rmax
    config['hidden_features'] = args.hidden_features
    config['input_dir'] = args.input_dir
    config['path'] = args.path
    config['path_origin'] = args.path_origin
    config['align_path'] = args.align_path
    config['epochs'] = args.epochs
    config['CHECK_ALL'] = args.CHECK_ALL
    config['DEVICE'] = args.DEVICE
    
    seed = config['SEED']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
    set_config(config)
    logging.info(config)
        

def main(config): 
    
    from load_data import sppmi_list, sap_emb, unique_name, type_list, ALL_rel_val_pairs, ALL_sim_val_pairs, my_objects, pos_sppmi, neg_sppmi, edges_map, edges_hie, sim_edges, rel_edges, same_desc_edge
    
    # load data to device
    device = torch.device(config['DEVICE'])

    if config['path_origin'] == "align_NA":
        sppmi_list = [inst_emb.to(device) for inst_emb in sppmi_list]
    else:
        # we have get aligned sppmi emb, stored in  .../align_sppmi folder
        print('load aligned sppmi embedding!')
        out_1 =  torch.load(f"{config['path']}/output/{config['align_path']}/align_sppmi.pth", map_location=device)
        out_1 = out_1.detach()
        out_1 = out_1.to(device)

    if (config['path_origin'] != 'align_NA') & (config['path_origin'] is not None):
        # simi embedding will be fixed duing related training process
        print('load similarity embedding!')
        x_sim_trained = torch.load(f"{config['path']}/output/{config['path_origin']}/sim_emb.pth", map_location=device)
        x_sim_trained = x_sim_trained.detach().to(device)
        
    sap_emb = sap_emb.to(device)
    
    # load GAT model
    model_all = inst_encoder(config)
    model_all = model_all.to(device)
    logging_config(config, start_time, no_console=False)
    logging.info(model_all)
    optimizer0 = optim.SGD(model_all.parameters(), lr=config['base_lr'])
    scheduler0 = CustomExponentialLR(optimizer0, gamma=config['gamma'], min_lr=config['min_lr'])
    
    # load edges
    ## edges_map: code mapping
    ## edges_hie: hierarchy
    ## edges_sim: non-hierarchy edges sim
    ## edges_rel: edges_rel
    ## pos_sppmi: uncommon edges selected by GPT4
    edges_sim = np.row_stack((edges_map, edges_hie,sim_edges))
    edge_all_sim = np.row_stack((id_map(edges_sim[:,0], unique_name), id_map(edges_sim[:,1], unique_name)))
    edge_all_sim =  torch.as_tensor(edge_all_sim, dtype=torch.long)
    
    edges_rel = np.row_stack((rel_edges, pos_sppmi.values))
    edge_all_rel = np.row_stack((id_map(edges_rel[:,0], unique_name), id_map(edges_rel[:,1], unique_name)))
    edge_all_rel =  torch.as_tensor(edge_all_rel, dtype=torch.long)

    record = -float('inf')
   
    edge_index = torch.cat((edge_all_sim, edge_all_rel), dim = 1)
    mask = (edge_index[0] >=0) & (edge_index[1] >=0)
    edge_index = edge_index[:, mask]
    edge_index = remove_duplicate_edge(edge_index)       
    undirected_edge_index = to_undirected(edge_index).to(device)
    print(f'undirected_edge_index.shape = {undirected_edge_index.shape}')
    
    # begin training
    for epoch in range(0, config['epochs']):
        now_time = time.time()
        case_store = False
        optimizer0.zero_grad()
        model_all.train()
        
        # align sppmi case
        if config['path_origin'] == "align_NA":
            with torch.cuda.amp.autocast():
                align_loss_term, x_sim = model_all(sppmi_list=sppmi_list, sap_emb=sap_emb, edge_index=undirected_edge_index, config=config)
                loss0 = align_loss_term
                loss = [my_item(loss0)]
    
                # update
                loss0.backward()
                optimizer0.step()
                scheduler0.step()

        # simi embedding training case
        elif config['path_origin'] is None:
            with torch.cuda.amp.autocast():
                x_sim = model_all(sap_emb=sap_emb, out_1=out_1, edge_index=undirected_edge_index, config=config)
                P_LOSS_hie, N_LOSS_hie, P_LOSS_OTOL, N_LOSS_OTOL, P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE = custom_loss(my_objects, x_sim, list(range(config['num_union'])), device, unique_name, config, TYP1=True)
                loss0 = P_LOSS_hie + N_LOSS_hie + P_LOSS_OTOL + N_LOSS_OTOL + P_LOSS_SIM_NO_HIE + N_LOSS_SIM_NO_HIE      
                loss = [my_item(P_LOSS_hie), my_item(N_LOSS_hie), 
                        my_item(P_LOSS_OTOL), my_item(N_LOSS_OTOL),
                        my_item(P_LOSS_SIM_NO_HIE), my_item(N_LOSS_SIM_NO_HIE)]
    
                # update
                loss0.backward()
                optimizer0.step()
                scheduler0.step()
    
                write_file(epoch, 0, config, start_time, loss=loss)
                
        # rela embedding training case
        else:
            with torch.cuda.amp.autocast():
                x_rel_part = model_all(sap_emb=sap_emb, out_1=out_1, edge_index=undirected_edge_index, config=config)
                x_rel = torch.cat((x_sim_trained, x_rel_part), dim=1) # concat fixed simi embedding
                P_REL, N_REL, P_sppmi, N_sppmi = custom_loss(my_objects, x_rel, list(range(config['num_union'])), device, unique_name, config, TYP1=False)
                loss0 = P_REL + N_REL + P_sppmi + N_sppmi
                loss = [my_item(P_REL), my_item(N_REL), my_item(P_sppmi), my_item(N_sppmi)]
            
                # update
                loss0.backward()
                optimizer0.step()
                scheduler0.step()

                write_file(epoch, 0, config, start_time, loss=loss)
                
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            # evaluate 
            if epoch % 5 == 0:
                model_all.eval()
                
                if config['path_origin'] == 'align_NA':
                    x_sim_test = model_all(sppmi_list, sap_emb=sap_emb, edge_index=undirected_edge_index, config=config)
                    # acc, sim, rel
                    PRE_new, AUC_new, AUC_new3 = test(x_sim_test, unique_name, config, type_list, similar_pairs=ALL_sim_val_pairs, related_pairs=ALL_rel_val_pairs, PRE=True, AUC=True, AUC_type=True)
                    write_file(
                        Epoch=epoch,
                        Batch=0,
                        config=config,
                        start_time=start_time,
                        loss=loss,
                        pre=PRE_new,
                        SIM_AUC=AUC_new[0][0],
                        REL_AUC=AUC_new3[0][0]
                    )
                    
                    # evalution
                    sim_auc = weight_auc(AUC_new[0])
                    rel_auc = weight_auc(AUC_new3[0])
                    
                    logging.info(f'Weighted Similar AUC = {sim_auc}')
                    logging.info(f'Weighted Related AUC = {rel_auc}')


                    # whether to break training and store model
                    case_store = (sim_auc > record)
                    if case_store:
                        if epoch > 1:
                            record = sim_auc
                    elif epoch > 50:
                        break

                elif config['path_origin'] is None:
                    x_sim_test = model_all(sap_emb=sap_emb, out_1=out_1, edge_index=undirected_edge_index, config=config)
                    PRE_new, AUC_new = test(x_sim_test, unique_name, config, type_list, similar_pairs=ALL_sim_val_pairs, PRE=True, AUC=True, AUC_type=True)
                    sim_auc = weight_auc(AUC_new[0])
                    write_file(
                        Epoch=epoch,
                        Batch=0,
                        config=config,
                        start_time=start_time,
                        loss=loss,
                        pre=PRE_new,
                        SIM_AUC=AUC_new[0][0]
                    )
                
                    logging.info(f'Weighted Similar AUC = {sim_auc}')
                    
                    case_store = (sim_auc > record)
                    if case_store:
                        record = sim_auc

                else:
                    x_rel_part_test = model_all(sap_emb=sap_emb, out_1=out_1, edge_index=undirected_edge_index, config=config)
                    x_rel_test = torch.cat((x_sim_trained, x_rel_part_test), dim=1)
                    AUC_new = test(x_rel_test, unique_name, config, type_list, related_pairs=ALL_rel_val_pairs, PRE=False, AUC=True, AUC_type=True)  
                    rel_auc = weight_auc(AUC_new[0][0])
                    write_file(
                        Epoch=epoch,
                        Batch=0,
                        config=config,
                        start_time=start_time,
                        loss=loss,
                        REL_AUC=AUC_new[0][0][0]
                    )
                    
                    logging.info(f'Weighted Related AUC = {rel_auc}')
                    
                    case_store = (rel_auc > record)
                    if case_store:
                        record = rel_auc  

            else:
                write_file(epoch, 0, config, start_time, loss=loss)  
        
        
            # Store the embedding or not
            if case_store:
                if config['path_origin'] == "align_NA":
                    torch.save(x_sim_test, f"{config['path']}/output/{start_time}/align_sppmi.pth")
                    torch.save(model_all.state_dict(), f"{config['path']}/output/{start_time}/model_align.pth")  

                elif config['path_origin'] is None:
                    torch.save(x_sim_test, f"{config['path']}/output/{start_time}/sim_emb_{epoch}.pth")
                    torch.save(model_all.state_dict(), f"{config['path']}/output/{start_time}/model_sim.pth")   
                    emb = pd.DataFrame(x_sim_test.cpu().detach().numpy())
                    emb.to_csv(f"{config['path']}/output/{start_time}/sim_emb.csv", index=None)

                else:
                    torch.save(x_rel_test, f"{config['path']}/output/{start_time}/rel_emb_{epoch}.pth")
                    torch.save(model_all.state_dict(), f"{config['path']}/output/{start_time}/model_rel.pth") 
                    emb = pd.DataFrame(x_rel_test.cpu().detach().numpy())
                    emb.to_csv(f"{config['path']}/output/{start_time}/rel_emb.csv", index=None)
                
                end_time = time.time()
                time_elapsed = end_time - START_time
                logging.info('EPOCH: {:03d}     Saved model...   All time ELAPSED: {:.2f}s'.format(epoch, time_elapsed))

        # record time
        end_time = time.time()
        time_elapsed = end_time - now_time
        logging.info('EPOCH: {:03d}     LOSS: {:.4f}     ELAPSED: {:.2f}s'.format(epoch, my_item(loss0), time_elapsed))
        
        # clean cache
        torch.cuda.empty_cache()
        gc.collect()

            
if __name__ == "__main__":
    
    update_config_from_args()
    
    from utils import *
    from data_structure import *
    from evaluate import *
    from Attention import *
    config = get_config()
    
    main(config)
