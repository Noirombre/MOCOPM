from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from utils.core_utils import Accuracy_Logger
import h5py
import time 
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer

# Training settings 
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default='F:/huaxi',
                    help='data directory')
parser.add_argument('--data_folder_s', type=str, default='x10', help='dir under data directory' )
parser.add_argument('--data_folder_l', type=str, default='x20', help='dir under data directory' )
parser.add_argument('--tg_file', type=str, default='F:/huaxi/x5/graph_files', help='dir under data directory' )
parser.add_argument('--results_dir', type=str, default='F:/Gnnexplain',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to sproject root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default='task_3_pt_staging_cls3_100',
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default='task_3_pt_staging_cls3_100',
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default='F:/Gnnexplain/meta_pri',
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'patch_gcn', 'dgc'], default='patch_gcn',
                    help='type of model (default: clam_sb)')
parser.add_argument('--mode', type=str, choices=['clam', 'graph'], default='graph', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--drop_out', action='store_true', default=False,
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=0, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False,
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, default='task_3_pt_staging_cls2')

### patch_gcn specific options
parser.add_argument('--num_gcn_layers',  type=int, default=4, help = '# of GCN layers to use.')
parser.add_argument('--edge_agg',        type=str, default='spatial', help="What edge relationship to use for aggregation.")
parser.add_argument('--resample',        type=float, default=0.00, help='Dropping out random patches.')

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir,
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'mode': args.mode,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            mode = args.mode,
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False,
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = '../our_work_yfy/dataset_csv/TCGA_RCC_subtyping.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  TG_file= args.tg_file,
                                  print_info = True,
                                  label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2},
                                  patient_strat= False,
                                  ignore=[])

elif args.task == 'task_3_pt_staging_cls3':
    args.n_classes = 3
    dataset = Generic_MIL_Dataset(csv_path = '/home/zhangyuedi/SGMF-main/Multiclass.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  TG_file= args.tg_file,
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'yuanfa': 0, 'wei': 1, 'zhichang': 2},
                                  patient_strat= False,
                                  ignore=[])

elif args.task == 'task_3_pt_staging_cls2':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = 'F:/result/Binary.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  TG_file= args.tg_file,
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'yuanfa': 0, 'zhuanyi': 1},
                                  patient_strat= False,
                                  ignore=[])


else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'patch_gcn':
        from models.model_SGMF import SGMF
        model_dict = {'num_layers': args.num_gcn_layers, 'edge_agg': args.edge_agg, 'resample': args.resample, 'n_classes': args.n_classes}
        model = SGMF(**model_dict)
    elif args.model_type == 'dgc':
        from models.model_graph_mil import DeepGraphConv_Surv
        model_dict = {'edge_agg': args.edge_agg, 'resample': args.resample, 'n_classes': args.n_classes}
        model = DeepGraphConv_Surv(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    # print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    # model.eval()
    return model

if __name__ == "__main__":
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
    model = initiate_model(args, ckpt_paths[ckpt_idx])
    print('Init Loaders')
    loader = get_simple_loader(split_dataset, mode=args.mode)
    slide_ids = loader.dataset.slide_data['slide_id']
    for batch_idx, (data_s, data_l, tissue_data, label, attn_mask_s, attn_mask_l) in enumerate(loader):
        data_s, data_l, tissue_data, label = data_s.to(device), data_l.to(device), tissue_data.to(device), label.to(device)
        attn_mask_s, attn_mask_l = attn_mask_s.to(device), attn_mask_l.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=5),
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='probs',
        ),
    )
        print('ok')
        explanation = explainer(
        tissue_data=tissue_data,
        index=None,
        **{'data_l': data_l, 'data_s': data_s, 'attn_mask_s': attn_mask_s, 'attn_mask_l': attn_mask_l}
    )
        print(f'Generated explanations in {explanation.available_explanations}')


        path_networkx = f'F:/Gnnexplain/GNNExplainer_results_networkx/subgraph{slide_id}.pdf'
        explanation.visualize_graph(path_networkx, backend = "networkx")
        print(f"Subgraph visualization plot has been saved to '{path_networkx}'")
