import torch
# import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import h5py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(path_tissue, path_l, path_h, save_path):
    pbar = tqdm(os.listdir(path_tissue))
    for h5_fname in pbar:
        pbar.set_description('%s - Creating attn_mask' % (h5_fname[:-3]))
        tissue_wsi_h5 = h5py.File(os.path.join(path_tissue, h5_fname), "r")
        l_wsi_h5 = h5py.File(os.path.join(path_l, h5_fname), "r")
        h_wsi_h5 = h5py.File(os.path.join(path_h, h5_fname), "r")
        attn_mask_l, attn_mask_h = get_attn_mask(tissue_wsi_h5, l_wsi_h5, h_wsi_h5)
        os.makedirs(os.path.join(save_path, 'x10', 'attn_mask'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'x20', 'attn_mask'), exist_ok=True)
        torch.save(attn_mask_l, os.path.join(save_path, 'x10', 'attn_mask', h5_fname[:-3]+'.pt'))
        torch.save(attn_mask_h, os.path.join(save_path, 'x20', 'attn_mask', h5_fname[:-3]+'.pt'))
        tissue_wsi_h5.close()
        l_wsi_h5.close()
        h_wsi_h5.close()
        

def get_attn_mask(tissue_wsi_h5, l_wsi_h5, h_wsi_h5):
    tissue_coords = np.array(tissue_wsi_h5['coords'])
    low_resolution_coords = np.array(l_wsi_h5['coords'])
    high_resolution_coords = np.array(h_wsi_h5['coords'])
    n_t = tissue_coords.shape[0]
    n_l = low_resolution_coords.shape[0]
    n_h = high_resolution_coords.shape[0]

    x_t_l = np.repeat(tissue_coords[:,0].reshape(-1,1), n_l, axis=1) # n_t x n_l
    x_l_t = np.repeat(low_resolution_coords[:,0].reshape(1,-1), n_t, axis=0) # n_t x n_l
    x_attn_l = np.array((x_t_l<=x_l_t)&((x_t_l+2048)>x_l_t))
    y_t_l = np.repeat(tissue_coords[:,1].reshape(-1,1), n_l, axis=1) # n_t x n_l
    y_l_t = np.repeat(low_resolution_coords[:,1].reshape(1,-1), n_t, axis=0) # n_t x n_l
    y_attn_l = np.array((y_t_l<=y_l_t)&((y_t_l+2048)>y_l_t))
    attn_l = np.array(x_attn_l&y_attn_l)

    x_t_h = np.repeat(tissue_coords[:,0].reshape(-1,1), n_h, axis=1) # n_t x n_h
    x_h_t = np.repeat(high_resolution_coords[:,0].reshape(1,-1), n_t, axis=0) # n_t x n_h
    x_attn_h = np.array((x_t_h<=x_h_t)&((x_t_h+2048)>x_h_t))
    y_t_h = np.repeat(tissue_coords[:,1].reshape(-1,1), n_h, axis=1) # n_t x n_h
    y_h_t = np.repeat(high_resolution_coords[:,1].reshape(1,-1), n_t, axis=0) # n_t x n_h
    y_attn_h = np.array((y_t_h<=y_h_t)&((y_t_h+2048)>y_h_t))
    attn_h = np.array(x_attn_h&y_attn_h)

    # attn_mask_l = torch.from_numpy(attn_l).type(torch.LongTensor).unsqueeze(0)
    # attn_mask_h = torch.from_numpy(attn_h).type(torch.LongTensor).unsqueeze(0)
    attn_mask_l = torch.from_numpy(attn_l).type(torch.ByteTensor).unsqueeze(0)
    attn_mask_h = torch.from_numpy(attn_h).type(torch.ByteTensor).unsqueeze(0)
    return attn_mask_l, attn_mask_h

