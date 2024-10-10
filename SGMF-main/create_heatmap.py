from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore
from utils.utils import *
import os
import h5py

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level=-1, top_n=20, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)

    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    # heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    # return heatmap
    if isinstance(scores, torch.Tensor):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = np.array(scores)
    top_indices = np.argsort(scores_np)[::-1][:top_n]

    # Extract the top 15 scores and their corresponding coordinates
    top_scores = scores_np[top_indices]
    top_coords = coords[top_indices]
    try:
        heatmap = wsi_object.visHeatmap(scores=top_scores, coords=top_coords, vis_level=vis_level, **kwargs)
    # try:
    #     heatmap = wsi_object.visHeatmap(scores, coords, vis_level=vis_level, **kwargs)
    except IndexError:
        print(f"WARNING: Unable to generate heatmap for slide at vis_level={vis_level}. Skipping this slide.")
        heatmap = None
    return heatmap

data_dir_s = os.path.join(r'G:\huaxi\x10\h5_files')
scores_path = os.path.join(r'G:\Gnnexplain\x10\score5.pt')
scores_dict = torch.load(scores_path)
print(scores_dict)
wsi_path = os.path.join(r'G:\huaxi\svs')
for slide in os.listdir(data_dir_s):
    full_path = os.path.join(data_dir_s, slide)
    # 在这里处理每个slide文件o
    slide_name = slide[:-3]
    if slide_name in scores_dict:
        print(full_path)
        scores = scores_dict[slide_name]
        print(len(scores))
        slide_path = os.path.join(wsi_path, slide_name + '.svs')
    else:
        continue
    heatmap_args = {
    'vis_level' : 1,
    'cmap': 'coolwarm',
    'blank_canvas' : False ,
    'blur' : False ,
    'binarize' : False ,
    'custom_downsample' : 1 ,
    'alpha': 0.5 ,
    'patch_size': (1024, 1024)
    #'top_n':15
}
    p_slide_save_dir = os.path.join(r'G:\Gnnexplain\x10\patch')
    if not os.path.exists(p_slide_save_dir):
        os.makedirs(p_slide_save_dir)
    heatmap_save_name = f'{slide}.png'
    with h5py.File(full_path, 'r') as hdf5_file:
        coords = hdf5_file['coords'][:]
        print(len(coords))
    heatmap = drawHeatmap(scores, coords, slide_path, **heatmap_args)
    if heatmap is not None:
        heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)