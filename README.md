# MOCOPM
The code of our model:MOCOPM

# Patch Extraction
To extract the patches from the downloaded WSIs, users need to run the following commands:
```bash
python get_patches_fp.py
python extract_features_fp.py
```
# Graph Construction
TO get graphs and masks, users need to run the following commands:
```bash
python graph_construction.py
python attn_mask.py
```
# Trained Model Checkpoints
```bash
python main.py
```
