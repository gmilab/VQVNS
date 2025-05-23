##### This script will take a 4D array of input images and recons from the VQVAE and perform a thorough quality eval to present in the paper

### Hrishikesh Suresh
### Ibrahim Lab 2025

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import argparse
import shutil
import torch
from monai.metrics import MultiScaleSSIMMetric, FIDMetric, MMDMetric, MSEMetric
from monai.networks.nets.resnet import ResNetFeatures
from tqdm.auto import tqdm, trange
from utils.misc_functions import map_local_hpc_path_to_actual_hpc_path

DEBUG = True

if not DEBUG:
    parser = argparse.ArgumentParser(description='Evaluate the reconstruction fidelity of VQVAE')
    parser.add_argument('-r', '--recons', type=str, required=True, help='Path to the input recon file')
    parser.add_argument('-i', '--images', type=str, required=True, help='Path to the input images file')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--synthseg_inputs_dir', type=str, required=True, help='Path to directory with synthseg outputs for input images')
    parser.add_argument('--synthseg_recons_dir', type=str, required=True, help='Path to directory with synthseg outputs for reconstructions')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the output directory')
    args = parser.parse_args()

    recons_path = args.recons
    images_path = args.images
    output_dir = args.output_dir
    synthseg_inputs_dir = args.synthseg_inputs_dir
    synthseg_recons_dir = args.synthseg_recons_dir
    overwrite = args.overwrite
else:
    # For debugging, set the arguments directly
    recons_path = 'test_predictions.nii.gz'  # <-- Set your test recon file
    images_path = 'test_originals.nii.gz'    # <-- Set your test originals file
    output_dir = './testing/'                # <-- Set your output directory
    synthseg_inputs_dir = './synthseg_inputs/'   # <-- Set your synthseg input dir
    synthseg_recons_dir = './synthseg_recons/'   # <-- Set your synthseg recon dir
    overwrite = False

if not os.path.exists(output_dir) and output_dir != '':
    os.makedirs(output_dir)
    print(f'Output directory {output_dir} created.')

#Load the images
print(f'Loading images from {images_path}\nand reconstructions from {recons_path}...', end=' ')
images = nib.load(images_path).get_fdata()

if os.path.exists(os.path.join(output_dir, 'recons_clipped.nii.gz')) and not overwrite:
    print(f'Found existing reconstructions in {output_dir}.')
    recons_clipped = nib.load(os.path.join(output_dir, 'recons_clipped.nii.gz')).get_fdata()
    recons = recons_clipped
else:
    recons = nib.load(recons_path).get_fdata()
    #Make sure there are no negative values in the recons as these are meaningless
    recons[recons < 0] = 0
    recons_clipped = nib.Nifti1Image(recons, nib.load(recons_path).affine, nib.load(recons_path).header)
    nib.save(recons_clipped, os.path.join(output_dir, 'recons_clipped.nii.gz'))
print('Done!')

#Check if the images and reconstructions have the same shape
if images.shape != recons.shape:
    raise ValueError(f'Input images and reconstructions must have the same shape. Got {images.shape} and {recons.shape}.')

# --- METRIC CALCULATION (MSE, MSSIM, FID) ---

#Make some tensors
images_tensor = torch.from_numpy(images).float().moveaxis(-1, 0).unsqueeze(1)
recons_tensor = torch.from_numpy(recons).float().moveaxis(-1, 0).unsqueeze(1)

batch_size = 2
ssim_metric = MultiScaleSSIMMetric(spatial_dims=3)
mse_metric = MSEMetric()
fid_metric = FIDMetric()
resnet_features = ResNetFeatures(model_name='resnet50', pretrained=True)
resnet_features.eval()

metrics_df = pd.DataFrame(columns=['MS_SSIM', 'MSE', 'FID'])

with torch.no_grad():
    for i in trange(0, images_tensor.shape[0], batch_size, desc='Evaluating batches'):
        images_batch = images_tensor[i:i+batch_size]
        recons_batch = recons_tensor[i:i+batch_size]
        ms_ssim = ssim_metric(recons_batch, images_batch).numpy().mean()
        mse = mse_metric(recons_batch, images_batch).numpy().mean()
        image_features = resnet_features(images_batch)[4]
        recon_features = resnet_features(recons_batch)[4]
        image_features = torch.mean(image_features, dim=[2,3,4])
        recon_features = torch.mean(recon_features, dim=[2,3,4])
        fid = fid_metric(recon_features, image_features).item()
        metrics_df = pd.concat([metrics_df, pd.DataFrame([[ms_ssim, mse, fid]], columns=metrics_df.columns)], axis=0, ignore_index=True)

metrics_df.to_csv(os.path.join(output_dir, 'evals.csv'), index=False)

print('Mean of each metric:')
for col in metrics_df.columns:
    print(f'{col}: {metrics_df[col].mean()}')

# --- CORRELATION FOR 4-STRUCTURE MODEL ---

# Define the 4-structure model
simple_class_regions = {
    'Grey Matter': [3, 8, 42, 47, 17, 53],
    'Subcortical Grey': [10, 11, 12, 13, 49, 50, 51, 52, 18, 26, 58],
    'White Matter': [2, 7, 41, 46],
    'CSF': [4, 5, 14, 15, 43, 44],
}

# Paths to synthseg outputs (assume already run)
synthseg_inputs_dir = '/d/gmi/1/hrishikeshsuresh/hpc_storage/hsuresh/vns_deep_learning/training_outputs/intermediates_vqvae_combined_pretraining_data_driven_dragon/testing/synthseg_inputs/'
synthseg_recons_dir = '/d/gmi/1/hrishikeshsuresh/hpc_storage/hsuresh/vns_deep_learning/training_outputs/intermediates_vqvae_combined_pretraining_data_driven_dragon/testing/synthseg_recons/'

recon_vol_df = pd.DataFrame(columns=['subj', 'structure', 'true_volume', 'recon_volume'])
synthseg_inputs = os.listdir(synthseg_inputs_dir)
for subj_file in tqdm(synthseg_inputs, desc='Processing synthseg outputs'):
    subj_name = subj_file.split('.')[0]
    input_file = os.path.join(synthseg_inputs_dir, subj_file)
    recon_file = os.path.join(synthseg_recons_dir, subj_file)
    input_img = nib.load(input_file).get_fdata()
    recon_img = nib.load(recon_file).get_fdata()
    for region, labels in simple_class_regions.items():
        true_volume = np.sum(np.isin(input_img, labels))
        recon_volume = np.sum(np.isin(recon_img, labels))
        recon_vol_df = pd.concat([
            recon_vol_df,
            pd.DataFrame([[subj_name, region, true_volume, recon_volume]], columns=recon_vol_df.columns)
        ], axis=0, ignore_index=True)

# Correlation plots for each structure
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
for structure in simple_class_regions.keys():
    structure_df = recon_vol_df.loc[recon_vol_df['structure'] == structure]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(data=structure_df, y='true_volume', x='recon_volume', ax=ax,
                line_kws={'color': 'darkred', 'label': 'Regression Line'}, 
                scatter_kws={'s': 10, 'alpha': 0.5}, 
                marker='o')
    r, p = pearsonr(structure_df['true_volume'], structure_df['recon_volume'])
    r_sq = r**2
    ax.set_xlabel('Reconstructed Volume (mm$^3$)')
    ax.set_ylabel('True Volume (mm$^3$)')
    ax.set_title(f'{structure} Volume\nPearson R: {r:.4f}, p: {p:.2e}, R^2: {r_sq:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{structure}_volume.png'), dpi=300, bbox_inches='tight')
    plt.show()
