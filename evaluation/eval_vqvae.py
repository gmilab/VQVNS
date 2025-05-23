##### This script will train a VAE model to generate the latent space for the VNS data that can them be used to predict the outcome of the VNS treatment

#### Hrishikesh Suresh
#### Ibrahim Lab 2025

# Import necessary libraries
import sys
from sys import exit
import os
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from tqdm import tqdm
from datasets.t1_datamodule import ImagingDataModule
from sklearn.model_selection import train_test_split
from models.vqvae import BrainVQVAE
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import shutil
from monai.networks.nets.patchgan_discriminator import PatchDiscriminator
from utils.loss_functions import Recon_Loss, Recon_Loss_GradMod, gradnorm
import yaml
import argparse
torch.set_float32_matmul_precision('high')

import warnings
 
warnings.filterwarnings(
    "ignore",
    message=".*`torch.cuda.amp.autocast*",
    category=FutureWarning
)

if os.environ.get('DEBUGGING') == '1':
    DEBUGGING = True
else:
    DEBUGGING = False

parser = argparse.ArgumentParser(description='Train a VQ-VAE model on any data with a datamodule')
parser.add_argument('--config', type=str, help='Path to the config file')
parser.add_argument('--resume_job_id', type=str, help='Job ID to resume from', required=False, default='')
parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint file', required=False, default='')
args = parser.parse_args()

# Load the config file
with open(args.config) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

images_dir = str(config['images_dir'])
metadata_csv = str(config['metadata_csv'])
intermediates_dir= str(config['intermediates_dir'])

if not os.path.exists(intermediates_dir):
    print('Evaluation requires that the intermediates directory exists.')
    sys.exit(1)

# Load the metadata
df = pd.read_csv(metadata_csv)

if not 'outcome' in df.columns:
    df['outcome'] = 0 # This is a placeholder for now when using healthy data

#Let's clean up the outcome column
df['outcome'] = df['outcome'].apply(lambda x: x if x >= 0 else 0)
df['raw_outcome'] = df['outcome'].copy()    
df['outcome'] = df['outcome'].apply(lambda x: 1 if x > 50 else 0)

# Define model parameters manually here
batch_size = int(config['batch_size'])
num_workers = int(config['num_workers'])
in_channels = int(config['in_channels'])
out_channels = int(config['out_channels'])
image_dim = tuple(map(int, config['image_dim']))
num_embeddings = int(config['num_embeddings'])
embedding_dim = int(config['embedding_dim'])
channels = tuple(map(int, config['channels']))
downsample_parameters = [tuple(map(int, x)) for x in config['downsample_parameters']]
upsample_parameters = [tuple(map(int, x)) for x in config['upsample_parameters']]
lr = float(config['lr'])
num_res_layers = int(config['num_res_layers'])
num_res_channels = tuple(map(int, config['num_res_channels']))
commitment_cost = float(config['commitment_cost'])
vqvae_decay = float(config['vqvae_decay'])

#Discriminator parameters
disc_lr = float(config['disc_lr'])
adv_weight = float(config['adv_weight'])
discriminator_train_start_epoch = int(config['discriminator_train_start_epoch'])
adversarial_loss_start_epoch = int(config['adversarial_loss_start_epoch'])
train_disc_every_n_batches = int(config['train_disc_every_n_batches'])
disc_spatial_dims = int(config['disc_spatial_dims'])
disc_in_channels = int(config['disc_in_channels'])
disc_channels = int(config['disc_channels'])
disc_num_layers_d = int(config['disc_num_layers_d'])
adv_epoch_weighting_denominator: 100 = int(config['adv_epoch_weighting_denominator'])
adv_loss_grad_weight = float(config['adv_loss_grad_weight'])

#Set up the data module
use_clinical = bool(config['use_clinical'])
use_train_transform = bool(config['use_train_transform'])
preload = bool(config['preload'])
augment_on_test = bool(config['augment_on_test'])

#Check if the debugger is on, if so make num_workers = 1
if DEBUGGING:
    num_workers = 1

data_module = ImagingDataModule(images_dir = images_dir, 
                                metadata_df = df, 
                                train_ids = [],
                                val_ids = [], 
                                test_ids = df['study_id'].values,
                                batch_size = batch_size, 
                                num_workers = num_workers,
                                use_clinical = use_clinical,
                                use_train_transform = use_train_transform,
                                preload = preload,
                                crop_or_pad_dim=image_dim, 
                                augment_on_test=augment_on_test)

#Instantiate discriminator
discriminator = PatchDiscriminator(spatial_dims=disc_spatial_dims, in_channels=disc_in_channels, channels=disc_channels, num_layers_d=disc_num_layers_d)

#Define the non-adversarial loss function
#Recon loss parameters
network_type = str(config['perceptual_network_type'])
perceptual_fake_3d = bool(config['perceptual_fake_3d'])
use_gradnorm = bool(config['use_gradnorm'])

if use_gradnorm:
    print('Using gradnorm for Recon Loss')
    try:
        pix_loss_grad_weight = float(config['pix_loss_grad_weight'])
        fft_loss_grad_weight = float(config['fft_loss_grad_weight'])
        percep_loss_grad_weight = float(config['percep_loss_grad_weight'])
    except KeyError:
        print('Gradnorm loss weights not provided in config file. Make sure to provide pix_loss_grad_weight, fft_loss_grad_weight, and percep_loss_grad_weight')
        sys.exit()

    recon_loss_fn = Recon_Loss_GradMod(pix_loss_grad_weight=pix_loss_grad_weight, fft_loss_grad_weight=fft_loss_grad_weight, percep_loss_grad_weight=percep_loss_grad_weight, perceptual_network_type=network_type, perceptual_fake_3d=perceptual_fake_3d)
else:
    print('Using normal Recon Loss with loss weights')
    try:
        fft_weight = float(config['fft_weight'])
        perceptual_weight = float(config['perceptual_weight'])
    except KeyError:
        print('Loss weights not provided in config file. Make sure to provide fft_weight and perceptual_weight when not using gradnorm')
        sys.exit()

    recon_loss_fn = Recon_Loss(perceptual_weight=perceptual_weight, fft_weight=fft_weight, perceptual_network_type=network_type, perceptual_fake_3d=perceptual_fake_3d)

#Define the model 
vae_model = BrainVQVAE(in_channels=in_channels,
                       out_channels=out_channels,
                       channels=channels,
                       num_res_layers=num_res_layers,
                       num_res_channels=num_res_channels,
                       downsample_parameters=downsample_parameters,
                       upsample_parameters=upsample_parameters,
                       num_embeddings=num_embeddings,
                       embedding_dim=embedding_dim,
                       intermediates_dir=intermediates_dir,
                       patch_discriminator_model=discriminator,
                       recon_loss_fn=recon_loss_fn,
                       discriminator_train_start_epoch=discriminator_train_start_epoch,
                       adversarial_loss_start_epoch=adversarial_loss_start_epoch,
                       lr=lr, 
                       disc_lr=disc_lr,
                       adv_weight=adv_weight,
                       train_disc_every_n_batches=train_disc_every_n_batches,
                       commitment_cost=commitment_cost,
                       decay=vqvae_decay,
                       adv_epoch_weighting_denominator=adv_epoch_weighting_denominator,
                       adv_loss_grad_weight=adv_loss_grad_weight
)


#Eval params
wandb_logger_save_dir = str(config['wandb_logger_save_dir'])
wandb_logger_project = str(config['wandb_logger_project'])
wandb_logger_entity = str(config['wandb_logger_entity'])
log_model = bool(config['log_model'])
max_epochs = int(config['max_epochs'])

if args.checkpoint_path == '':
    print('Evaluation cannot proceed without a job_id and a checkpoint path. Exiting...')
    sys.exit()

# Resume WandB session
# wandb.init(project=wandb_logger_project, entity=wandb_logger_entity, id=args.resume_job_id, resume="must", sync_tensorboard=True)
wandb.init(project=wandb_logger_project, entity=wandb_logger_entity, id='eval_run', sync_tensorboard=True)

# Set up WandB logger
wandb_logger = WandbLogger(
    dir=wandb_logger_save_dir,
    project=wandb_logger_project,
    entity=wandb_logger_entity,
    resume=False  # Ensures it continues logging to the same run
)

# Reinitialize Trainer and Resume from Checkpoint
trainer = pl.Trainer(
    max_epochs=max_epochs,
    logger=[wandb_logger],
    log_every_n_steps=1
)

trainer.test(vae_model, data_module, ckpt_path=args.checkpoint_path)