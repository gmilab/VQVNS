#### This is the VAE model that will be used to generate the latent space for the VNS data 
#### The latent space will be used to capture features from the data that can be used to predict the outcome of the VNS treatment

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from monai.networks.nets import varautoencoder
import sys
from utils.loss_functions import *
import numpy as np
from utils.misc_functions import save_np_as_nifti
import wandb
import os
import shutil

# Define the VAE model
class BrainVAE(pl.LightningModule):
   
    def __init__(self, in_shape, in_channels, out_channels, latent_size, channels, strides, intermediates_dir, kl_beta=0.1, lr=1e-5, num_res_units=0):
        super(BrainVAE, self).__init__()
        self.save_hyperparameters()
        self.model = varautoencoder.VarAutoEncoder(
            spatial_dims=3,
            in_shape=(in_channels, *in_shape),  # image spatial shape
            out_channels=1, # number of output channels
            latent_size=latent_size, # latent vector size
            channels=channels, # number of features for each layer 12 x 14 x 12 after 4 conv layers with stride 2
            strides=strides, # strides for each conv layer
            num_res_units=num_res_units, # number of residual units
        )
        self.variational_loss = VariationalLoss(recon_loss="mse")   
        self.recon_loss = Recon_Loss(perceptual_weight=0.3, fft_weight=1.0)

        # Global monitoring variables
        self.train_xhat_samples = []
        self.val_xhat_samples = []
        self.test_predictions = []
        self.val_losses = []
        self.intermediates_dir = intermediates_dir
        self.epoch_count = 0
        self.kl_beta = kl_beta

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat, mu, log_var, z = self.model(x)
        loss = self.variational_loss(x_hat, x, mu, log_var) + self.recon_loss(x_hat, x)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        if batch_idx == 1:
            self.train_xhat_samples.append(x_hat.detach().cpu().numpy())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat, mu, log_var, z = self.model(x)
        loss = self.variational_loss(x_hat, x, mu, log_var) + self.recon_loss(x_hat, x)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        if batch_idx == 1:
            self.val_xhat_samples.append(x_hat.detach().cpu().numpy())
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat, mu, log_var, z = self.model(x)
        loss = self.variational_loss(x_hat, x, mu, log_var) + self.recon_loss(x_hat, x)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.test_predictions.append(x_hat.detach().cpu().numpy())
        return loss
    
    def on_train_epoch_end(self):
        self.train_xhat_samples = np.concatenate(self.train_xhat_samples, axis=0)
        save_np_as_nifti(self.train_xhat_samples, f'{self.intermediates_dir}/epoch{self.epoch_count}_train_xhat_samples.nii.gz')
        if self.epoch_count > 0: 
            os.remove(f'{self.intermediates_dir}/epoch{self.epoch_count - 1}_train_xhat_samples.nii.gz')
        sample_img = self.train_xhat_samples[0].squeeze() 
        slices = [sample_img[80, :, :], sample_img[:, 96, :], sample_img[:, :, 80]]
        self.logger.experiment.log({'train_reconstructions_x': wandb.Image(slices[0])})
        self.logger.experiment.log({'train_reconstructions_y': wandb.Image(slices[1])})
        self.logger.experiment.log({'train_reconstructions_z': wandb.Image(slices[2])})
        self.train_xhat_samples = []
        self.epoch_count += 1
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):
        self.val_xhat_samples = np.concatenate(self.val_xhat_samples, axis=0)
        save_np_as_nifti(self.val_xhat_samples, f'{self.intermediates_dir}/epoch{self.epoch_count}_val_xhat_samples.nii.gz')
        if self.epoch_count > 0:
            os.remove(f'{self.intermediates_dir}/epoch{self.epoch_count - 1}_val_xhat_samples.nii.gz')
        sample_img = self.val_xhat_samples[0].squeeze()
        slices = [sample_img[80, :, :], sample_img[:, 96, :], sample_img[:, :, 80]]
        self.logger.experiment.log({'val_reconstructions_x': wandb.Image(slices[0])})
        self.logger.experiment.log({'val_reconstructions_y': wandb.Image(slices[1])})
        self.logger.experiment.log({'val_reconstructions_z': wandb.Image(slices[2])})
        self.val_xhat_samples = []
        return super().on_validation_epoch_end()
    
    def on_test_epoch_end(self):
        self.test_predictions = np.concatenate(self.test_predictions, axis=0)
        save_np_as_nifti(self.test_predictions, f'{self.intermediates_dir}/epoch{self.epoch_count}_test_predictions.nii.gz')
        self.test_predictions = []
        return super().on_test_epoch_end()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.lr)
        return optimizer
    

    
    

    




   