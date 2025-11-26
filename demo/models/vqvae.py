#### This is the VAE model that will be used to generate the latent space for the VNS data 
#### The latent space will be used to capture features from the data that can be used to predict the outcome of the VNS treatment

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import monai.networks.nets.vqvae as vqvae
import sys
from utils.loss_functions import *
import numpy as np
from utils.misc_functions import save_np_as_nifti
import wandb
import os
import shutil
from torch.nn import MSELoss
from monai.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai.losses.adversarial_loss import PatchAdversarialLoss
from models.quantizers import SimVQQuantizer, SimVQQuantizerWrapper
from monai.losses.ssim_loss import SSIMLoss
from monai.losses.multi_scale import MultiScaleLoss
from monai.losses.image_dissimilarity import GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss

# Define the VAE model
class BrainVQVAE(pl.LightningModule):
   
    def __init__(self, in_channels, out_channels, num_embeddings, downsample_parameters, 
                 upsample_parameters, embedding_dim, channels, intermediates_dir, 
                 num_res_layers, patch_discriminator_model, recon_loss_fn, 
                 num_res_channels,lr=1e-5, disc_lr=5e-5, adv_weight=0.5, 
                 discriminator_train_start_epoch=0, adversarial_loss_start_epoch=0, decay=0.5, commitment_cost=0.25, train_disc_every_n_batches=5, 
                 adv_epoch_weighting_denominator=100, lecam_loss_weight=0.1, lecam_beta=0.9, adv_loss_grad_weight=1.0, quant_loss_grad_weight=1.0, 
                 grad_norm_quant=False, replace_quantizer=False):
        super(BrainVQVAE, self).__init__()

        self.model = vqvae.VQVAE(
            spatial_dims=3,
            in_channels=in_channels,  # number of input channels
            out_channels=out_channels, # number of output channels
            channels=channels, # number of features for each layer 12 x 14 x 12 after 4 conv layers with stride 2
            downsample_parameters=downsample_parameters, # stride (int), kernel_size (int), dilation (int) and padding (int).
            upsample_parameters=upsample_parameters, # stride (int), kernel_size (int), dilation (int), padding (int), output_padding (int).
            num_embeddings=num_embeddings, # number of embeddings
            embedding_dim=embedding_dim, # embedding dimension
            embedding_init= "normal",
            commitment_cost = commitment_cost,
            decay = decay,
            epsilon = 1e-5,
            dropout = 0.0,
            num_res_layers = num_res_layers,
            num_res_channels = num_res_channels,
        )
        self.replace_quantizer = replace_quantizer
        if replace_quantizer:
            self.model.quantizer = SimVQQuantizerWrapper(
                        quantizer = SimVQQuantizer(
                            spatial_dims=3,
                            num_embeddings=num_embeddings,
                            embedding_dim=embedding_dim,
                            commitment_cost=commitment_cost,
                            embedding_init="normal",
                )
            )

        self.lr = lr
        self.disc_lr = disc_lr  
        self.automatic_optimization = False  # Enable manual optimization
        self.num_embeddings = num_embeddings

        # Discriminator that is used for the adversarial loss
        self.discriminator = patch_discriminator_model
        self.train_disc_every_n_batches = train_disc_every_n_batches

        # Global monitoring variables
        self.train_xhat_samples = []
        self.train_x_samples = []
        self.val_xhat_samples = []
        self.val_x_samples = []
        self.test_predictions = []
        self.test_originals = []
        self.val_losses = []
        self.intermediates_dir = intermediates_dir

        # Other vars
        self.recon_loss_fn = recon_loss_fn
        self.adv_loss_fn = PatchAdversarialLoss()
        self.adv_weight = adv_weight
        self.discriminator_train_start_epoch = discriminator_train_start_epoch
        self.adversarial_loss_start_epoch = adversarial_loss_start_epoch
        self.adv_epoch_weighting_denominator = adv_epoch_weighting_denominator
        self.adv_loss_grad_weight = adv_loss_grad_weight
        self.quant_loss_grad_weight = quant_loss_grad_weight
        self.grad_norm_quant = grad_norm_quant
        self.mssim_loss = MultiScaleLoss(SSIMLoss(spatial_dims=3))
        self.gmi_loss = GlobalMutualInformationLoss()
        self.lncc_loss = LocalNormalizedCrossCorrelationLoss(reduction='mean')

        #LeCam Advertisarial Loss
        self.lecam_loss_weight = lecam_loss_weight # how much weight the lecam loss should have
        self.lecam_anchor_real = 0
        self.lecam_anchor_fake = 0
        self.lecam_beta = lecam_beta #how much to weight the anchor logits when updating the anchor logits

        self.sync_dist = True if torch.cuda.device_count() > 1 else False
        
        self.save_hyperparameters()
    def forward(self, x):
        return self.model(x)
    
    def get_generator_loss(self, x, x_hat, quant_loss):

        loss_vqvae, pix_loss, fft_loss, perceptual_loss = self.recon_loss_fn(x_hat, x)

        total_generator_loss = loss_vqvae + quant_loss

        if self.discriminator is not None and self.current_epoch >= self.adversarial_loss_start_epoch:
            recon_for_gan = gradnorm(x_hat, weight=self.adv_loss_grad_weight)
            logits_fake = self.discriminator(recon_for_gan)[-1]
            loss_adv = self.adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
            total_generator_loss += loss_adv 

        return total_generator_loss, pix_loss, fft_loss, perceptual_loss, loss_adv
    
    def _log_losses(self, total_loss, pix_loss, fft_loss, perceptual_loss, loss_adv, quant_loss, mode):
        self.log(f'{mode}_loss', total_loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log(f'{mode}_recon_loss', pix_loss + fft_loss + perceptual_loss, on_epoch=True, on_step=True, sync_dist=self.sync_dist)
        self.log(f'{mode}_adv_loss', loss_adv, on_epoch=True, on_step=True, sync_dist=self.sync_dist)
        self.log(f'{mode}_pix_loss', pix_loss, on_epoch=True, on_step=True, sync_dist=self.sync_dist)
        self.log(f'{mode}_fft_loss', fft_loss, on_epoch=True, on_step=True, sync_dist=self.sync_dist)
        self.log(f'{mode}_perceptual_loss', perceptual_loss, on_epoch=True, on_step=True, sync_dist=self.sync_dist)
        self.log(f'{mode}_quant_loss', quant_loss, on_epoch=True, on_step=True, sync_dist=self.sync_dist)
 
    def training_step(self, batch, batch_idx):
        # Get optimizers
        if self.discriminator is not None:
            optimizer_vqvae, optimizer_disc = self.optimizers()
        else:
            optimizer_vqvae = self.optimizers()

        #Get generator outputs
        x, _ = batch

        #Forward pass
        z = self.model.encode(x) #First generate the latent representation
        if self.grad_norm_quant:
            norm_z = gradnorm(z, weight=self.quant_loss_grad_weight) #Normatlize the gradients
            quantized, quant_loss = self.model.quantize(norm_z) #Quantize, and get quant loss. But since Z has been normalized, we can use it for the quant loss
        else:
            quantized, quant_loss = self.model.quantize(z)
        x_hat = self.model.decode(quantized) #Decode the quantized representation
        #x_hat, quant_loss = self.model(x)
        perplexity = self.model.quantizer.perplexity

        #Lots encoded latent variance
        self.log("train_latent_variance", z.var(), on_epoch=True, on_step=True, sync_dist=self.sync_dist)

        # === Train Discriminator ===
        if self.discriminator is not None and self.current_epoch >= self.discriminator_train_start_epoch and self.global_step % self.train_disc_every_n_batches == 0:
            
            logits_fake = self.discriminator(x_hat.detach())[-1]
            logits_real = self.discriminator(x)[-1]

            loss_disc_fake = self.adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
            loss_disc_real = self.adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
            loss_disc = (loss_disc_fake + loss_disc_real) / 2

            #LeCam Adversarial Loss
            avg_real_logits = logits_real.mean().item()
            avg_fake_logits = logits_fake.mean().item()
            self.lecam_anchor_real = self.lecam_beta * self.lecam_anchor_real + (1 - self.lecam_beta) * avg_real_logits
            self.lecam_anchor_fake = self.lecam_beta * self.lecam_anchor_fake + (1 - self.lecam_beta) * avg_fake_logits

            lecam_loss = ((logits_real - self.lecam_anchor_fake)**2).mean() + ((logits_fake - self.lecam_anchor_real)**2).mean()
            total_disc_loss = loss_disc + lecam_loss * self.lecam_loss_weight

            #loss_disc = loss_disc * self.adv_weight
            self.log("train_isolated_disc_loss", loss_disc, on_epoch=True, on_step=True, sync_dist=self.sync_dist)
            self.log("train_lecam_loss", lecam_loss * self.lecam_loss_weight, on_epoch=True, on_step=True, sync_dist=self.sync_dist)
            self.log('train_total_disc_loss', total_disc_loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=self.sync_dist)
            optimizer_disc.zero_grad()
            self.manual_backward(total_disc_loss)
            optimizer_disc.step()

         # === Train VQ-VAE (Generator) ===
        total_generator_loss, pix_loss, fft_loss, perceptual_loss, loss_adv = self.get_generator_loss(x, x_hat, quant_loss)
        self._log_losses(total_generator_loss, pix_loss, fft_loss, perceptual_loss, loss_adv, quant_loss, 'train')
        self.log("codebook_perplexity", perplexity, on_epoch=True, on_step=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log("codebook_perplexity_pct", (perplexity / self.num_embeddings) * 100 , on_epoch=True, on_step=True, prog_bar=True, sync_dist=self.sync_dist)
        if self.replace_quantizer:
            self.log('codebook_w_rank', self.model.quantizer.w_rank, on_epoch=True, on_step=True, prog_bar=True, sync_dist=self.sync_dist)
            self.log('codebook_w_frobenius_norm', self.model.quantizer.w_frobenius_norm, on_epoch=True, on_step=True, prog_bar=True, sync_dist=self.sync_dist)
        optimizer_vqvae.zero_grad()
        self.manual_backward(total_generator_loss)
        optimizer_vqvae.step()

        if batch_idx == 1:
            self.train_xhat_samples.append(x_hat.detach().cpu().numpy())
            self.train_x_samples.append(x.detach().cpu().numpy())
        
        return total_generator_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        # === Validate VQ-VAE (Generator) ===
        z = self.model.encode(x) #First generate the latent representation
        if self.grad_norm_quant:
            norm_z = gradnorm(z, weight=self.quant_loss_grad_weight) #Normatlize the gradients
            quantized, quant_loss = self.model.quantize(norm_z) #Quantize, and get quant loss. But since Z has been normalized, we can use it for the quant loss
        else:
            quantized, quant_loss = self.model.quantize(z)
        x_hat = self.model.decode(quantized) #Decode the quantized representation

        total_generator_loss, pix_loss, fft_loss, perceptual_loss, loss_adv = self.get_generator_loss(x, x_hat, quant_loss)
        self._log_losses(total_generator_loss, pix_loss, fft_loss, perceptual_loss, loss_adv, quant_loss, 'val')
        self.log("val_latent_variance", z.var(), on_epoch=True, on_step=True, sync_dist=self.sync_dist)

        # === Validate Discriminator ===
        if self.discriminator is not None and self.current_epoch >= self.discriminator_train_start_epoch and self.global_step % self.train_disc_every_n_batches == 0:
           logits_fake = self.discriminator(x_hat.detach())[-1]
           logits_real = self.discriminator(x)[-1]
           loss_disc_fake = self.adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
           loss_disc_real = self.adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
           loss_disc = (loss_disc_fake + loss_disc_real) / 2

           #Log it
           self.log("val_isolated_disc_loss", loss_disc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=self.sync_dist)

        if batch_idx == 1:
            self.val_xhat_samples.append(x_hat.detach().cpu().numpy())
            self.val_x_samples.append(x.detach().cpu().numpy())

        return total_generator_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat, quant_loss = self.model(x)
        total_generator_loss, pix_loss, fft_loss, perceptual_loss, loss_adv = self.get_generator_loss(x, x_hat, quant_loss)
        self._log_losses(total_generator_loss, pix_loss, fft_loss, perceptual_loss, loss_adv, quant_loss, 'test')
        self.test_predictions.append(x_hat.detach().cpu().numpy())
        self.test_originals.append(x.detach().cpu().numpy())
        return total_generator_loss

    def _log_slices(self, subject_volumes, name):
        x_slices = []
        y_slices = []
        z_slices = []
        for subject in subject_volumes:
            x_slices.append(subject[80, :, :])
            y_slices.append(subject[:, 96, :])
            z_slices.append(subject[:, :, 80])

        self.logger.experiment.log({name + '_reconstructions_x': [wandb.Image(slice_) for slice_ in x_slices]})
        self.logger.experiment.log({name + '_reconstructions_y': [wandb.Image(slice_) for slice_ in y_slices]})
        self.logger.experiment.log({name + '_reconstructions_z': [wandb.Image(slice_) for slice_ in z_slices]})
    
    def on_train_epoch_end(self):
        self.train_xhat_samples = np.concatenate(self.train_xhat_samples, axis=0)
        save_np_as_nifti(self.train_xhat_samples, f'{self.intermediates_dir}/epoch{self.current_epoch}_train_xhat_samples.nii.gz')
        if self.current_epoch > 0 and os.path.exists(f'{self.intermediates_dir}/epoch{self.current_epoch - 1}_train_xhat_samples.nii.gz'): 
            try:
                os.remove(f'{self.intermediates_dir}/epoch{self.current_epoch - 1}_train_xhat_samples.nii.gz')
            except:
                print(f"File {self.intermediates_dir}/epoch{self.current_epoch - 1}_train_xhat_samples.nii.gz does not exist")
        
        self.train_x_samples = np.concatenate(self.train_x_samples, axis=0)
        save_np_as_nifti(self.train_x_samples, f'{self.intermediates_dir}/train_x_samples.nii.gz')

        #Log slices from 2 subjects
        self._log_slices(self.train_xhat_samples[:5].squeeze(), 'train')

        # #They have to be tensors
        # self.train_xhat_samples = torch.tensor(self.train_xhat_samples).squeeze(dim=0)
        # self.train_x_samples = torch.tensor(self.train_x_samples)

        # #Loss metrics that are *not* part of the total loss
        # mssim = 1 - self.mssim_loss(self.train_xhat_samples, self.train_x_samples) #1 minus becuase this is a loss function and returns the loss, not the similarity by default
        # gmi = -self.gmi_loss(self.train_xhat_samples, self.train_x_samples)
        # lncc = 1 - self.lncc_loss(self.train_xhat_samples, self.train_x_samples)

        # self.log('train_mssim', mssim, on_epoch=True, on_step=False)
        # self.log('train_gmi', gmi, on_epoch=True, on_step=False)
        # self.log('train_lncc', lncc, on_epoch=True, on_step=False)
        
        self.train_xhat_samples = []
        self.train_x_samples = []
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):
        self.val_xhat_samples = np.concatenate(self.val_xhat_samples, axis=0)
        save_np_as_nifti(self.val_xhat_samples, f'{self.intermediates_dir}/epoch{self.current_epoch}_val_xhat_samples.nii.gz')
        if self.current_epoch > 0 and os.path.exists(f'{self.intermediates_dir}/epoch{self.current_epoch - 1}_val_xhat_samples.nii.gz'):
            try: 
                os.remove(f'{self.intermediates_dir}/epoch{self.current_epoch - 1}_val_xhat_samples.nii.gz')
            except:
                print(f"File {self.intermediates_dir}/epoch{self.current_epoch - 1}_val_xhat_samples.nii.gz does not exist")

        if not os.path.exists(f'{self.intermediates_dir}/val_x_samples.nii.gz'):
            self.val_x_samples = np.concatenate(self.val_x_samples, axis=0)
            save_np_as_nifti(self.val_x_samples, f'{self.intermediates_dir}/val_x_samples.nii.gz')
        
        #Log slices from 2 subjects
        self._log_slices(self.val_xhat_samples[:5].squeeze(), 'val')

        self.val_xhat_samples = []
        self.val_x_samples = []
        return super().on_validation_epoch_end()
    
    def on_test_epoch_end(self):
        self.test_predictions = np.concatenate(self.test_predictions, axis=0)
        save_np_as_nifti(self.test_predictions, f'{self.intermediates_dir}/test_predictions.nii.gz')
        
        self.test_originals = np.concatenate(self.test_originals, axis=0)
        save_np_as_nifti(self.test_originals, f'{self.intermediates_dir}/test_originals.nii.gz')

        self.test_predictions = []
        self.test_originals = []

        return super().on_test_epoch_end()
    
    def configure_optimizers(self):
        optimizer_vqvae = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.discriminator is not None:
            optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr, betas=(0.5, 0.99))
            return [optimizer_vqvae, optimizer_disc], []
        else:
            return optimizer_vqvae
        
    def encode(self, x):
        return self.model.encode(x)
    
    def quantize(self, z):
        return self.model.quantize(z)
        
    def encode_and_quantize(self, x):
        z = self.model.encode(x)
        quantized, _ = self.model.quantize(z)
        return quantized
    
    def decode(self, quantized):
        return self.model.decode(quantized)
    

        
    

    
    

    




   