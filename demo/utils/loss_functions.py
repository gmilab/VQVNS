#### Custom loss functionst that might be use in training
import torch.nn as nn
import torch
import sys
import lightning as pl

#KL Divergence Loss for VAE
class VariationalLoss(nn.Module):
    def __init__(self, recon_loss="mse"):
        super(VariationalLoss, self).__init__()
        if recon_loss == "mse":
            self.recon_loss = nn.MSELoss()
        elif recon_loss == "bce":
            self.recon_loss = nn.BCELoss()
        else:
            raise ValueError("Unsupported reconstruction loss: choose 'mse' or 'bce'")
 
    def forward(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = self.recon_loss(recon_x, x)
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= x.size(0) * x.size(1) * x.size(2) * x.size(3)  # Normalize by batch size and dimensions
        return recon_loss + kl_loss
    

import torch
import torch.nn as nn
import torch.nn.functional as F
# from lpips import LPIPS
from torch.fft import fftn
from monai.losses.perceptual import PerceptualLoss

class Recon_Loss(nn.Module):
    def __init__(self, perceptual_weight=0.001, fft_weight=1.0, perceptual_network_type='radimagenet_resnet50', perceptual_fake_3d=True):
        """
        VQ-VAE Loss Function with:
        - L1 Reconstruction Loss
        - Spectral Loss (FFT-based)
        - Perceptual Loss (LPIPS)
        
        Args:
            perceptual_weight (float): Weight for perceptual loss (from paper: 0.001).
            fft_weight (float): Weight for spectral (FFT-based) loss (from paper: 1.0).
        """
        super(Recon_Loss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.fft_weight = fft_weight

        # Perceptual loss (LPIPS using AlexNet)
        self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type=perceptual_network_type, is_fake_3d=perceptual_fake_3d)

    def forward(self, reconstruction, target):
        """
        Computes the total loss for VQ-VAE.

        Args:
            reconstruction (Tensor): Reconstructed image.
            target (Tensor): Ground truth image.

        Returns:
            Tensor: Total loss.
        """
        target = target.float()
        y_pred = reconstruction.float()

        # === 1. Reconstruction Loss (L1 Norm) ===
        recon_loss = F.l1_loss(y_pred, target)

        # === 2. Spectral Loss (Fourier Transform) ===
        fft_target = torch.abs(fftn(target, norm="ortho"))
        fft_pred = torch.abs(fftn(y_pred, norm="ortho"))
        fft_loss = F.mse_loss(fft_pred, fft_target) * self.fft_weight

        # === 3. Perceptual Loss (LPIPS) ===
        perceptual_loss = self.perceptual_loss(reconstruction, target) * self.perceptual_weight

        # === Total Loss ===
        total_loss = recon_loss + fft_loss + perceptual_loss

        return total_loss
    
class Recon_Loss_GradMod(pl.LightningModule):
    def __init__(self, pix_loss_grad_weight=0.1, fft_loss_grad_weight=0.5, percep_loss_grad_weight=1, perceptual_network_type='alex', perceptual_fake_3d=True):
        """
        VQ-VAE Loss Function with:
        - L1 Reconstruction Loss
        - Spectral Loss (FFT-based)
        - Perceptual Loss (LPIPS)
        
        Args:
            perceptual_weight (float): Weight for perceptual loss (from paper: 0.001).
            fft_weight (float): Weight for spectral (FFT-based) loss (from paper: 1.0).
        """
        super(Recon_Loss_GradMod, self).__init__()
        self.percep_loss_grad_weight = percep_loss_grad_weight
        self.fft_loss_grad_weight = fft_loss_grad_weight
        self.pix_loss_grad_weight = pix_loss_grad_weight

        # Perceptual loss (LPIPS using AlexNet)
        self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type=perceptual_network_type, is_fake_3d=perceptual_fake_3d)

    def forward(self, reconstruction, target):
        """
        Computes the total loss for VQ-VAE.

        Args:
            reconstruction (Tensor): Reconstructed image.
            target (Tensor): Ground truth image.

        Returns:
            Tensor: Total loss.
        """
        target = target.float()
        y_pred_for_pix = gradnorm(reconstruction, weight=self.pix_loss_grad_weight)
        y_pred_for_fft = gradnorm(reconstruction, weight=self.fft_loss_grad_weight)
        y_pred_for_percep = gradnorm(reconstruction, weight=self.percep_loss_grad_weight)

        # === 1. Reconstruction Loss (L1 Norm) ===
        recon_loss = F.l1_loss(y_pred_for_pix, target)

        # === 2. Spectral Loss (Fourier Transform) ===
        with torch.amp.autocast(self.device.type, enabled=False):
            fft_target = torch.abs(fftn(target.float(), norm="ortho"))
            fft_pred = torch.abs(fftn(y_pred_for_fft.float(), norm="ortho"))

        fft_loss = F.mse_loss(fft_pred, fft_target)

        # === 3. Perceptual Loss (LPIPS) ===
        perceptual_loss = self.perceptual_loss(y_pred_for_percep, target) 

        # === Total Loss ===
        total_loss = recon_loss + fft_loss + perceptual_loss

        return total_loss, recon_loss, fft_loss, perceptual_loss
    
class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.saved_tensors[0]

        # grad_output_norm = torch.linalg.vector_norm(
        #     grad_output, dim=list(range(1, len(grad_output.shape))), keepdim=True
        # ).mean()
        with torch.amp.autocast(grad_output.device.type, enabled=False):
            grad_output_norm = torch.norm(grad_output).mean().item()

        grad_output_normalized = weight * grad_output / (grad_output_norm + 1e-8)

        return grad_output_normalized, None


def gradnorm(x, weight=1.0):
    weight = torch.tensor(weight, device=x.device)
    return GradNormFunction.apply(x, weight)

def contrastive_loss(embeddings, labels, temperature=0.1):
    """
    Computes supervised contrastive loss on embeddings.
    Encourages embeddings from the same class (label) to be close, LO
    and embeddings from different classes to be far apart.

    https://arxiv.org/pdf/2004.11362

    Args:
        embeddings: Tensor [batch_size, embedding_dim]
        labels: Tensor [batch_size] with binary labels (0 or 1)
        temperature: Scaling factor controlling the sharpness of distribution
    
    Returns:
        Scalar contrastive loss.
    """
    # Step 1: Normalize embeddings to unit sphere for cosine similarity.
    embeddings = F.normalize(embeddings, p=2, dim=1)  # shape: [batch_size, embedding_dim]

    # Step 2: Compute similarity matrix (cosine similarities scaled by temperature).
    similarity_matrix = torch.matmul(embeddings, embeddings.T)  # [batch_size, batch_size]
    similarity_matrix = similarity_matrix / temperature  # sharper distributions

    # Step 3: Reshape labels for mask creation.
    labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]

    # Step 4: Create a binary mask where mask[i, j] = 1 if labels[i] == labels[j], else 0.
    mask = torch.eq(labels, labels.T).float()  # [batch_size, batch_size]
    mask.fill_diagonal_(0) # mask out self-similarities

    # Step 5: For numerical stability, subtract max similarity from each row.
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Step 6: Exponentiate logits, removing the diagonal elements (self-similarities).
    exp_logits = torch.exp(logits) * (1 - torch.eye(labels.shape[0], device=labels.device))

    # Step 7: Compute log probability by normalizing with sum of exponentiated logits per row.
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

    # Step 8: Compute mean log probability for positive pairs (same class embeddings).
    # mask.sum(1) counts how many positives per row; avoids division by zero.
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # Step 9: Loss is the negative mean of these log probabilities (we want positives high).
    loss = -mean_log_prob_pos.mean()

    return loss

