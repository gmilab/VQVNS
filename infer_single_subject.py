import argparse
import numpy as np
import torch
import pickle
import nibabel as nib
from models.vqvae import BrainVQVAE
import monai.transforms as transforms


def load_mri(mri_path, pad_shape=(176, 208, 176)):
    """Load and preprocess a single MRI file, padding if needed."""
    img = nib.load(mri_path) 
    data = img.get_fdata()
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    transform = transforms.Compose([
                                # Crop or pad to match the expected dimensions (using self.crop_or_pad_dim)
                                transforms.SpatialPad(spatial_size=pad_shape)])
    data = transform(data)
    return data.unsqueeze(0)  # Add batch dimension (B, C, D, H, W)
    
def run_vqvae(model: BrainVQVAE, x, device):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        encoded = model.encode(x)
        quantized = model.quantize(encoded)[0]
        flat_latent = torch.mean(quantized, dim=(2, 3, 4))  # shape: (B, C)
        flat_latent = flat_latent.view(flat_latent.size(0), -1)  # flatten the latent space
        return flat_latent.cpu().numpy()  # convert to numpy array


def main(vqvae_checkpoint, svm_weights, features_to_keep, mri):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VQ-VAE model
    vqvae = BrainVQVAE.load_from_checkpoint(vqvae_checkpoint, map_location=device)
    vqvae = vqvae.to(device)

    # Load SVM and features
    svm = pickle.load(open(svm_weights, 'rb'))
    features_to_keep = np.load(features_to_keep)

    # Load and preprocess MRI
    x = load_mri(mri)

    # Pass through VQ-VAE to get latent features
    latents = run_vqvae(vqvae, x, device)

    # Select features
    latents_selected = latents[:, features_to_keep]

    # Predict with SVM
    pred = svm.predict(latents_selected)
    if pred == 1:
        print("Prediction: Responder")
    else:
        print("Prediction: Non-responder")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single subject inference using VQ-VAE and SVM.")
    parser.add_argument('--mri', type=str, required=True, help='Path to preprocessed MRI file (NIfTI)')
    parser.add_argument('--vqvae_checkpoint', type=str, default='weights/vqvae.ckpt', help='Path to VQ-VAE checkpoint')
    parser.add_argument('--svm_weights', type=str, default='weights/svm_weights.pkl', help='Path to SVM weights (pkl)')
    parser.add_argument('--features_to_keep', type=str, default='weights/features_to_keep.npy', help='Path to features_to_keep.npy')
    args = parser.parse_args()
    
    vqvae_checkpoint = args.vqvae_checkpoint
    svm_weights = args.svm_weights
    features_to_keep = args.features_to_keep
    mri = args.mri

    main(vqvae_checkpoint, svm_weights, features_to_keep, mri)