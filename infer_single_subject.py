import argparse
import numpy as np
import torch
import pickle
import nibabel as nib
from models.vqvae import BrainVQVAE

#Suppres future warnings
import warnings
warnings.filterwarnings("ignore")

#Add some colour to the print statements
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_mri(mri_path, pad_shape=(176, 208, 176)):
    """Load and preprocess a single MRI file, padding if needed."""
    img = nib.load(mri_path)
    data = img.get_fdata().astype(np.float32)
    # Pad to pad_shape if needed
    pad_width = [(0, max(0, p - s)) for s, p in zip(data.shape, pad_shape)]
    if any(pw[1] > 0 for pw in pad_width):
        data = np.pad(data, pad_width, mode='constant')
    # Crop if larger than pad_shape
    data = data[:pad_shape[0], :pad_shape[1], :pad_shape[2]]
    # Add batch and channel dimensions if needed
    if data.ndim == 3:
        data = data[None, None, ...]  # shape: (1, 1, D, H, W)
    elif data.ndim == 4:
        data = data[None, ...]        # shape: (1, C, D, H, W)
    return torch.from_numpy(data)


def run_vqvae(model: BrainVQVAE, x, device):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        quantized = model.encode_and_quantize(x)
        flat_latent = torch.mean(quantized, dim=(2, 3, 4))  # shape: (B, C)
        flat_latent = flat_latent.view(flat_latent.size(0), -1)  # flatten the latent space
        return flat_latent.cpu().numpy()  # convert to numpy array


def main(vqvae_checkpoint, svm_weights, features_to_keep, mri):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VQ-VAE model
    print(f"Loading VQ-VAE model...")
    vqvae = BrainVQVAE.load_from_checkpoint(vqvae_checkpoint, map_location=device)
    vqvae = vqvae.to(device)

    # Load SVM and features
    print(f"Loading SVM model and features...")
    with open(svm_weights, 'rb') as f:
        svm = pickle.load(f)
    features_to_keep = np.load(features_to_keep)

    # Load and preprocess MRI
    print(f"Loading and preprocessing MRI...")
    x = load_mri(mri)

    # Pass through VQ-VAE to get latent features
    latents = run_vqvae(vqvae, x, device)

    # Select features
    latents_selected = latents[:, features_to_keep]

    # Predict with SVM
    print(f"Predicting...")
    pred = svm.predict(latents_selected)
    if pred == 1:
        print(f"{bcolors.OKGREEN}Prediction: Responder{bcolors.ENDC}")
    else:
        print(f"{bcolors.FAIL}Prediction: Non-responder{bcolors.ENDC}")


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