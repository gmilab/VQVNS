##### This script will evaluate the perforamnce of the SVM and then grad cam and other stuff

#### Hrishikesh Suresh
#### Ibrahim Lab 2025

# Import necessary libraries
import sys
import os
if not os.path.isdir('/hpf/projects/'):
    sys.path.insert(0, '../')
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
from tqdm.auto import tqdm
from datasets.t1_datamodule import ImagingDataModule
from sklearn.model_selection import train_test_split
from models.vqvae import BrainVQVAE
from models.classifiers import TransformerEncoderClassifier
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import shutil
from monai.networks.nets.patchgan_discriminator import PatchDiscriminator
from utils.loss_functions import Recon_Loss, Recon_Loss_GradMod, gradnorm
import yaml
import argparse
torch.set_float32_matmul_precision('high')
import pickle
import warnings
import shap
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from utils.misc_functions import save_model_space_image
 
warnings.filterwarnings(
    "ignore",
    message=".*`torch.cuda.amp.autocast*",
    category=FutureWarning
)

if os.environ.get('DEBUGGING') == '1':
    DEBUGGING = True
else:
    DEBUGGING = False

#Plot it all nicely 
def plot_curve_and_matrix(preds, probas, labels, dataset_name, intermediates_dir):

    #Adjust samplew weight to account for the imbalance
    non_responder_count = np.sum(labels == 0)
    responder_count = np.sum(labels == 1)
    pos_weight = non_responder_count / responder_count

    sample_weight = np.ones_like(labels)
    sample_weight[labels == 1] = pos_weight

    fpr, tpr, thresholds = roc_curve(labels, probas, sample_weight=sample_weight)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt='d', ax=ax[0], cmap='Blues')
    ax[0].set_title(f'{dataset_name} Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')

    auc = roc_auc_score(labels, probas, sample_weight=sample_weight)
    ax[1].plot(fpr, tpr, label=f'AUC: {auc:.4f}')
    ax[1].plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    ax[1].set_title(f'{dataset_name} ROC Curve')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    #add legend for random classifier and the model
    ax[1].legend(loc='lower right')
    plt.savefig(os.path.join(intermediates_dir, f'{dataset_name}_roc_curve.png'))
    plt.show()

    #Print the optimal threshold 
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return auc, optimal_threshold   

def eval(config):
    images_dir = str(config['images_dir'])
    metadata_csv = str(config['metadata_csv'])
    intermediates_dir= str(config['intermediates_dir'])
    outcome_col = str(config['outcome_col'])

    if os.path.exists(intermediates_dir):
        shutil.rmtree(intermediates_dir)
    os.makedirs(intermediates_dir)

    # Load the metadata
    df = pd.read_csv(metadata_csv)

    if not outcome_col in df.columns:
        raise ValueError('The metadata file must contain an outcome column. Since this is a classification task')
    
    if len(np.unique(df[outcome_col])) != 2:
        if 'binarization_threshold' in config:
                df['raw_outcome'] = df[outcome_col].copy()
                df[outcome_col] = df[outcome_col].apply(lambda x: 1 if x > float(config['binarization_threshold']) else 0)
        else:
            raise ValueError('The outcome column must be binary. Please binarize the outcome columnor provide a binarization threshold in the config file')

    # Define universal params
    batch_size = int(config['batch_size'])
    num_workers = int(config['num_workers'])

    #Set up the data module
    use_clinical = bool(config['use_clinical'])
    use_train_transform = bool(config['use_train_transform'])
    preload = bool(config['preload'])
    image_dim = tuple(map(int, config['image_dim']))

    #Check if the debugger is on, if so make num_workers = 1
    if DEBUGGING:
        num_workers = 1
    
    #Load the checkpoint
    vqvae = BrainVQVAE.load_from_checkpoint(config['vqvae_checkpoint_path'])
    vqvae.intermediates_dir = intermediates_dir

    #Load the SVM classifier
    classifier = pickle.load(open(config['svm_path'], 'rb'))
    features_to_keep = np.load(config['svm_feature_indices'])

    
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
                                    outcome_col=outcome_col)

    data_module.setup()
    test_loader = data_module.test_dataloader()

    vqvae.eval()
    predictions = []
    probabilities = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Generating predictions', total=len(test_loader)):
            x, label = batch

            x = x.to(vqvae.device)
            label = label.to(vqvae.device)

            latents = vqvae.encode(x)
            quantized_latents = vqvae.quantize(latents)[0]

            mean_latent = torch.mean(quantized_latents, dim=[2,3,4])

            detached_latent = mean_latent.detach().cpu().numpy()[:, features_to_keep].reshape(1, -1)

            outcome_pred = classifier.predict(detached_latent)
            outcome_prob = classifier.predict_proba(detached_latent)[:,1]

            predictions.append(outcome_pred)
            probabilities.append(outcome_prob)
            labels.append(label.cpu().detach().numpy())

    #Save the predictions and labels
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    labels = np.array(labels)
    np.save(os.path.join(intermediates_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(intermediates_dir, 'probabilities.npy'), probabilities)
    np.save(os.path.join(intermediates_dir, 'labels.npy'), labels)

    #Plot the confusion matrix and ROC curve
    auc, optimal_threshold = plot_curve_and_matrix(predictions, probabilities, labels, 'test', intermediates_dir)
    print(f'Optimal threshold: {optimal_threshold}')
    print(f'AUC: {auc}')

def run_gradcam_analysis(config):
    images_dir = str(config['images_dir'])
    intermediates_dir= str(config['intermediates_dir'])

    if not os.path.exists(intermediates_dir):
        os.makedirs(intermediates_dir)

    train_df = pd.read_csv(os.path.join(config['dataframes_folder'], 'train_df.csv'))
    val_df = pd.read_csv(os.path.join(config['dataframes_folder'], 'val_df.csv'))
    test_df = pd.read_csv(os.path.join(config['dataframes_folder'], 'test_df.csv'))

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    df = pd.concat([train_df, val_df, test_df], axis=0)

    if not 'outcome' in df.columns:
        raise ValueError('The metadata file must contain an outcome column. Since this is a classification task')

    # Define universal params
    batch_size = int(config['batch_size'])
    num_workers = int(config['num_workers'])

    #Set up the data module
    use_clinical = bool(config['use_clinical'])
    use_train_transform = bool(config['use_train_transform'])
    preload = bool(config['preload'])
    image_dim = tuple(map(int, config['image_dim']))
    outcome_col = str(config['outcome_col'])

    #Check if the debugger is on, if so make num_workers = 1
    if DEBUGGING:
        num_workers = 1

    data_module = ImagingDataModule(images_dir = images_dir, 
                                    metadata_df = df, 
                                    train_ids = [],
                                    val_ids = df[df['split'] == 'train']['study_id'].values, 
                                    test_ids = df['study_id'].values,
                                    batch_size = 1, 
                                    num_workers = num_workers,
                                    use_clinical = use_clinical,
                                    use_train_transform = use_train_transform,
                                    preload = preload,
                                    crop_or_pad_dim=image_dim, 
                                    outcome_col=outcome_col)

    #Load the checkpoint
    vqvae = BrainVQVAE.load_from_checkpoint(config['vqvae_checkpoint_path'])
    vqvae.intermediates_dir = intermediates_dir

    #Load the SVM classifier
    classifer = pickle.load(open(config['svm_path'], 'rb'))
    features_to_keep = np.load(config['svm_feature_indices'])

    #Target layer for gradcam
    target_layer = vqvae.model.encoder.blocks[16].conv

    #Get the dataloaders 
    data_module.setup()
    test_loader = data_module.test_dataloader()
    train_loader = data_module.val_dataloader()

    vqvae.eval()

    train_predictions = []
    full_quantized_latents = []

    grad_cam_dir = os.path.join(intermediates_dir, 'gradcam_v2')
    if not os.path.exists(grad_cam_dir):
        os.makedirs(grad_cam_dir)

    with torch.no_grad():
        #First run all the training data (in the validation data loader) to get the median of each feature
        for batch in tqdm(train_loader, desc='Generating predictions for training data', total=len(train_loader)):
            x, label = batch

            x = x.to(vqvae.device)
            label = label.to(vqvae.device)

            latents = vqvae.encode(x)
            quantized_latents = vqvae.quantize(latents)[0]

            mean_latent = torch.mean(quantized_latents, dim=[2,3,4])
            mean_latent = mean_latent.view(1, 32, -1)

            detached_latent = mean_latent.detach().cpu().numpy().reshape(-1)[features_to_keep]

            train_predictions.append(detached_latent)
            full_quantized_latents.append(quantized_latents.detach().cpu().numpy())

    #Make it a numpy array
    train_predictions = np.array(train_predictions)
    full_quantized_latents = np.array(full_quantized_latents).squeeze()

    #Get the median of each feature
    median_latent = np.median(train_predictions, axis=0).reshape(1, -1) 

    #Make the explainer
    def get_pred(x):
        return classifer.predict_proba(x)[:,1]
    
    explainer = shap.Explainer(get_pred, median_latent)
 
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    latent_heatmaps = []
    heatmaps = []
    images = []
    labels = []
    preds = []

    for batch in tqdm(test_loader, desc='Generating gradient maps', total=len(test_loader)):
        activations.clear()
        gradients.clear()

        x, label = batch

        x = x.to(vqvae.device)
        label = label.to(vqvae.device)

        latents = vqvae.encode(x)
        quantized_latents = vqvae.quantize(latents)[0]

        mean_latent = torch.mean(quantized_latents, dim=[2,3,4])
        mean_latent = torch.abs(mean_latent)  # Take the absolute value of the mean latent
        mean_latent = mean_latent.view(1, 32, -1)

        detached_latent = mean_latent.detach().cpu().numpy()[:, features_to_keep, :].reshape(1, -1) 

        output = classifer.predict_proba(detached_latent)[:,1]

        shap_values = explainer(detached_latent)
        shap_values = np.array(shap_values.values).reshape(1, -1)

        # Clone the mean latent
        shap_weight_latent = mean_latent.clone()
        shap_weight_latent[:, :, :] = 0  # Zero out all values
        shap_weight_latent[:, features_to_keep, :] = torch.tensor(shap_values.reshape(1, -1, 1), device=shap_weight_latent.device, dtype=shap_weight_latent.dtype)  # Set SHAP values

        dot_product = torch.dot(mean_latent.view(-1), shap_weight_latent.view(-1))

        vqvae.zero_grad()
        dot_product.backward()

        import numpy as np

        weights = torch.mean(gradients[0], dim=[2,3,4]).view(1,32,1,1,1)
        heatmap = torch.sum(weights * activations[0], dim=1).squeeze()
        heatmap = heatmap.cpu().detach().numpy()
        heatmap /= np.max(np.abs(heatmap))

        upsampled_heamap = torch.nn.Upsample(size=(x.shape[2], x.shape[3], x.shape[4]), mode='trilinear')(torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)).squeeze()

        squeezed_img = x.squeeze().cpu().detach().numpy()

        latent_heatmaps.append(heatmap)
        heatmaps.append(upsampled_heamap)
        images.append(squeezed_img)
        preds.append(output)
        labels.append(label.cpu().detach().numpy())

    #Save the heatmaps and images
    latent_heatmaps = np.array(latent_heatmaps)
    heatmaps = np.array(heatmaps)
    images = np.array(images)
    preds = np.array(preds)
    labels = np.array(labels)

    latent_heatmaps_dir = os.path.join(grad_cam_dir, 'latent_heatmaps')
    heatmaps_dir = os.path.join(grad_cam_dir, 'heatmaps')
    images_dir = os.path.join(grad_cam_dir, 'images')

    if not os.path.exists(latent_heatmaps_dir):
        os.makedirs(latent_heatmaps_dir)

    if not os.path.exists(heatmaps_dir):
        os.makedirs(heatmaps_dir)

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for idx in tqdm(range(len(heatmaps)), desc='Saving Heatmaps and Images'):
        save_model_space_image(heatmaps[idx], os.path.join(heatmaps_dir, f'heatmap_{idx}.nii.gz'))
        save_model_space_image(images[idx], os.path.join(images_dir, f'image_{idx}.nii.gz'))

        latent_heatmap = latent_heatmaps[idx].squeeze()
        latent_heatmap_img = nib.Nifti1Image(latent_heatmap, np.eye(4))
        nib.save(latent_heatmap_img, os.path.join(latent_heatmaps_dir, f'latent_heatmap_{idx}.nii.gz'))

    #Save the predictions and labels
    np.save(os.path.join(grad_cam_dir, 'preds.npy'), preds)
    np.save(os.path.join(grad_cam_dir, 'labels.npy'), labels)

    registered_images_dir = os.path.join(grad_cam_dir, 'registered_images')
    registered_heatmaps_dir = os.path.join(grad_cam_dir, 'registered_heatmaps')
    registration_transform_dir = os.path.join(grad_cam_dir, 'registration_transforms')

    if not os.path.exists(registration_transform_dir):
        os.makedirs(registration_transform_dir)

    if not os.path.exists(registered_images_dir):
        os.makedirs(registered_images_dir)

    if not os.path.exists(registered_heatmaps_dir):
        os.makedirs(registered_heatmaps_dir)    

    #Register the images and heatmaps
    import ants
    mni_1mm_brain = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
    mni_1mm_brain = ants.image_read(mni_1mm_brain)

    for idx in tqdm(range(len(os.listdir(heatmaps_dir))), desc='Registering Images and Heatmaps'):
        image = ants.image_read(os.path.join(images_dir, f'image_{idx}.nii.gz'))
        heatmap = ants.image_read(os.path.join(heatmaps_dir, f'heatmap_{idx}.nii.gz'))

        registered_image = ants.registration(fixed=mni_1mm_brain, moving=image, type_of_transform='SyN')
        registered_heatmap = ants.apply_transforms(fixed=mni_1mm_brain, moving=heatmap, transformlist=registered_image['fwdtransforms'])

        ants.image_write(registered_image['warpedmovout'], os.path.join(registered_images_dir, f'registered_image_{idx}.nii.gz'))
        ants.image_write(registered_heatmap, os.path.join(registered_heatmaps_dir, f'registered_heatmap_{idx}.nii.gz'))

        for transform in registered_image['fwdtransforms']:
            if '.nii.gz' in transform:
                shutil.move(transform, os.path.join(registration_transform_dir, f'warp_{idx}.nii.gz'))
            elif '.mat' in transform:
                shutil.move(transform, os.path.join(registration_transform_dir, f'affine_{idx}.mat'))
            else:
                raise ValueError('Unknown transform type')
        for transform in registered_image['invtransforms']:
            if '.nii.gz' in transform:
                shutil.move(transform, os.path.join(registration_transform_dir, f'inv_warp_{idx}.nii.gz'))

    responders = np.where(labels == 1)[0]
    non_responders = np.where(labels == 0)[0]

    responder_images = np.zeros(mni_1mm_brain.shape)
    for idx in tqdm(responders, desc='Averaging Responder Images'):
        image = ants.image_read(os.path.join(registered_heatmaps_dir, f'registered_heatmap_{idx}.nii.gz'))
        responder_images += image.numpy()

    responder_images /= len(responders)

    non_responder_images = np.zeros(mni_1mm_brain.shape)
    for idx in tqdm(non_responders, desc='Averaging Non-Responder Images'):
        image = ants.image_read(os.path.join(registered_heatmaps_dir, f'registered_heatmap_{idx}.nii.gz'))
        non_responder_images += image.numpy()

    non_responder_images /= len(non_responders)

    responder_average = ants.from_numpy(responder_images, origin=mni_1mm_brain.origin, spacing=mni_1mm_brain.spacing, direction=mni_1mm_brain.direction)
    non_responder_average = ants.from_numpy(non_responder_images, origin=mni_1mm_brain.origin, spacing=mni_1mm_brain.spacing, direction=mni_1mm_brain.direction)

    ants.image_write(responder_average, os.path.join(grad_cam_dir, 'responder_average.nii.gz'))
    ants.image_write(non_responder_average, os.path.join(grad_cam_dir, 'non_responder_average.nii.gz'))

    all_subjects_average = np.zeros(mni_1mm_brain.shape)

    for idx in tqdm(range(len(os.listdir(registered_heatmaps_dir))), desc='Averaging All Subjects Images'):
        image = ants.image_read(os.path.join(registered_heatmaps_dir, f'registered_heatmap_{idx}.nii.gz'))
        all_subjects_average += image.numpy()

    all_subjects_average /= len(os.listdir(registered_heatmaps_dir))
    all_subjects_average = ants.from_numpy(all_subjects_average, origin=mni_1mm_brain.origin, spacing=mni_1mm_brain.spacing, direction=mni_1mm_brain.direction)
    ants.image_write(all_subjects_average, os.path.join(grad_cam_dir, 'all_subjects_average.nii.gz'))

    all_subjects_abs_average = np.abs(all_subjects_average.numpy())
    all_subjects_abs_average_img = ants.from_numpy(all_subjects_abs_average, origin=mni_1mm_brain.origin, spacing=mni_1mm_brain.spacing, direction=mni_1mm_brain.direction)
    ants.image_write(all_subjects_abs_average_img, os.path.join(grad_cam_dir, 'all_subjects_abs_average.nii.gz'))

    all_subjects_abs_average_masked = all_subjects_abs_average * (mni_1mm_brain.numpy() > 0).astype(np.float32)
    all_subjects_abs_average_masked_img = ants.from_numpy(all_subjects_abs_average_masked, origin=mni_1mm_brain.origin, spacing=mni_1mm_brain.spacing, direction=mni_1mm_brain.direction)
    ants.image_write(all_subjects_abs_average_masked_img, os.path.join(grad_cam_dir, 'all_subjects_abs_average_masked.nii.gz'))

def median_decoding(config):

    '''
    This function decodes and saves the median responder and non-responder images for contrastive analysis.
    '''

    images_dir = str(config['images_dir'])
    intermediates_dir = str(config['intermediates_dir'])

    if not os.path.exists(intermediates_dir):
        os.makedirs(intermediates_dir)

    train_df = pd.read_csv(os.path.join(config['dataframes_folder'], 'train_df.csv'))
    val_df = pd.read_csv(os.path.join(config['dataframes_folder'], 'val_df.csv'))
    test_df = pd.read_csv(os.path.join(config['dataframes_folder'], 'test_df.csv'))

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    df = pd.concat([train_df, val_df, test_df], axis=0)

    if not 'outcome' in df.columns:
        raise ValueError('The metadata file must contain an outcome column. Since this is a classification task')

    # Define universal params
    batch_size = int(config['batch_size'])
    num_workers = int(config['num_workers'])

    # Set up the data module
    use_clinical = bool(config['use_clinical'])
    use_train_transform = bool(config['use_train_transform'])
    preload = bool(config['preload'])
    image_dim = tuple(map(int, config['image_dim']))
    outcome_col = str(config['outcome_col'])

    # Check if the debugger is on, if so make num_workers = 1
    if DEBUGGING:
        num_workers = 1

    # Load the checkpoint
    vqvae = BrainVQVAE.load_from_checkpoint(config['vqvae_checkpoint_path'])
    vqvae.intermediates_dir = intermediates_dir

    # Load the SVM classifier
    classifier = pickle.load(open(config['svm_path'], 'rb'))
    features_to_keep = np.load(config['svm_feature_indices'])

    data_module = ImagingDataModule(
        images_dir=images_dir,
        metadata_df=df,
        train_ids=[],
        val_ids=df[df['split'] == 'train']['study_id'].values,
        test_ids=df['study_id'].values,
        batch_size=1,
        num_workers=num_workers,
        use_clinical=use_clinical,
        use_train_transform=use_train_transform,
        preload=preload,
        crop_or_pad_dim=image_dim,
        outcome_col=outcome_col
    )

    # Get the dataloaders
    data_module.setup()
    train_loader = data_module.val_dataloader()  # 

    train_mean_latent_predictions = []
    encoded_volumes = []
    train_labels = []

    grad_cam_dir = os.path.join(intermediates_dir, 'gradcam')
    if not os.path.exists(grad_cam_dir):
        os.makedirs(grad_cam_dir)

    with torch.no_grad():
        # Run all the training data (in the validation data loader) to get the median of each feature
        for batch in tqdm(train_loader, desc='Generating predictions for training data', total=len(train_loader)):
            x, label = batch

            x = x.to(vqvae.device)
            label = label.to(vqvae.device)

            latents = vqvae.encode(x)
            quantized_latents = vqvae.quantize(latents)[0]

            mean_latent = torch.mean(quantized_latents, dim=[2,3,4])
            mean_latent = mean_latent.view(1, 32, -1)

            detached_latent = mean_latent.detach().cpu().numpy().reshape(-1)[features_to_keep]

            train_mean_latent_predictions.append(detached_latent)
            encoded_volumes.append(latents.detach())
            train_labels.append(label.detach())

    # Make it a numpy array
    train_mean_latent_predictions = torch.tensor(train_mean_latent_predictions)
    train_labels = torch.tensor(train_labels)

    # Generate mean responder and non-responder latents
    encoded_volumes = torch.stack(encoded_volumes).squeeze()
    responder_indices = torch.where(train_labels == 1)[0]
    non_responder_indices = torch.where(train_labels == 0)[0]

    responder_encoded = encoded_volumes[responder_indices]
    non_responder_encoded = encoded_volumes[non_responder_indices]

    median_responder_encoded = torch.median(responder_encoded, dim=0).values.unsqueeze(0)
    median_non_responder_encoded = torch.median(non_responder_encoded, dim=0).values.unsqueeze(0)

    median_responder_quantized = vqvae.quantize(median_responder_encoded)[0]
    median_non_responder_quantized = vqvae.quantize(median_non_responder_encoded)[0]

    responder_decoded = vqvae.decode(median_responder_quantized)
    non_responder_decoded = vqvae.decode(median_non_responder_quantized)

    save_model_space_image(responder_decoded.squeeze().cpu().detach().numpy(), os.path.join(grad_cam_dir, 'median_responder_image.nii.gz'))
    save_model_space_image(non_responder_decoded.squeeze().cpu().detach().numpy(), os.path.join(grad_cam_dir, 'median_non_responder_image.nii.gz'))

def main():
    parser = argparse.ArgumentParser(description='Evaluate SVM, run Grad-CAM analysis, or median decoding for VQ-VAE model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--svm_path', type=str, help='Path to the svm checkpoint', required=False)
    parser.add_argument('--svm_feature_indices', type=str, help='Path to the svm feature indices', required=False)
    parser.add_argument('--dataframes_folder', type=str, help='Path to the dataframes folder with train.df, val.df, test.df', required=False)
    parser.add_argument('--mode', type=str, choices=['eval', 'gradcam', 'median_decoding'], default='eval', help="Which mode to run: 'eval' for evaluation, 'gradcam' for Grad-CAM analysis, 'median_decoding' to save median responder/non-responder images")
    # Accept any additional arguments
    parser.add_argument('args', nargs=argparse.REMAINDER)

    args, unknown = parser.parse_known_args()

    config_file = args.config
    # Load the config file
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Override config with command-line arguments if provided
    if args.svm_path:
        config['svm_path'] = args.svm_path
    if args.svm_feature_indices:
        config['svm_feature_indices'] = args.svm_feature_indices
    if args.dataframes_folder:
        config['dataframes_folder'] = args.dataframes_folder

    # Copy the config file to the intermediates directory
    intermediates_dir = str(config['intermediates_dir'])
    if not os.path.exists(intermediates_dir):
        os.makedirs(intermediates_dir)
    shutil.copy(config_file, os.path.join(intermediates_dir, 'config.yaml'))

    if args.mode == 'eval':
        eval(config)
    elif args.mode == 'gradcam':
        run_gradcam_analysis(config)
    elif args.mode == 'median_decoding':
        median_decoding(config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()
