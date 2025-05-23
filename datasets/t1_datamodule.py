import os
import torch
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchio as tio
import monai.transforms as transforms
import sys
from utils.misc_functions import save_np_as_nifti
from tqdm import tqdm

class ImagingDataset(Dataset):
    def __init__(self, image_dir, metadata_df, study_ids, transform=None, use_clinical=False, preload=False, outcome_col='outcome', return_study_id=False):
        self.image_dir = image_dir
        self.metadata = metadata_df
        self.metadata = self.metadata[self.metadata['study_id'].isin(study_ids)]
        self.study_ids = self.metadata['study_id'].tolist()
        self.transform = transform
        self.use_clinical = use_clinical
        self.clinical_features = self.metadata.drop(columns=['study_id', outcome_col]).values if use_clinical else None
        self.outcomes = self.metadata[outcome_col].values

        self.preload = preload
        self.images = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_study_id = return_study_id

        if preload:
            for study_id in tqdm(self.study_ids, desc=f"Loading images from {image_dir}"):
                img_path = os.path.join(self.image_dir, f"{study_id}.nii.gz")
                image = self.load_subject(img_path)
                self.images.append(image)

    def __len__(self):
        return len(self.study_ids)

    def load_subject(self, filepath):
        img = nib.load(filepath)
        data = img.get_fdata()
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

        return data

    def __getitem__(self, idx):
        subject_id = self.study_ids[idx]
        
        if self.preload:
            image = self.images[idx]
        else:
            img_path = os.path.join(self.image_dir, f"{subject_id}.nii.gz")
            image = self.load_subject(img_path)

        if self.transform:
            image = self.transform(image)

        # Handle clinical variables if used
        if self.use_clinical:
            clinical_data = torch.tensor(self.clinical_features[idx], dtype=torch.float32)
            return image, clinical_data, torch.tensor(self.outcomes[idx], dtype=torch.float32)
        else:
            if self.return_study_id:
                return image, torch.tensor(self.outcomes[idx], dtype=torch.float32), subject_id 
            else:
                return image, torch.tensor(self.outcomes[idx], dtype=torch.float32)

class ImagingDataModule(pl.LightningDataModule):
    def __init__(self, 
                 images_dir, 
                 metadata_df, 
                 train_ids, 
                 val_ids, 
                 test_ids, 
                 crop_or_pad_dim, 
                 batch_size=8, 
                 num_workers=4, 
                 use_clinical=True, 
                 use_train_transform=True, 
                 num_channels=1, 
                 preload=False, 
                 testing_mode=False, 
                 augment_on_test=False,
                 outcome_col='outcome'):
        
        super().__init__()
        self.images_dir = images_dir
        self.metadata_df = metadata_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_clinical = use_clinical
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.use_train_transform = use_train_transform
        self.num_channels = num_channels
        self.preload = preload
        self.crop_or_pad_dim = crop_or_pad_dim
        self.testing_mode = testing_mode
        self.augment_on_test = augment_on_test
        self.outcome_col = outcome_col

    def setup(self, stage=None):

        #Define the transforms we need include adding the channel dimension
        # Define the transform pipeline using self.crop_or_pad_dim
        self.train_transform = transforms.Compose([
                                    # Apply random augmentations with 80% probability
                                    transforms.OneOf([
                                        transforms.Compose([
                                            #transforms.RandFlip(prob=0.5, spatial_axis=0),  # Flip along the x-axis
                                            transforms.RandBiasField(prob=0.5),  # Random bias field correction
                                            transforms.RandGaussianNoise(prob=0.5, mean=0.0, std=0.02),  # Random noise
                                            transforms.RandAdjustContrast(prob=0.5, gamma=(0.99, 1.01)),  # Random contrast adjustment
                                            transforms.RandShiftIntensity(prob=0.5, offsets=0.05),  # Random intensity shifts
                                            transforms.RandAffine(prob=0.2,  # Affine transformations (20% probability)
                                                                rotate_range=(0.04, 0.04, 0.04), 
                                                                translate_range=(2, 2, 2), 
                                                                scale_range=(0.00, 0.00, 0.00)),
                                        ]),
                                        transforms.Identity()  # No-op with 20% probability
                                    ], weights=[0.8, 0.2]),

                                    # Crop or pad to match the expected dimensions (using self.crop_or_pad_dim)
                                    transforms.SpatialPad(spatial_size=self.crop_or_pad_dim),

                                    # Intensity thresholding
                                    #transforms.ScaleIntensity(minv=0, maxv=1),

                                ])


        self.val_test_transform = transforms.Compose([
                                    # Crop or pad to match the expected dimensions (using self.crop_or_pad_dim)
                                    transforms.SpatialPad(spatial_size=self.crop_or_pad_dim),

                                    # Intensity thresholding
                                    #transforms.ScaleIntensity(minv=0, maxv=1),

        ])

        if not self.testing_mode: ## Only load the testing dataset when in evaluation mode
            if self.use_train_transform:
                self.train_dataset = ImagingDataset(self.images_dir, self.metadata_df, self.train_ids, use_clinical=self.use_clinical, transform=self.train_transform, preload=self.preload, outcome_col=self.outcome_col)
            else:
                self.train_dataset = ImagingDataset(self.images_dir, self.metadata_df, self.train_ids, use_clinical=self.use_clinical, transform=self.val_test_transform, preload=self.preload, outcome_col=self.outcome_col)

            self.val_dataset = ImagingDataset(self.images_dir, self.metadata_df, self.val_ids, use_clinical=self.use_clinical, transform=self.val_test_transform, preload=self.preload, outcome_col=self.outcome_col)
            self.test_dataset = ImagingDataset(self.images_dir, self.metadata_df, self.test_ids, use_clinical=self.use_clinical, transform=self.val_test_transform, preload=self.preload, outcome_col=self.outcome_col)
        else:
            if self.augment_on_test:
                self.test_dataset = ImagingDataset(self.images_dir, self.metadata_df, self.test_ids, use_clinical=self.use_clinical, transform=self.train_transform, preload=self.preload, outcome_col=self.outcome_col)
            else:
                self.test_dataset = ImagingDataset(self.images_dir, self.metadata_df, self.test_ids, use_clinical=self.use_clinical, transform=self.val_test_transform, preload=self.preload, outcome_col=self.outcome_col)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

