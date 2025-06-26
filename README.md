# VQVNS

**This is the official code repository for the project:**

**Predicting Response to Vagus Nerve Stimulation Using Deep Representation Learning**  
Hrishikesh Suresh MD, *et al.*, George M Ibrahim MD PhD FRCSC

VQ-VAE based predictive modelling of VNS Response

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Data Preparation](#data-preparation)
- [Training the VQ-VAE](#training-the-vq-vae)
- [Using the Trained VQ-VAE](#using-the-trained-vq-vae)
- [Training the SVM Classifier](#training-the-svm-classifier)
- [Evaluating & Using the SVM](#evaluating--using-the-svm)
- [Utilities & Preprocessing](#utilities--preprocessing)
- [Weights & Checkpoints](#weights--checkpoints)
- [Single Subject Inference](#single-subject-inference)

---

## Overview
This repository provides a framework for predictive modeling of VNS (Vagus Nerve Stimulation) response using a Vector Quantized Variational Autoencoder (VQ-VAE) and an SVM classifier. It includes scripts for data preprocessing, model training, evaluation, and utilities for MRI intensity normalization.

## Repository Structure
- `configs/` — YAML configuration files for VQ-VAE and SVM training/evaluation
- `datasets/` — Data loading and preprocessing modules
- `evaluation/` — Scripts for evaluating reconstructions and classifier performance
- `models/` — Model definitions (VQ-VAE, quantizers, etc.)
- `training/` — Training scripts for VQ-VAE and SVM
- `utils/` — Utility scripts (metrics, loss functions, normalization, etc.)
- `weights/` — Directory for saving trained model checkpoints
- `preprocessing/` — Scripts for MRI preprocessing

## Setup & Installation
1. **Clone the repository**
 ```bash
   git clone <repo_url>
   cd VQVNS
 ```
2. **Install dependencies using uv and pyproject.toml**
  Install all dependencies directly from `pyproject.toml`:
 ```bash
   uv pip install -r pyproject.toml
 ```
   > **Note:** [uv](https://github.com/astral-sh/uv) is a fast Python package installer. Install it with `pip install uv` if not already available.

## Data Preparation

**Preprocessing your MRI data is required before training or inference.**

The local preprocessing pipeline is intended to be run locally

### 1. Run Preprocessing Locally (Skip HPC Script if Processing Locally)

- Use the `utils/preprocessing/generic_t1_preprocess.py` script to preprocess your MRI data.
- Specify the following when running the script:
  - Path to the input MRI images
  - Output directory for preprocessed files
  - Path to the MNI 1mm brain reference template (included with FSL v6)
- Example usage:
  ```bash
  python utils/preprocessing/generic_t1_preprocess.py \
    --input_dir path/to/raw_mri/ \
    --output_dir path/to/preprocessed_mri/ \
    --mni_template path/to/MNI152_T1_1mm_brain.nii.gz
  ```
- Ensure you have FSL v6 installed to access the required MNI template.

---

> **Note:** All downstream training and inference scripts expect preprocessed and cropped MRI files as input. Do not use raw MRI data directly.

## Single Subject Inference
To run inference on a single subject using a preprocessed and cropped MRI, use the `infer_single_subject.py` script. **The input MRI must be processed with the provided pipeline and should be the cropped output.**
Download weights file [here](https://utoronto-my.sharepoint.com/:u:/g/personal/h_suresh_mail_utoronto_ca/Ed8sZqTEh0tKtdi39W3jPUwBXRjEWr079Ouiq98R8edgAw?e=UPQGbS)
Please open an issue if link does not work as OneDrive links expire every 30 days. 

> We are unable to provide images to test against as confidential patient data cannot be shared. 

Example usage:
```bash
python infer_single_subject.py \
  --mri path/to/subject_cropped.nii.gz \
  --vqvae_checkpoint weights/vqvae.ckpt \
  --svm_weights weights/svm_weights.pkl \
  --features_to_keep weights/features_to_keep.npy
```
- The script will automatically pad or crop the MRI to 176x208x176 if needed.
- The output will be a prediction for the subject.

> Ensure your input MRI is preprocessed and cropped using the project's pipeline before running inference.

---

## Training the VQ-VAE
1. **Configure training**
   - Edit or use an existing config in `configs/vqvae/` (e.g., `vqvae_healthy_gradnorm_hpc.yaml`).
2. **Start training**
```bash
python training/train_vqvae.py --config configs/vqvae/vqvae_healthy_gradnorm_hpc.yaml
```
- Checkpoint and intermediate file locations are specified in the config file. 

## Using the Trained VQ-VAE

- To evaluate or reconstruct using a trained VQ-VAE:
Download weights file [here](https://utoronto-my.sharepoint.com/:u:/g/personal/h_suresh_mail_utoronto_ca/Ed8sZqTEh0tKtdi39W3jPUwBXRjEWr079Ouiq98R8edgAw?e=UPQGbS)
```bash
python evaluation/eval_vqvae.py --config configs/vqvae/vqvae_healthy_gradnorm_hpc.yaml --checkpoint weights/vqvae.ckpt
```
- To evaluate reconstruction fidelity:
```bash
python evaluation/eval_recon_fidelity.py --config configs/vqvae/vqvae_healthy_gradnorm_hpc.yaml --checkpoint weights/vqvae.ckpt
```

## Training the SVM Classifier
1. **Configure training**
   - Edit or use an existing config in `configs/classifier/` (e.g., `vns_svm.yaml`).
2. **Start training**
```bash
python training/train_svm_classifier.py --config configs/classifier/vns_svm.yaml
```
   - The classifier will be trained on VQ-VAE latent codes or other features as specified.

## Evaluating & Using the SVM
- To evaluate the SVM classifier:
```bash
python evaluation/eval_svm.py --config configs/classifier/vns_svm.yaml
```
- Adjust the config file to point to the correct feature and label files as needed.

## Utilities & Preprocessing
- **MRI Intensity Normalization:**
  - Utilities for normalization are in `utils/packages/intensity-normalization/`.
  - See the [official documentation](utils/packages/intensity-normalization/docs/index.rst) for usage.
- **Other utilities:**
  - `utils/metrics.py`, `utils/loss_functions.py`, etc., provide supporting functions for training and evaluation.

## Weights & Checkpoints
- Trained model weights and checkpoints are saved in the `weights/` directory.
- Use these checkpoints for evaluation or further training.
- For SVM evaluation and prediction, ensure the following files are present in `weights/`:
  - `svm_weights.pkl`: Trained SVM model weights
  - `features_to_keep.npy`: Numpy array specifying the features to use for SVM classification
- Update your config files to reference these files as needed for SVM evaluation and prediction.

For further details, refer to the code and configuration files in each directory. For questions or issues, please open an issue in the repository.
