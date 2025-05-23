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

The preprocessing pipeline is designed for use with both local and HPC (High Performance Computing) environments. It consists of three main stages:

### 1. Preparation for HPC (Prep Mode)
This step generates the commands and scripts needed to run the main preprocessing on an HPC cluster.

**Run prep mode locally:**
```bash
python preprocessing/t1_processing_hpc.py \
  --prep_mode True \
  --input_dir /path/to/raw_mri/ \
  --hpc_output_dir /path/to/hpc_job_dir/ \
  --output_dir /path/to/processed_mri/ \
  --reference_image /path/to/reference_image.nii.gz
```
- `--input_dir`: Directory containing your raw T1 MRI files.
- `--hpc_output_dir`: Directory where the script will write the generated HPC commands and scripts.
- `--output_dir`: Directory where processed files will be written by the HPC jobs.
- `--reference_image`: Path to a reference image for registration.

This will create a `commands.txt` file in your `hpc_output_dir` with all the commands needed for batch processing on the cluster. It will also copy the script itself to the HPC directory.

### 2. Run Preprocessing on HPC
- Transfer the contents of your `hpc_output_dir` (including the script and `commands.txt`) to your HPC environment.
- Submit jobs to your cluster using the generated `commands.txt`. You may use a shell script (e.g., `t1_hpc_run.sh`) or your cluster's job submission system to run these commands in parallel.
- Each command will process a batch of MRI files, performing:
  - Brain extraction (hybrid mask using SynthStrip and SynthSeg)
  - Reorientation, resampling, intensity normalization
  - Registration to template spaces (MNI, DBM)
  - Cropping to nonzero brain region
  - Output of cropped images and CSVs with crop dimensions

### 3. Post-HPC Processing (Combine Crop Info)
After all jobs have finished and you have all the cropped CSVs from the HPC output:

**Run the post-HPC script locally:**
```bash
python preprocessing/post_hpc_processing.py
```
- Edit the script to set `csvs_path` to the directory containing all the per-image crop CSVs (produced in the previous step).
- Set `output_path` to where you want the combined CSV written.
- The script will:
  - Combine all crop CSVs into one file
  - Compute and print the minimum image size needed to cover all crops
  - Write this info to a text file

---

**Summary of Steps:**
1. Run `t1_processing_hpc.py` in prep mode locally to generate HPC commands.
2. Transfer the generated scripts and commands to your HPC and run the jobs (using `commands.txt`).
3. After all jobs finish, run `post_hpc_processing.py` locally to combine crop info and determine the minimum image size.

> **Note:** All downstream training and inference scripts expect preprocessed and cropped MRI files as input. Do not use raw MRI data directly.

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
Download weights file [here](https://utoronto-my.sharepoint.com/:u:/g/personal/h_suresh_mail_utoronto_ca/Ed8sZqTEh0tKtdi39W3jPUwB9Onz8im11893As9kHBonTQ?e=gXlGef)
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

## Single Subject Inference
To run inference on a single subject using a preprocessed and cropped MRI, use the `infer_single_subject.py` script. **The input MRI must be processed with the provided pipeline and should be the cropped output.**

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

For further details, refer to the code and configuration files in each directory. For questions or issues, please open an issue in the repository.
