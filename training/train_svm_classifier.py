##### This script will take the latents of the VQVAE and then try to train an XGBoost classifier on them to predict the outcome

#### Hrishikesh Suresh
#### Ibrahim Lab 2025

# Import necessary libraries
import sys
sys.path.insert(0, '/d/gmi/1/hrishikeshsuresh/vns_deep_learning/')
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.vqvae import BrainVQVAE
from datasets.t1_datamodule import ImagingDataModule
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from tqdm import tqdm
import yaml
import nibabel as nib
import argparse
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
from hyperopt.mongoexp import MongoTrials
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import shutil


 #Check if running in interactive mode
if hasattr(sys, 'ps1'):
    INTERACTIVE = True
else:
    INTERACTIVE = False

if not INTERACTIVE:
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train an SVM model on the latents of a VQVAE model')
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    args = parser.parse_args()
    config_path = args.config
else:
    config_path = '/d/gmi/1/hrishikeshsuresh/vns_deep_learning/configs/classifier/tle_svm.yaml'

# Load the config file
with open(config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

images_dir = str(config['images_dir'])
metadata_csv = str(config['metadata_csv'])
intermediates_dir= str(config['intermediates_dir'])
vqvae_checkpoint_path = str(config['vqvae_checkpoint_path'])
use_2080_outcomes = config['use_2080_outcomes']
no_equal = config['no_equal']
outcome_col = str(config['outcome_col'])

affine_to_use = np.eye(4)
affine_to_use[0, 0] = -1

if not os.path.exists(intermediates_dir):
    os.makedirs(intermediates_dir)

#Copy the config file to the intermediates directory
config_file_name = os.path.basename(config_path)
config_output_path = os.path.join(intermediates_dir, config_file_name)
shutil.copyfile(config_path, config_output_path)

#Get the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Run the data through the VQVAE and get the latents
def load_data(recreate_data=False, total_non_train_ratio=0.2):
    latent_full_dataset_path = os.path.join(intermediates_dir, 'latent_full_dataset.npz')

    if not os.path.exists(latent_full_dataset_path) or recreate_data:
        print('Latent dataset not found. Generating it now')

        # Load the metadata
        df = pd.read_csv(metadata_csv)

        if not outcome_col in df.columns:
            raise ValueError('The metadata file must contain an outcome column. Since this is a classification task')

        #Keep agreement only as needed
        agreement_only = bool(config['agreement_only'])
        if agreement_only:
            if 'agreement' not in df.columns:
                raise ValueError('The metadata file must contain an agreement column when using this option. This only applies if using the VNS dataset')
            df = df[df['agreement'] == 1]
        else:
            # We have to use the long outcome column so where long_outcome is not blank, we replace outcome with long_outcome
            if 'long_outcome' in df.columns:
                df['outcome'] = df['outcome'].where(df['long_outcome'].isna(), df['long_outcome'])
            else:
                print('The metadata file does not contain a long_outcome column. This only applies if using the VNS dataset')
                print('Continuing with the outcome column. Make sure this is correct')

        outcomes_already_binary = False
        if len(np.unique(df[outcome_col])) != 2:
            if 'binarization_threshold' in config:
                    df[outcome_col] = df[outcome_col].apply(lambda x: x if x >= 0 else 0) #Zero out negative outcomes
                    df['raw_outcome'] = df[outcome_col].copy()
                    df[outcome_col] = df[outcome_col].apply(lambda x: 1 if x > float(config['binarization_threshold']) else 0)
                    #Make sure raw_outcome is between 0 and 1, and not 0 and 100 
                    df['raw_outcome'] = (df['raw_outcome'] / 100)
            else:
                raise ValueError('The outcome column must be binary. Please binarize the outcome columnor provide a binarization threshold in the config file')
        else:
            outcomes_already_binary = True
            print('The outcome column is already binary. No need to binarize. Proceeding...')

        if use_2080_outcomes and not outcomes_already_binary:
            if no_equal:
                df = df[(df['raw_outcome'] < 0.2) | (df['raw_outcome'] > 0.8)]
            else:
                df = df[(df['raw_outcome'] <= 0.2) | (df['raw_outcome'] >= 0.8)]

        #Let's split the data into training and validation sets
        train_ids, test_ids = train_test_split(df['study_id'].values, test_size=total_non_train_ratio/2, random_state=22, stratify=df['outcome'].values)

        train_df = df[df['study_id'].isin(train_ids)]
        test_df = df[df['study_id'].isin(test_ids)]

        #Print response fraction in each dataset
        print(f"Intermediate Train response fraction: {train_df['outcome'].mean() * 100}")
        print(f"Intermediate Test response fraction: {test_df['outcome'].mean() * 100}")

        val_ratio = (total_non_train_ratio/2) / (1 - total_non_train_ratio/2)

        train_ids, val_ids = train_test_split(train_df['study_id'].values, test_size=val_ratio, random_state=22, stratify=train_df['outcome'].values)   

        train_df = df[df['study_id'].isin(train_ids)]
        val_df = df[df['study_id'].isin(val_ids)]

        #Print response fraction in each dataset
        print(f"Final Train response fraction: {train_df['outcome'].mean() * 100}")
        print(f"Final Val response fraction: {val_df['outcome'].mean() * 100}")
        print(f"Final Test response fraction: {test_df['outcome'].mean() * 100}")

        #Dataset sizes
        print(f"Train size: {len(train_df)}")
        print(f"Val size: {len(val_df)}")
        print(f"Test size: {len(test_df)}")

        #Save the dataframes to the intermediates directory
        train_df.to_csv(os.path.join(intermediates_dir, 'train_df.csv'), index=False)
        val_df.to_csv(os.path.join(intermediates_dir, 'val_df.csv'), index=False)
        test_df.to_csv(os.path.join(intermediates_dir, 'test_df.csv'), index=False)

        # Load the VQVAE model
        vqvae = BrainVQVAE.load_from_checkpoint(vqvae_checkpoint_path)
        vqvae.eval()
        vqvae.to(device)

        #Set up the data module
        use_clinical = bool(config['use_clinical'])
        use_train_transform = bool(config['use_train_transform'])
        preload = bool(config['preload'])
        image_dim = tuple(map(int, config['image_dim']))

        #Load the datamodule 
        data_module = ImagingDataModule(images_dir = images_dir, 
                                        metadata_df = df, 
                                        train_ids = train_ids, 
                                        val_ids = val_ids,  
                                        test_ids = test_ids,
                                        batch_size = 1, 
                                        num_workers = 1,
                                        use_clinical = use_clinical,
                                        use_train_transform = use_train_transform,
                                        preload = preload,
                                        crop_or_pad_dim=image_dim,
                                        outcome_col=outcome_col)
        data_module.setup()

        #Get dataloaders 
        train_loader = data_module.train_dataloader()
        train_loader.shuffle = False
        val_loader = data_module.val_dataloader()
        val_loader.shuffle = False
        test_loader = data_module.test_dataloader()
        test_loader.shuffle = False

        def process_latents(loader, model, loader_name):
            latents = []
            labels = []
            for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Processing {loader_name} dataloader'):
                x, y = batch
                x = x.to(model.device)
                y = y.to(model.device)
                with torch.no_grad():
                    latents.append(model.encode_and_quantize(x).cpu().numpy())
                    labels.append(y.cpu().numpy())
            latents = np.concatenate(latents, axis=0)
            labels = np.concatenate(labels, axis=0)
            return latents, labels

        train_latents, train_labels = process_latents(train_loader, vqvae, 'train')
        val_latents, val_labels = process_latents(val_loader, vqvae, 'val')
        test_latents, test_labels = process_latents(test_loader, vqvae, 'test')

        latent_data = {
            'train_latents': train_latents,
            'train_labels': train_labels,
            'val_latents': val_latents,
            'val_labels': val_labels,
            'test_latents': test_latents,
            'test_labels': test_labels,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df
        }

        with open(latent_full_dataset_path, 'wb') as f:
            pickle.dump(latent_data, f)

    else:
        print('Latent dataset found. Loading it now')
        with open(latent_full_dataset_path, 'rb') as f:
            latent_data = pickle.load(f)
        
        train_latents = latent_data['train_latents']
        train_labels = latent_data['train_labels']
        val_latents = latent_data['val_latents']
        val_labels = latent_data['val_labels']
        test_latents = latent_data['test_latents']
        test_labels = latent_data['test_labels']
        train_df = latent_data['train_df']
        val_df = latent_data['val_df']
        test_df = latent_data['test_df']

    return train_latents, train_labels, val_latents, val_labels, test_latents, test_labels, train_df, val_df, test_df

#Plot it all nicely 
def plot_curve_and_matrix(preds, probas, labels, dataset_name):
    fpr, tpr, thresholds = roc_curve(labels, probas)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt='d', ax=ax[0], cmap='Blues')
    ax[0].set_title(f'{dataset_name} Confusion Matrix')
    ax[0].set_xlabel('Predicted', fontsize=12)
    ax[0].set_ylabel('True', fontsize=12)
    ax[0].set_xticklabels(['Non-responder', 'Responder'])
    ax[0].set_yticklabels(['Non-responder', 'Responder'])

    auc = roc_auc_score(labels, probas)
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

def plot_auc_only(labels, probas, dataset_name):
    fpr, tpr, thresholds = roc_curve(labels, probas)
    fig, ax = plt.subplots(figsize=(4, 4))
    auc = roc_auc_score(labels, probas)
    ax.plot(fpr, tpr, label=f'AUC: {auc:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random Classifier')
    ax.set_title(f'{dataset_name} ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    #add legend for random classifier and the model
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(intermediates_dir, f'{dataset_name}_roc_curve.png'), dpi=300)
    plt.show()

train_latents, train_labels, val_latents, val_labels, test_latents, test_labels, train_df, val_df, test_df = load_data(recreate_data=True, 
                                                                                                                        total_non_train_ratio=0.4)

responder_count = np.sum(train_labels == 1)
non_responder_count = np.sum(train_labels == 0)
class_weight = non_responder_count / responder_count
print(f'Class weight: {class_weight}')

#Basic exploration of latent space
channels_in_latent_space = train_latents.shape[1]
print(f'Number of channels in latent space: {channels_in_latent_space}')

#Average the latents across the channels
train_latents_avg = train_latents.mean(axis=(2,3,4))
val_latents_avg = val_latents.mean(axis=(2,3,4))
test_latents_avg = test_latents.mean(axis=(2,3,4))

#Flatten
train_latents_flat = train_latents_avg.reshape(train_latents_avg.shape[0], -1)
val_latents_flat = val_latents_avg.reshape(val_latents_avg.shape[0], -1)
test_latents_flat = test_latents_avg.reshape(test_latents_avg.shape[0], -1)

#Responder and non-responder latents
responder_latents = train_latents_flat[train_labels == 1]
non_responder_latents = train_latents_flat[train_labels == 0]

train_latents_backup = train_latents_flat.copy()
val_latents_backup = val_latents_flat.copy()
test_latents_backup = test_latents_flat.copy()

train_latents_flat = train_latents_backup.copy()
val_latents_flat = val_latents_backup.copy()
test_latents_flat = test_latents_backup.copy()

# #Quick t-SNE plot
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# tsne_latents = tsne.fit_transform(train_latents_flat)

# fig, ax = plt.subplots(figsize=(12, 12))
# sns.scatterplot(x=tsne_latents[:, 0], y=tsne_latents[:, 1], hue=train_labels, ax=ax)
# ax.set_title('t-SNE Plot of Latents')
# plt.savefig(os.path.join(intermediates_dir, 'tsne_latents.png'))
# plt.show()

# #QQuick umap
# import umap

# umap_latents = umap.UMAP(n_components=2, random_state=42).fit_transform(train_latents_flat)

# for col in train_df.columns[1:-3]:
#     fig, ax = plt.subplots(figsize=(6, 6))
#     sns.scatterplot(x=umap_latents[:, 0], y=umap_latents[:, 1], hue=train_df[col], ax=ax)
#     ax.set_title(f'UMAP Plot of Latents Colored by {col}')
#     plt.savefig(os.path.join(intermediates_dir, 'umap_latents.png'))
#     plt.show()



#Quick test across latents 
channel_wise_test_df = pd.DataFrame()
sig_latents = 0
features_to_keep = []
sig_latent_dfs = []
for latent_idx in range(train_latents_flat.shape[1]):
    responder_values = responder_latents[:, latent_idx]
    non_responder_values = non_responder_latents[:, latent_idx]

    latent_df = pd.DataFrame()
    latent_df['values'] = np.concatenate([responder_values, non_responder_values])
    latent_df['labels'] = np.concatenate([np.ones_like(responder_values), np.zeros_like(non_responder_values)])

    #Do a t-test
    t, tp = ttest_ind(responder_values, non_responder_values)

    #Do a mann-whitney u test
    u, up = mannwhitneyu(responder_values, non_responder_values)

    #Quick fit a logistic regression model
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(train_latents_flat[:, latent_idx].reshape(-1, 1), train_labels)
    train_preds = lr.predict(train_latents_flat[:, latent_idx].reshape(-1, 1))
    val_preds = lr.predict(val_latents_flat[:, latent_idx].reshape(-1, 1))
    train_auc = roc_auc_score(train_labels, train_preds)
    val_auc = roc_auc_score(val_labels, val_preds)

    row = pd.DataFrame({
        'latent_idx': latent_idx,
        't': t,
        'tp': tp,
        'u': u,
        'up': up,
        'train_auc': train_auc,
        'val_auc': val_auc
    }, index=[0])

    if tp < 0.05:
        # fig, ax = plt.subplots(figsize=(6,6))
        # sns.boxplot(data=latent_df, x='labels', y='values', ax=ax, color='white')
        # sns.stripplot(data=latent_df, x='labels', y='values', ax=ax, hue='labels', dodge=False, alpha=0.5)
        # ax.set_title(f'Latent Index: {latent_idx}')
        # #Remove box plot background
        # ax.patch.set_alpha(0.0)
        sig_latents += 1
        features_to_keep.append(latent_idx)
        sig_latent_dfs.append(latent_df)

    channel_wise_test_df = pd.concat([channel_wise_test_df, row], axis=0)

#Save the channel wise test df
channel_wise_test_df.to_csv(os.path.join(intermediates_dir, 'channel_wise_test_df.csv'), index=False)

# Create a master grid based on the number of significant latent plots
n_sig = len(features_to_keep)
if n_sig == 0:
    print("No significant latent plots found.")
else:
    rows = int(np.ceil(np.sqrt(n_sig)))
    cols = int(np.ceil(n_sig / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    
    # Flatten axes array for easier indexing (handle when only one subplot exists)
    if n_sig == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each significant latent on its corresponding axis
    for i, (latent_idx, latent_df) in enumerate(zip(features_to_keep, sig_latent_dfs)):
        ax = axes[i]
        sns.boxplot(data=latent_df, x='labels', y='values', ax=ax, color='white')
        sns.stripplot(data=latent_df, x='labels', y='values', ax=ax, hue='labels', dodge=False, alpha=0.5)
        ax.set_title(f'Latent Index: {latent_idx}')
        ax.patch.set_alpha(0.0)
        # Optionally remove the legend if not needed
        ax.get_legend().remove() if ax.get_legend() else None

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(intermediates_dir, 'significant_latents.png'))
    plt.show()

print(f'Significant latents: {sig_latents}')

#Save the features to keep 
features_to_keep = np.array(features_to_keep)
np.save(os.path.join(intermediates_dir, 'features_to_keep.npy'), features_to_keep)


train_latents_flat = train_latents_flat[:, features_to_keep]
val_latents_flat = val_latents_flat[:, features_to_keep]
test_latents_flat = test_latents_flat[:, features_to_keep]

#Simple SVM model to predict the outcome
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

#Fit the model
svm_X_train = train_latents_flat
svm_X_val = val_latents_flat
svm_X_test = test_latents_flat
print('Ready to start training the SVM model')

#Train the model 
svm = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42, C=5000, verbose=False)
svm.fit(svm_X_train, train_labels)

#Predict on the training and validation sets
train_preds = svm.predict(svm_X_train)
val_preds = svm.predict(svm_X_val)
train_probs = svm.predict_proba(svm_X_train)[:, 1]
val_probs = svm.predict_proba(svm_X_val)[:, 1]

#Save these
train_preds_path = os.path.join(intermediates_dir, 'train_preds.npy')
val_preds_path = os.path.join(intermediates_dir, 'val_preds.npy')
train_probs_path = os.path.join(intermediates_dir, 'train_probs.npy')
val_probs_path = os.path.join(intermediates_dir, 'val_probs.npy')
train_labels_path = os.path.join(intermediates_dir, 'train_labels.npy')
val_labels_path = os.path.join(intermediates_dir, 'val_labels.npy')
np.save(train_preds_path, train_preds)
np.save(val_preds_path, val_preds)
np.save(train_probs_path, train_probs)
np.save(val_probs_path, val_probs)
np.save(train_labels_path, train_labels)
np.save(val_labels_path, val_labels)

#Plot the confusion matrix and ROC curve
train_auc, train_thresh = plot_curve_and_matrix(train_preds, train_probs, train_labels, 'Train')
val_auc, val_thresh = plot_curve_and_matrix(val_preds, val_probs, val_labels, 'Val')

def permute_and_plot_auc(labels, probs, true_auc, n_permutations=1000, dataset_name='Train'):
    #Label permutation testing
    null_aucs = []
    tprs = []
    fprs = []
    for i in tqdm(range(n_permutations)):
        #Shuffle the labels
        permuted_labels = np.random.permutation(labels)
        auc = roc_auc_score(permuted_labels, probs)
        null_aucs.append(auc)
        fpr, tpr, thresholds = roc_curve(permuted_labels, probs)
        tprs.append(tpr)
        fprs.append(fpr)

    null_aucs = np.array(null_aucs)
    p = np.sum(null_aucs >= true_auc) / len(null_aucs)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(null_aucs, bins=50, ax=ax, color='blue', alpha=0.5)
    ax.set_title('Null Distribution of AUCs')
    ax.axvline(true_auc, color='red', linestyle='--')
    ax.axvline(np.percentile(null_aucs, 95), color='green', linestyle='--')
    ax.legend([f'{dataset_name} AUC (p = {p:.2f})', 'Chance AUC (p = 0.05)'], loc='upper right')
    ax.set_xlabel('AUC')
    ax.set_ylabel('Count')
    plt.show()

if val_auc > 0.6:
    #Predict on the test set
    test_preds = svm.predict(svm_X_test)
    test_probs = svm.predict_proba(svm_X_test)[:, 1]

    #Plot it all nicely
    test_auc, test_thresh = plot_curve_and_matrix(test_preds, test_probs, test_labels, 'Test')

    #Save these
    test_preds_path = os.path.join(intermediates_dir, 'test_preds.npy')
    test_probs_path = os.path.join(intermediates_dir, 'test_probs.npy')
    test_labels_path = os.path.join(intermediates_dir, 'test_labels.npy')
    np.save(test_preds_path, test_preds)
    np.save(test_probs_path, test_probs)
    np.save(test_labels_path, test_labels)

    #Save the SVM model to disk
    model_save_path = os.path.join(intermediates_dir, f'svm_classifier_valauc{val_auc}_testauc{test_auc}.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump(svm, f)

    permute_and_plot_auc(test_labels, test_probs, test_auc, n_permutations=1000, dataset_name='Test_Full')

    plot_auc_only(test_labels, test_probs, 'Test')

    