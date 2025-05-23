##### Function that are used randomly through the project

import nibabel as nib
import numpy as np
import torch
import smtplib
import email.mime.text

# Email server configuration - set these for your environment
server = 'smtp.example.com'  # e.g., 'smtp.gmail.com'
port = 587  # or 25, 465, etc.
target_email = 'your_email@example.com'
sender = 'your_email@example.com'


def sendEmailNotification(subject, message, target_email=target_email):
    with smtplib.SMTP(server, port) as smtp:
        msg = email.mime.text.MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = target_email
        smtp.starttls()
        smtp.send_message(msg)

def save_np_as_nifti(data, filename):
    """
    Save a numpy array as a nifti file
    Args:
        data (np.ndarray): 4D numpy array to save with subjects in the first dimension
        affine (np.ndarray): 4x4 affine matrix
        filename (str): path to save the nifti file
    """
    affine = get_model_space_affine()
    data = np.squeeze(data)
    data= np.moveaxis(data, 0, -1) # Move subjects to the last dimension
    data = data.astype(np.float32)
    img = nib.Nifti1Image(data, get_model_space_affine())
    img.header.set_xyzt_units(2)
    img.header.set_sform(get_model_space_affine(), code=1)
    img.header.set_qform(get_model_space_affine(), code=1)
    nib.save(img, filename)


def get_model_space_affine():

    '''
    This function returns the affine matrix for the model space.
    This affine has the same diagonal scale elements as MNI which is -1 1 1 
    However the translation is not aligned to MNI space and are just the center of the 176 x 208 x 176 grid.

    The affine is in the form of a 4x4 matrix.

    '''

    image_size = (176, 208, 176)
    affine = np.eye(4)
    affine[0, 0] = -1
    affine[1, 1] = 1
    affine[2, 2] = 1
    affine[0, 3] = np.floor(image_size[0] / 2) 
    affine[1, 3] = -np.floor(image_size[1] / 2) 
    affine[2, 3] = -np.floor(image_size[2] / 2) 

    return affine

def save_model_space_image(data, filename):
    """
    Save a numpy array as a nifti file
    Args:
        data (np.ndarray): 3D numpy array to save - this only works for single subjects
        affine (np.ndarray): 4x4 affine matrix
        filename (str): path to save the nifti file
    """
    data = np.squeeze(data)
    data = data.astype(np.float32)
    img = nib.Nifti1Image(data, get_model_space_affine())
    img.header.set_xyzt_units(2)
    img.header.set_sform(get_model_space_affine(), code=1)
    img.header.set_qform(get_model_space_affine(), code=1)
    nib.save(img, filename)

def mask_subcortical_structures(img_path, output_path, binarize=False):
    """
    Mask out the values that are not the brain from the synthseg output. This won't work for anything else. 

    Parameters
    ----------
    img_path : str
        The path to the image
    output_path : str
        The output path for the brain mask
    """
    labels_structures = {
    0: "Background",
    2: "Left cerebral white matter",
    3: "Left cerebral cortex",
    4: "Left lateral ventricle",
    5: "Left inferior lateral ventricle",
    7: "Left cerebellum white matter",
    8: "Left cerebellum cortex",
    10: "Left thalamus",
    11: "Left caudate",
    12: "Left putamen",
    13: "Left pallidum",
    14: "3rd ventricle",
    15: "4th ventricle",
    16: "Brain-stem",
    17: "Left hippocampus",
    18: "Left amygdala",
    24: "CSF (SynthSeg 2.0 only)",
    26: "Left accumbens area",
    28: "Left ventral DC",
    41: "Right cerebral white matter",
    42: "Right cerebral cortex",
    43: "Right lateral ventricle",
    44: "Right inferior lateral ventricle",
    46: "Right cerebellum white matter",
    47: "Right cerebellum cortex",
    49: "Right thalamus",
    50: "Right caudate",
    51: "Right putamen",
    52: "Right pallidum",
    53: "Right hippocampus",
    54: "Right amygdala",
    58: "Right accumbens area",
    60: "Right ventral DC"
}

    #Keep only the the ones that we want for brain extraction
    brain_structures = [10, 11, 12, 13, 16, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60]

    img = nib.load(img_path)
    img_data = img.get_fdata()

    brain_mask = np.zeros_like(img_data)

    for structure in brain_structures:
        brain_mask[img_data == structure] = 1

    masked_img_data = img_data * brain_mask
    if binarize:
        masked_img_data[brain_mask > 0] = 1

    masked_img = nib.Nifti1Image(masked_img_data, img.affine, img.header)
    nib.save(masked_img, output_path)

def map_server_path_to_local_hpc_path(path_to_map, server_path='path/to/server/', hpc_path='path/to/hpc_storage/'):
    """
    Map the server path to the HPC path
    Args:
        server_path (str): path on the server
        hpc_path (str): path on the HPC
    Returns:
        str: path on the HPC
    """
    return path_to_map.replace(server_path, hpc_path)

def map_local_hpc_path_to_actual_hpc_path(path_to_map, server_path='path/to/hpc_storage/', hpc_path='path/to/actual_hpc/'):
    """
    Map the local HPC path to the actual HPC path
    Args:
        server_path (str): path on the local HPC
        hpc_path (str): path on the actual HPC
    Returns:
        str: path on the actual HPC
    """
    return path_to_map.replace(server_path, hpc_path)
