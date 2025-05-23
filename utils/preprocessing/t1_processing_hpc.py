import os
import glob
import subprocess
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
from intensity_normalization.normalize.fcm import FCMNormalize
from intensity_normalization.typing import Modality
import nibabel as nib
from tqdm import tqdm
import ants
import concurrent.futures
import numpy as np
import pandas as pd
import shutil
import argparse

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e}")


def mask_out_values(img_path, output_path):
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
    brain_structures = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28, 41, 42, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]

    img = nib.load(img_path)
    img_data = img.get_fdata()

    brain_mask = np.zeros_like(img_data)

    for structure in brain_structures:
        brain_mask[img_data == structure] = 1

    brain_mask_img = nib.Nifti1Image(brain_mask, img.affine, img.header)
    nib.save(brain_mask_img, output_path)

def brain_extraction(input_file, output_prefix, synthstrip_weight=0.05, synthseg_weight=0.95, n_threads=7):
    '''
    This function extracts the brain using synthstrip and synthseg and generates a hybrid mask and a brain-extracted image.
    All intermediate files are saved in the same directory using the output_prefix.
    Once the final output (named "<output_prefix>_brain_T1w.nii.gz") is produced,
    the intermediate files are deleted.
    '''

    # Define commands for synthstrip and synthseg using the output_prefix
    synthstrip_cmd = f'mri_synthstrip -i {input_file} -m {output_prefix}_synthstripmask.nii.gz -b 0 --no-csf'
    synthseg_cmd = f'mri_synthseg --i {input_file} --o {output_prefix}_synthseg.nii.gz --threads {n_threads}'

    try:
        subprocess.run(synthstrip_cmd, shell=True, check=True)
        subprocess.run(synthseg_cmd, shell=True, check=True)

        # Mask out non-brain values from synthseg output
        mask_out_values(f'{output_prefix}_synthseg.nii.gz', f'{output_prefix}_synthsegmask.nii.gz')

        # Resample synthseg mask to synthstrip mask space
        resample_cmd = (
            f'flirt -in {output_prefix}_synthsegmask.nii.gz -ref {output_prefix}_synthstripmask.nii.gz '
            f'-out {output_prefix}_synthsegmask_resample.nii.gz -applyxfm -interp nearestneighbour'
        )
        subprocess.run(resample_cmd, shell=True, check=True)

        # Binarize the resampled mask
        binarize_cmd = f'fslmaths {output_prefix}_synthsegmask_resample.nii.gz -bin {output_prefix}_synthsegmask_resample.nii.gz'
        subprocess.run(binarize_cmd, shell=True, check=True)

        # Generate hybrid mask using weighted sum of synthstrip and synthseg masks
        synth_mask = nib.load(f'{output_prefix}_synthstripmask.nii.gz')
        synthseg_mask = nib.load(f'{output_prefix}_synthsegmask_resample.nii.gz')
        hybrid_mask = (synth_mask.get_fdata() * synthstrip_weight) + (synthseg_mask.get_fdata() * synthseg_weight)
        hybrid_mask_img = nib.Nifti1Image(hybrid_mask, synth_mask.affine, synth_mask.header)
        hybrid_mask_path = f'{output_prefix}_hybridmask.nii.gz'
        nib.save(hybrid_mask_img, hybrid_mask_path)

        # Binarize the hybrid mask
        binarize_cmd = f'fslmaths {hybrid_mask_path} -bin {output_prefix}_hybridmask_bin.nii.gz'
        subprocess.run(binarize_cmd, shell=True, check=True)

        # Apply the binary mask to extract the brain from the input image
        brain_temp_path = f'{output_prefix}_brain.nii.gz'
        apply_mask_cmd = f'fslmaths {input_file} -mas {output_prefix}_hybridmask_bin.nii.gz {brain_temp_path}'
        subprocess.run(apply_mask_cmd, shell=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f'Error in extracting the brain: {e}')
        return False

    # Define the final output path in the same directory as output_prefix, with a final name
    final_brain_file = f'{output_prefix}.nii.gz'
    shutil.move(brain_temp_path, final_brain_file)

    # List of intermediate files to delete (all except the final file)
    intermediate_files = [
        f'{output_prefix}_synthstripmask.nii.gz',
        f'{output_prefix}_synthseg.nii.gz',
        f'{output_prefix}_synthsegmask.nii.gz',
        f'{output_prefix}_synthsegmask_resample.nii.gz',
        f'{output_prefix}_hybridmask.nii.gz',
        f'{output_prefix}_hybridmask_bin.nii.gz',
        f'{output_prefix}_brain.nii.gz'
    ]
    
    for file_path in intermediate_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as err:
                print(f"Error deleting {file_path}: {err}")

    return final_brain_file

def process_niftis(nifti_file, output_dir, reference_image, normalize_method, normalize_prefix="", 
                   dbm_template_img_path = 'path/to/dbm_template.nii.gz',
                   mni_template_img_path = 'path/to/mni_template.nii.gz'):
    
    base_name = os.path.basename(nifti_file).split('.')[0]

    #Intermediate folders 
    brain_extracted_dir = os.path.join(output_dir, f"brain_extracted")
    reoriented_dir = os.path.join(output_dir, "reoriented")
    resampled_dir = os.path.join(output_dir, "resampled")
    normalized_dir = os.path.join(output_dir, f"{normalize_prefix}normalized")
    mni_reg_dir = os.path.join(output_dir, f"{normalize_prefix}mni_reg")
    dbm_reg_dir = os.path.join(output_dir, f"{normalize_prefix}dbm_reg")
    cropped_dir = os.path.join(output_dir, f"{normalize_prefix}mni_reg_cropped")
    dimensions_dir = os.path.join(output_dir, f"{normalize_prefix}mni_reg_cropped_dimensions")

    if not os.path.exists(brain_extracted_dir):
        os.makedirs(brain_extracted_dir)

    if not os.path.exists(reoriented_dir):
        os.makedirs(reoriented_dir)

    if not os.path.exists(resampled_dir):
        os.makedirs(resampled_dir)

    if not os.path.exists(normalized_dir):
        os.makedirs(normalized_dir)

    if not os.path.exists(mni_reg_dir):
        os.makedirs(mni_reg_dir)

    if not os.path.exists(dbm_reg_dir):
        os.makedirs(dbm_reg_dir)

    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)

    if not os.path.exists(dimensions_dir):
        os.makedirs(dimensions_dir)

    #Brain extraction first
    brain_extraction_prefix = os.path.join(brain_extracted_dir, f"{base_name}")
    brain_extracted_file = brain_extraction(nifti_file, brain_extraction_prefix)

    # Reorient to standard
    std_oriented = os.path.join(reoriented_dir, f"{base_name}.nii.gz")
    if not os.path.exists(std_oriented):
        run_command(f"fslreorient2std {brain_extracted_file} {std_oriented}")
    
    # Resample to 1mm isotropic
    resampled = os.path.join(resampled_dir, f"{base_name}.nii.gz")
    if not os.path.exists(resampled):
        run_command(f"mri_convert {std_oriented} {resampled} -vs 1 1 1 > /dev/null")
    
    # Load image for normalization
    normalized_path = os.path.join(normalized_dir, f"{base_name}.nii.gz")
    if not os.path.exists(normalized_path):
        img = nib.load(resampled)
        img_data = img.get_fdata()
        normalizer = normalize_method()
        normalized_data = normalizer(img_data, modality=Modality.T1)
        normalized_img = nib.nifti1.Nifti1Image(normalized_data, img.affine, img.header)
        
        # Save normalized image
        nib.save(normalized_img, normalized_path)

    #Let's align it to the template dbm space 
    dbm_warped_path = os.path.join(dbm_reg_dir, f"{base_name}.nii.gz")
    mni_warped_path = os.path.join(mni_reg_dir, f"{base_name}.nii.gz")

    normalized_img = ants.image_read(normalized_path)

    if not os.path.exists(dbm_warped_path):
        dbm_template_img = ants.image_read(dbm_template_img_path)
        dbm_reg = ants.registration(fixed=dbm_template_img, moving=normalized_img, type_of_transform='Rigid')
        dbm_warped = dbm_reg['warpedmovout']
        ants.image_write(dbm_warped, dbm_warped_path)

    if not os.path.exists(mni_warped_path):
        mni_template_img = ants.image_read(mni_template_img_path)
        mni_reg = ants.registration(fixed=mni_template_img, moving=normalized_img, type_of_transform='Rigid')
        mni_warped = mni_reg['warpedmovout']
        ants.image_write(mni_warped, mni_warped_path)

    #Let's crop the image to the smallest non-zero values
    cropped_path = os.path.join(cropped_dir, f"{base_name}.nii.gz")
    cropped_dims_df_path = os.path.join(dimensions_dir, f"{base_name}.csv")
    if not os.path.exists(cropped_path):
        # Load the image
        img = nib.load(mni_warped_path)
        data = img.get_fdata()
        
        #Identify the start of non-zero values in each dimension
        start_x, end_x = np.where(data.sum(axis=(1, 2)) > 0)[0][0], np.where(data.sum(axis=(1, 2)) > 0)[0][-1]
        start_y, end_y = np.where(data.sum(axis=(0, 2)) > 0)[0][0], np.where(data.sum(axis=(0, 2)) > 0)[0][-1]
        start_z, end_z = np.where(data.sum(axis=(0, 1)) > 0)[0][0], np.where(data.sum(axis=(0, 1)) > 0)[0][-1]

        # Calculate the dimensions
        dims = (end_x - start_x, end_y - start_y, end_z - start_z)

        #Crop the image
        data = data[start_x:end_x, start_y:end_y, start_z:end_z]
        cropped_img = nib.Nifti1Image(data, img.affine)
        nib.save(cropped_img, os.path.join(output_dir, cropped_path))

        #Write it out
        df = pd.DataFrame(columns=['study_id', 'start_x', 'end_x', 'start_y', 'end_y', 'start_z', 'end_z'])
        row = pd.DataFrame({'study_id': base_name, 'start_x': start_x, 'end_x': end_x, 'start_y': start_y, 'end_y': end_y, 'start_z': start_z, 'end_z': end_z}, index=[0])
        df = pd.concat([df, row], axis=0, ignore_index=True)
        df.to_csv(cropped_dims_df_path, index=False)
        
    return True

def prepare_for_hpc_run(input_dir, hpc_output_dir, output_dir, reference_image, hpc_job_limit=1498):

    def remap_local_to_hpc(local_path: str, 
                           replacement_string='/local/storage/', 
                           target_string='/hpc/storage/'):
        # Update these strings to match your local and HPC storage mount points
        return local_path.replace(replacement_string, target_string)

    if not os.path.exists(hpc_output_dir):
        os.makedirs(hpc_output_dir)

    #get current script path
    script_path = os.path.abspath(__file__)
    script_name = os.path.basename(script_path)

    # Copy the script to the HPC
    output_script_path = os.path.join(hpc_output_dir, script_name)
    shutil.copy(script_path, output_script_path)
    
    images_to_process = os.listdir(input_dir)

    #images per job
    images_per_job = int(np.ceil(len(images_to_process) / hpc_job_limit))

    command_base = 'python {} -i {} -o {} -r {}'
    command_base = remap_local_to_hpc(command_base)
    commands_to_run = ''
    log_dir = os.path.join(hpc_output_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    else:
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)

    for idx in range(0, len(images_to_process), images_per_job):
        print(f"Processing images {idx} to {idx + images_per_job}")
        images_string = ''
        for image in images_to_process[idx: min(idx + images_per_job, len(images_to_process))]:
            images_string += remap_local_to_hpc(os.path.join(input_dir, image)) + ' '
        images_string = images_string.strip()

        subject_command = command_base.format(remap_local_to_hpc(output_script_path), images_string, remap_local_to_hpc(output_dir), remap_local_to_hpc(reference_image))
        commands_to_run += f'{subject_command}\n'

    # Write the commands to a file
    with open(os.path.join(hpc_output_dir, 'commands.txt'), 'w') as f:
        f.write(commands_to_run)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess healthy T1 images')
    parser.add_argument('--input_images', '-i', type=str, required=False, help='Input image path', nargs='+')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--reference_image', '-r', type=str, required=True, help='Reference image path')
    parser.add_argument('--prep_mode', '-m', type=bool, default=False, help='Prepare for HPC run', required=False)
    parser.add_argument('--hpc_output_dir', '-hpc', type=str,  help='HPC output directory', required=False)
    parser.add_argument('--input_dir', '-d', type=str, help='Input directory', required=False)
    parser.add_argument('--dbm_template', type=str, default='path/to/dbm_template.nii.gz', help='DBM template image path')
    parser.add_argument('--mni_template', type=str, default='path/to/mni_template.nii.gz', help='MNI template image path')

    args = parser.parse_args()
    prep_mode = args.prep_mode
    output_dir = args.output_dir
    reference_image = args.reference_image
    dbm_template = args.dbm_template
    mni_template = args.mni_template
    if not prep_mode:
        input_images = args.input_images
    else:
        hpc_output_dir = args.hpc_output_dir
        input_dir = args.input_dir


    # Define the normalization method
    if not prep_mode:
        normalize_method = FCMNormalize
        for input_image in input_images:
            process_niftis(input_image, output_dir, reference_image, normalize_method, normalize_prefix="fcm_", dbm_template_img_path=dbm_template, mni_template_img_path=mni_template)
            print(f"Finished processing {input_image}")
    else:
        prepare_for_hpc_run(input_dir, hpc_output_dir, output_dir, reference_image)
        print(f"Finished preparing for HPC run")
