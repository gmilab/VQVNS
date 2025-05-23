##### This script just registers the 1mm MNI brain to the DBM space so that we can MNI space atlases in DBM space

import os
import ants
import shutil
import subprocess

def calculate_MNI_DBV_registration(
    mni_path: str = 'path/to/MNI152_T1_1mm_brain.nii.gz',
    dbm_path: str = 'path/to/template_sharpen_shapeupdate.nii.gz',
    output_dir: str = 'path/to/output_dir/'
):

    mni = ants.image_read(mni_path)
    dbm = ants.image_read(dbm_path)

    registration = ants.registration(fixed=dbm, moving=mni, type_of_transform='SyN')

    warped_mni = ants.apply_transforms(fixed=dbm, moving=mni, transformlist=registration['fwdtransforms'])

    ants.image_write(warped_mni, os.path.join(output_dir, 'mni_in_dbm.nii.gz'))

    fwd_transforms = registration['fwdtransforms']
    inv_transforms = registration['invtransforms']

    for i, transform in enumerate(fwd_transforms):
        if '.nii.gz' in transform:
            shutil.copy2(transform, os.path.join(output_dir, 'mni_to_dbm.nii.gz'))
        elif '.mat' in transform:
            shutil.copy2(transform, os.path.join(output_dir, 'mni_to_dbm.mat'))

    for i, transform in enumerate(inv_transforms):
        if '.nii.gz' in transform:
            shutil.copy2(transform, os.path.join(output_dir, 'dbm_to_mni.nii.gz'))
        elif '.mat' in transform:
            shutil.copy2(transform, os.path.join(output_dir, 'dbm_to_mni.mat'))

def apply_mni_to_dbm_transform(input_image_path: str, 
                               output_image_path: str, 
                               mni_to_dbm_dir: str,
                               reference_img_path: str):
    input_image = ants.image_read(input_image_path)
    reference_img = ants.image_read(reference_img_path)
    
    fwd_warp = os.path.join(mni_to_dbm_dir, 'mni_to_dbm.nii.gz')
    fwd_affine = os.path.join(mni_to_dbm_dir, 'mni_to_dbm.mat')

    warped_image = ants.apply_transforms(fixed=reference_img, moving=input_image, transformlist=[fwd_warp, fwd_affine])

    ants.image_write(warped_image, output_image_path)

def apply_dbm_to_mni_transform(input_image_path: str,
                                 output_image_path: str,
                                 dbm_to_mni_dir: str,
                                 reference_img_path: str = 'path/to/MNI152_T1_1mm_brain.nii.gz',
                                 interp: str = 'linear'): #use 'nearestNeighbor' for binary images, and genericLabel for label maps
     
     input_image = ants.image_read(input_image_path)
     reference_img = ants.image_read(reference_img_path)
     
     inv_warp = os.path.join(dbm_to_mni_dir, 'dbm_to_mni.nii.gz')
     inv_affine = os.path.join(dbm_to_mni_dir, 'dbm_to_mni.mat')
    
     warped_image = ants.apply_transforms(fixed=reference_img, moving=input_image, transformlist=[inv_affine, inv_warp], interpolator=interp)
    
     ants.image_write(warped_image, output_image_path)

def apply_reorient_and_resample(input_image_path: str, 
                                output_image_dir: str, 
                                resample_mm: int = 1, 
                                binarize: bool = False):
    
    base_name = os.path.basename(input_image_path).replace(".nii.gz", "").replace(".nii", "")

     # Reorient to standard
    std_oriented = os.path.join(output_image_dir, f"{base_name}_std.nii.gz")
    subprocess.run(f"fslreorient2std {input_image_path} {std_oriented}", shell=True, check=True)
    
    # Resample to 1mm isotropic
    resampled = os.path.join(output_image_dir, f"{base_name}_std_1mm.nii.gz")
    subprocess.run(f"mri_convert {std_oriented} {resampled} -vs {resample_mm} {resample_mm} {resample_mm}", shell=True, check=True)

    #Binaraize the image
    if binarize:
        subprocess.run(f"fslmaths {resampled} -bin {resampled}", shell=True, check=True)

    return resampled

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Register MNI brain to DBM space and apply transforms.')
    parser.add_argument('--mni_path', type=str, default='path/to/MNI152_T1_1mm_brain.nii.gz', help='Path to MNI152_T1_1mm_brain.nii.gz')
    parser.add_argument('--dbm_path', type=str, default='path/to/template_sharpen_shapeupdate.nii.gz', help='Path to DBM template image')
    parser.add_argument('--output_dir', type=str, default='path/to/output_dir/', help='Directory to save outputs')
    args = parser.parse_args()
    calculate_MNI_DBV_registration(mni_path=args.mni_path, dbm_path=args.dbm_path, output_dir=args.output_dir)