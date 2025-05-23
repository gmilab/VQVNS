#### The purpose of this script is to loop through the input images and identify the smallest image crop that can be used to crop all images to the same size

# Import necessary libraries
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify smallest image crop and crop all images accordingly.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save cropped images')
    parser.add_argument('--crop_dims_txt', type=str, default='smallest_image_crop.txt', help='Path to save smallest crop dimensions (txt)')
    parser.add_argument('--crop_csv', type=str, default='image_crops.csv', help='Path to save crop info for each image (csv)')
    args = parser.parse_args()

    image_dir = args.image_dir
    output_dir = args.output_dir
    crop_dims_txt = args.crop_dims_txt
    crop_csv = args.crop_csv

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the list of all image files
    image_files = os.listdir(image_dir)

    # Initialize the minimum dimensions
    min_dims = np.zeros(3)

    # Let's make a df to store the values for each subject to undo the cropping later
    df = pd.DataFrame(columns=['study_id', 'start_x', 'end_x', 'start_y', 'end_y', 'start_z', 'end_z'])

    # Loop through the images and identify the smallest dimensions without cutting off any non-zero values
    prog = tqdm(image_files, desc='Identifying smallest image crop and cropping')
    for image_file in prog:
        # Load the image
        img = nib.load(os.path.join(image_dir, image_file))
        data = img.get_fdata()
        
        # Identify the start of non-zero values in each dimension
        start_x, end_x = np.where(data.sum(axis=(1, 2)) != 0)[0][0], np.where(data.sum(axis=(1, 2)) != 0)[0][-1]
        start_y, end_y = np.where(data.sum(axis=(0, 2)) != 0)[0][0], np.where(data.sum(axis=(0, 2)) != 0)[0][-1]
        start_z, end_z = np.where(data.sum(axis=(0, 1)) != 0)[0][0], np.where(data.sum(axis=(0, 1)) != 0)[0][-1]

        # Calculate the dimensions
        dims = (end_x - start_x, end_y - start_y, end_z - start_z)

        # Crop the image
        data = data[start_x:end_x, start_y:end_y, start_z:end_z]
        cropped_img = nib.Nifti1Image(data, img.affine)
        nib.save(cropped_img, os.path.join(output_dir, image_file))

        # Update the dataframe
        row = pd.DataFrame({'study_id': image_file.split('.')[0], 'start_x': start_x, 'end_x': end_x, 'start_y': start_y, 'end_y': end_y, 'start_z': start_z, 'end_z': end_z}, index=[0])
        df = pd.concat([df, row], axis=0, ignore_index=True)

        # Update the minimum dimensions
        min_dims = np.maximum(min_dims, dims)

        # Update the progress bar
        prog.set_postfix(min_dims=min_dims)

    # Print the smallest dimensions
    print(f"Smallest image crop dimensions: {min_dims}")

    # Save the smallest dimensions to a text file
    np.savetxt(crop_dims_txt, min_dims)
    df.to_csv(crop_csv, index=False)