# src/data_preprocessing.py
import os
import h5py
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter
import argparse

def inpaint_gaussian(volume, sigma=1):
    # (Your inpaint_gaussian function here)
    nan_mask = np.isnan(volume)
    if not np.any(nan_mask):
        return volume
    filled = np.copy(volume)
    filled[nan_mask] = 0
    valid_mask = (~nan_mask).astype(float)
    smooth_data = gaussian_filter(filled, sigma=sigma)
    smooth_mask = gaussian_filter(valid_mask, sigma=sigma)
    result = smooth_data / (smooth_mask + 1e-8)
    volume[nan_mask] = result[nan_mask]
    return volume

def save_chunk(images, labels, sites, output_path):
    # (Your save_chunk function, slightly modified)
    chunk_data = {
        'images': images,
        'labels': labels,
        'site_ids': sites
    }
    with open(output_path, 'wb') as f:
        pickle.dump(chunk_data, f)
    print(f"Saved chunk to {output_path}")

def process_raw_data(base_dir, output_dir, chunk_size=10, nan_threshold=0.05):
    """
    Finds .mat files, cleans them, and saves them into valid chunks.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find all .mat files
    mat_file_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mat'):
                mat_file_paths.append(os.path.join(root, file))
    print(f"Found {len(mat_file_paths)} .mat files.")

    # 2. Process and save into valid chunks
    valid_fmri_data, valid_y, valid_site = [], [], []
    valid_chunk_idx = 0
    
    for i, mat_path in enumerate(mat_file_paths):
        print(f"Processing file {i+1}/{len(mat_file_paths)}: {os.path.basename(mat_path)}")
        try:
            with h5py.File(mat_path, 'r') as f:
                if 'fMRIdata' not in f:
                    continue
                
                original_image = f['fMRIdata']['orig'][()]
                brain_mask = f['fMRIdata']['mask'][()]
                subject_label = f['fMRIdata']['dx'][()]
                site_id = mat_path.split(os.sep)[-2] # More robust way to get site_id

                masked_image = original_image * brain_mask
                
                nan_ratio = np.isnan(masked_image).mean()
                if nan_ratio > nan_threshold:
                    print(f"  Skipping due to high NaN ratio: {nan_ratio:.2%}")
                    continue

                interpolated_data = masked_image.copy()
                for t in range(interpolated_data.shape[3]):
                    slice_3d = interpolated_data[:, :, :, t]
                    if np.any(np.isnan(slice_3d)):
                        interpolated_data[:, :, :, t] = inpaint_gaussian(slice_3d)

                valid_fmri_data.append(interpolated_data)
                valid_y.append(subject_label)
                valid_site.append(site_id)

                if len(valid_y) >= chunk_size:
                    chunk_path = os.path.join(output_dir, f"{valid_chunk_idx}.pkl")
                    save_chunk(valid_fmri_data, valid_y, valid_site, chunk_path)
                    valid_fmri_data, valid_y, valid_site = [], [], []
                    valid_chunk_idx += 1

        except Exception as e:
            print(f"  Error processing {os.path.basename(mat_path)}: {e}")

    # Save any remaining data
    if valid_y:
        chunk_path = os.path.join(output_dir, f"{valid_chunk_idx}.pkl")
        save_chunk(valid_fmri_data, valid_y, valid_site, chunk_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess ABIDE .mat files into cleaned .pkl chunks.")
    parser.add_argument('--raw_data_dir', type=str, required=True, help="Directory containing the raw ABIDE .mat files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the cleaned .pkl chunks.")
    parser.add_argument('--chunk_size', type=int, default=10, help="Number of subjects per chunk.")
    args = parser.parse_args()
    
    process_raw_data(args.raw_data_dir, args.output_dir, args.chunk_size)
