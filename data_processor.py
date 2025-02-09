import json
import nibabel as nib
import numpy as np
import os
from pathlib import Path

def get_label_from_id(file_id):
    """Convert BraTS ID to one-hot encoded label"""
    if "GLI" in file_id:
        return np.array([1, 0, 0])
    elif "MEN" in file_id:
        return np.array([0, 1, 0])
    elif "MET" in file_id:
        return np.array([0, 0, 1])
    else:
        raise ValueError(f"Unknown tumor type in ID: {file_id}")

def find_largest_area_slices(segmentation):
    """Find the slice indices with the largest area, prioritizing label 1, then 2, then 3"""
    # print(f"Values in segmentation: {np.unique(segmentation)}")
    
    def find_best_slice(areas_1, areas_2, areas_3):
        """Helper to find best slice following priority rules"""
        if np.max(areas_1) > 0:  # If label 1 exists, use it
            return np.argmax(areas_1)
        elif np.max(areas_2) > 0:  # If no label 1 but label 2 exists, use it
            return np.argmax(areas_2)
        else:  # Otherwise use label 3
            return np.argmax(areas_3)
    
    # Axial direction (z-axis)
    axial_areas_1 = np.sum(segmentation == 1, axis=(0, 1))
    axial_areas_2 = np.sum(segmentation == 2, axis=(0, 1))
    axial_areas_3 = np.sum(segmentation == 3, axis=(0, 1))
    axial_slice = find_best_slice(axial_areas_1, axial_areas_2, axial_areas_3)
    # print(f"Axial slice {axial_slice} - Label 1: {np.max(axial_areas_1)}, Label 2: {np.max(axial_areas_2)}, Label 3: {np.max(axial_areas_3)}")
    
    # Sagittal direction (x-axis)
    sagittal_areas_1 = np.sum(segmentation == 1, axis=(1, 2))
    sagittal_areas_2 = np.sum(segmentation == 2, axis=(1, 2))
    sagittal_areas_3 = np.sum(segmentation == 3, axis=(1, 2))
    sagittal_slice = find_best_slice(sagittal_areas_1, sagittal_areas_2, sagittal_areas_3)
    
    # Coronal direction (y-axis)
    coronal_areas_1 = np.sum(segmentation == 1, axis=(0, 2))
    coronal_areas_2 = np.sum(segmentation == 2, axis=(0, 2))
    coronal_areas_3 = np.sum(segmentation == 3, axis=(0, 2))
    coronal_slice = find_best_slice(coronal_areas_1, coronal_areas_2, coronal_areas_3)
    
    return {
        'axial': axial_slice,
        'sagittal': sagittal_slice,
        'coronal': coronal_slice
    }

def process_slice(mri_data, direction, slice_idx):
    """Process a single slice from the specified direction"""
    if direction == 'axial':
        t2f_slice = mri_data['t2f'][:, :, slice_idx]
        t1c_slice = mri_data['t1c'][:, :, slice_idx]
        t2w_slice = mri_data['t2w'][:, :, slice_idx]
        seg_slice = mri_data['seg'][:, :, slice_idx]
    elif direction == 'sagittal':
        t2f_slice = mri_data['t2f'][slice_idx, :, :]
        t1c_slice = mri_data['t1c'][slice_idx, :, :]
        t2w_slice = mri_data['t2w'][slice_idx, :, :]
        seg_slice = mri_data['seg'][slice_idx, :, :]
    else:  # coronal
        t2f_slice = mri_data['t2f'][:, slice_idx, :]
        t1c_slice = mri_data['t1c'][:, slice_idx, :]
        t2w_slice = mri_data['t2w'][:, slice_idx, :]
        seg_slice = mri_data['seg'][:, slice_idx, :]
    
    # print(f"\nSegmentation map for {direction} slice {slice_idx}:")
    # print("Labels present:", np.unique(seg_slice))
    # print("Label counts:")
    # for label in [1, 2, 3]:
    #     count = np.sum(seg_slice == label)
    #     print(f"Label {label}: {count} pixels")
    # print("Segmentation shape:", seg_slice.shape)
    
    # Create binary tumor mask
    tumor_mask = (seg_slice > 0).astype(np.float32)
    
    # Create brain mask
    brain_mask = ((t2f_slice > 0) | (t1c_slice > 0) | (t2w_slice > 0)).astype(np.float32)
    
    # Stack and normalize MRI channels
    slice_data = np.stack([t2f_slice, t1c_slice, t2w_slice], axis=-1)
    
    # Normalize each channel
    for channel in range(3):
        channel_data = slice_data[:, :, channel]
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        if max_val > min_val:
            slice_data[:, :, channel] = (channel_data - min_val) / (max_val - min_val)
        # print(f"Channel {channel} normalized range: {np.min(slice_data[:,:,channel]):.2f} to {np.max(slice_data[:,:,channel]):.2f}")
    
    return slice_data, tumor_mask, brain_mask

def process_case(case_id, output_path):
    """Process a single case and save slices from all three directions"""
    tumor_type = case_id.split('-')[1]  # GLI, MEN, or MET
    print(f"Processing case: {case_id}")
    base_path = Path('ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/')

    try:
        mri_data = {
            't2f': nib.load(os.path.join(base_path, case_id, f"{case_id}-t2f.nii.gz")).get_fdata(),
            't1c': nib.load(os.path.join(base_path, case_id, f"{case_id}-t1c.nii.gz")).get_fdata(),
            't2w': nib.load(os.path.join(base_path, case_id, f"{case_id}-t2w.nii.gz")).get_fdata(),
            'seg': nib.load(os.path.join(base_path, case_id, f"{case_id}-seg.nii.gz")).get_fdata()
        }
    except FileNotFoundError as e:
        print(f"Error loading files for case {case_id}: {e}")
        return

    # Find largest slices in each direction
    largest_slices = find_largest_area_slices(mri_data['seg'])
    
    # Get label
    label = get_label_from_id(case_id)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Process and save slices for each direction
    for direction, slice_idx in largest_slices.items():
        slice_data, tumor_mask, brain_mask = process_slice(mri_data, direction, slice_idx)
        
        # Save slice, label, and masks
        output_file = os.path.join(output_path, f"{case_id}_{direction}")
        np.savez_compressed(
            output_file,
            image=slice_data.astype(np.float32),
            label=label,
            tumor_mask=tumor_mask,
            brain_mask=brain_mask
        )

def main():
    output_path = "planes_val_GLI"
    
    # Load split file
    with open('train-test-split.json', 'r') as f:
        split_data = json.load(f)
    
    # Process training cases
    train_cases = set()
    for file_path in split_data['val']:
        if "BraTS" in file_path:
            case_id = '-'.join(file_path.split('-')[:-1])
            train_cases.add(case_id)
    
    # Process each unique case
    for case_id in train_cases:
        tumor_type = case_id.split('-')[1]
        if tumor_type != "GLI":
            continue
        process_case(case_id, output_path)

if __name__ == "__main__":
    main()