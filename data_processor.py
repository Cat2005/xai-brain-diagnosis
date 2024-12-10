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

def find_largest_area_slice(segmentation):
    """Find the slice index with the largest area in the segmentation map"""
    areas = np.sum(segmentation > 0, axis=(0, 1))  # Count non-zero pixels in each slice
    return np.argmax(areas)  # Return the index of the slice with the largest area

def process_case(case_id, output_path):
    """Process a single case and save the resulting slices"""
    # Determine tumor type and construct path
    
   
    tumor_type = case_id.split('-')[1]  # GLI, MEN, or MET
    if tumor_type != "MEN":
        return
    print(f"Processing case: {case_id}")
    base_path = Path('BraTS-MEN-Train/')
    

    case_folder = os.path.join(base_path, f"BraTS-{tumor_type}", case_id)
    
    # Load the required modalities
    try:
        # print(os.path.join(base_path, f"{case_id}", f"{case_id}-t2f.nii.gz"))
        t2f = nib.load(os.path.join(base_path, f"{case_id}", f"{case_id}-t2f.nii.gz")).get_fdata()
        t1c = nib.load(os.path.join(base_path, f"{case_id}", f"{case_id}-t1c.nii.gz")).get_fdata()
        t2w = nib.load(os.path.join(base_path, f"{case_id}", f"{case_id}-t2w.nii.gz")).get_fdata()
        segmentation = nib.load(os.path.join(base_path, f"{case_id}", f"{case_id}-seg.nii.gz")).get_fdata()
    except FileNotFoundError as e:
        print(f"Error loading files for case {case_id}: {e}")
        return
    
    # Find the slice with the largest area in the segmentation map
    largest_slice_idx = find_largest_area_slice(segmentation)
    
    # Get the corresponding slices for each modality
    slices = {
        't2f': t2f[:, :, largest_slice_idx],
        't1c': t1c[:, :, largest_slice_idx],
        't2w': t2w[:, :, largest_slice_idx]
    }
    
    # Get label
    label = get_label_from_id(case_id)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create 3-channel image
    slice_data = np.stack([slices['t2f'], slices['t1c'], slices['t2w']], axis=-1)
    
    # Normalize each channel
    for channel in range(3):
        channel_data = slice_data[:, :, channel]
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        if max_val > min_val:
            slice_data[:, :, channel] = (channel_data - min_val) / (max_val - min_val)
    
    # Save slice and label
    output_file = os.path.join(output_path, f"{case_id}_largest_slice")
    np.savez_compressed(
        output_file,
        image=slice_data.astype(np.float32),
        label=label
    )

def main():
    # Configuration
  
    output_path = "test_processed_data_MEN"  # Update this path
    
    # Load split file
    with open('train-test-split.json', 'r') as f:
        split_data = json.load(f)
    
    # Process training cases
    train_cases = set()
    for file_path in split_data['test']:
        #only looks at brats data e.g BraTS-GLI-01661-000-t2f
        if "BraTS" in file_path:
            # Extract case ID (remove modality suffix)
            # print(file_path)
            case_id = '-'.join(file_path.split('-')[:-1])
            train_cases.add(case_id)
    
    # Process each unique case
    for case_id in train_cases:
        
        process_case(case_id, output_path)

if __name__ == "__main__":
    main()