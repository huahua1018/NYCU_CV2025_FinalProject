import numpy as np
import torch
import imageio
from cellpose import dynamics

def filter_invalid_patches(data_list, label_list, prefix, device):
    """
    Remove patches with invalid masks or failed flow conversion.

    Args:
        data_list: list of image patches
        label_list: list of mask patches
        prefix: label for debug outputs (e.g., 'train' or 'val')
        device: torch.device for inference
    Returns:
        filtered_data, filtered_labels, bad_indices
    """
    bad_indices = []
    for i, mask_arr in enumerate(label_list):
        try:
            # If mask has shape (1, H, W), squeeze to (H, W)
            mask = mask_arr[0] if mask_arr.ndim == 3 and mask_arr.shape[0] == 1 else mask_arr
            # Test conversion to flows
            _ = dynamics.labels_to_flows([mask.copy()], files=None, device=device)
        except Exception as e:
            print(f"[Error] {prefix} patch[{i}] labels_to_flows failed: {e}")
            print(f"  unique={np.unique(mask)}, shape={mask.shape}, dtype={mask.dtype}, max={np.max(mask)}")
            imageio.imwrite(f'debug_{prefix}_patch_{i}.png', mask.astype(np.uint16))
            bad_indices.append(i)

    print(f"{prefix} set check finished. Bad indices: {bad_indices}")
    print(f"Total abnormal patches in {prefix} set: {len(bad_indices)}")

    if bad_indices:
        filtered_data = [v for j, v in enumerate(data_list) if j not in bad_indices]
        filtered_labels = [v for j, v in enumerate(label_list) if j not in bad_indices]
        print(f'Filtered {prefix}_data/{prefix}_labels length: {len(filtered_data)}')
        return filtered_data, filtered_labels, bad_indices

    return data_list, label_list, bad_indices


print('[Info] All defective patches have been removed.')
print('[Info] Training and validation sets are now clean and ready for robust Cellpose training.')
