"""
Because the LiveCell dataset is too large for training CellposeSAM, we split it into two parts.
"""

import os
import argparse
from collections import defaultdict
import shutil
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split LiveCell dataset into two parts')
    parser.add_argument('--data_dir', type=str, default='sartorius-cell-instance-segmentation/livecell_processed/train/',
                        help='Directory containing the LiveCell dataset')
    parser.add_argument('--part1_dir', type=str, default='sartorius-cell-instance-segmentation/livecell_processed/train1/',
                        help='Directory to save the first part of the dataset')
    parser.add_argument('--part2_dir', type=str, default='sartorius-cell-instance-segmentation/livecell_processed/train2/',
                        help='Directory to save the second part of the dataset')
    args = parser.parse_args()
    data_dir = args.data_dir
    part1_dir = args.part1_dir
    part2_dir = args.part2_dir
    
    os.makedirs(part1_dir, exist_ok=True)
    os.makedirs(part2_dir, exist_ok=True)

    random.seed(42)  # for reproducibility

    # classify files by class
    class_files = defaultdict(list)
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(('.tif')): continue
        class_name = fname.split('_')[0]
        base_name = os.path.splitext(fname)[0]
        seg_name = base_name + '_seg.npy'
        if os.path.exists(os.path.join(data_dir, seg_name)):
            class_files[class_name].append(base_name)

    # shuffle and split files into two parts
    for class_name, base_names in class_files.items():
        random.shuffle(base_names)
        half = len(base_names) // 2
        part1 = base_names[:half]
        part2 = base_names[half:]

        for base in part1:
            for ext in ['.tif', '_seg.npy']:
                src = os.path.join(data_dir, base + ext)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(part1_dir, base + ext))

        for base in part2:
            for ext in ['.tif', '_seg.npy']:
                src = os.path.join(data_dir, base + ext)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(part2_dir, base + ext))
