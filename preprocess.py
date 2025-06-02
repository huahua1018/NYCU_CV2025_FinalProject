"""
Split the Sartorius Cell Instance Segmentation dataset into training and validation sets,
and get _seg.npy files for each image containing the segmentation masks.
The _seg.npy is required for training the CellposeSAM.
"""

import os
import argparse
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def enc2mask(encs, shape):
    """
    Convert RLE encoded strings to a binary mask.
    ref : https://github.com/CarnoZhao/mmdetection/blob/sartorius_solution/sartorius/data.ipynb
    """
    img = np.zeros(shape[0]*shape[1], dtype = np.uint16)
    for m, enc in enumerate(encs):
        if isinstance(enc, float) and np.isnan(enc):
            continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m # 1-based indexing
    return img.reshape(shape)

def produce_seg_npy(df, output_dir, image_dir="sartorius-cell-instance-segmentation/train", save_image=True):
    os.makedirs(output_dir, exist_ok=True)

    grouped = df.groupby('id')
    for image_id, group in tqdm(grouped, desc="Generating masks"):
        height = group.iloc[0]['height']
        width = group.iloc[0]['width']
        rle_list = group['annotation'].tolist()  # 所有該 image_id 的 RLE

        # Generate full instance mask
        full_mask = enc2mask(rle_list, (height, width))
        print(f"full_mask shape: {full_mask.shape}")

        # Save mask as a dictionary
        mask_dict = {'masks': full_mask}
        out_path = os.path.join(output_dir, f"{image_id}_seg.npy")
        with open(out_path, 'wb') as f:
            np.save(f, mask_dict, allow_pickle=True)

        # Optionally copy the image to the output directory
        if save_image:
            img_path = os.path.join(image_dir, f"{image_id}.png")
            img = cv2.imread(img_path)
            if img is not None:
                out_img_path = os.path.join(output_dir, f"{image_id}.png")
                cv2.imwrite(out_img_path, img)
            else:
                print(f"[Warning] Failed to read image: {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Sartorius Cell Instance Segmentation dataset for training and validation set."
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="sartorius-cell-instance-segmentation/train",
        help="Directory containing training images."
    )
    parser.add_argument(
        "--label_csv",
        type=str,
        default="sartorius-cell-instance-segmentation/train.csv",
        help="CSV file containing labels and annotations."
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="sartorius-cell-instance-segmentation/mytrain",
        help="Directory to save training set."
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="sartorius-cell-instance-segmentation/myval",
        help="Directory to save validation set."
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the validation split."
    )

    args = parser.parse_args()


    labels = pd.read_csv(args.label_csv)
    image_dir = args.img_dir

    train_dir = args.train_dir
    val_dir = args.val_dir
    val_size = args.val_size


    # Aggregate image-level info by majority cell type
    image_info = labels.groupby("id").agg({
        "cell_type": lambda x: x.mode()[0],  # Take the most frequent cell_type per image
        "width": "first",
        "height": "first"
    }).reset_index()


    # Perform stratified split based on cell_type
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    for train_idx, val_idx in splitter.split(image_info["id"], image_info["cell_type"]):
        train_ids = image_info.loc[train_idx, "id"].values
        val_ids = image_info.loc[val_idx, "id"].values

    # Filter labels for training and validation sets
    train_df = labels[labels["id"].isin(train_ids)].reset_index(drop=True)
    val_df = labels[labels["id"].isin(val_ids)].reset_index(drop=True)

    # Print dataset statistics
    print("Train images:", train_df["id"].nunique())
    print("Val images:", val_df["id"].nunique())
    print("Cell types in train:")
    print(train_df["cell_type"].value_counts())
    print("Cell types in val:")
    print(val_df["cell_type"].value_counts())

    # Generate and save masks
    produce_seg_npy(train_df, train_dir, image_dir)
    produce_seg_npy(val_df, val_dir, image_dir)