"""
Visualize CellposeSAM model predictions and compare with different training settings.
"""
import os
import argparse
from collections import Counter, defaultdict
import json

import cv2
import numpy as np
from tqdm.auto import tqdm
from cellpose import models, core, io
import matplotlib.pyplot as plt



def plot_multiple_image_and_mask_overlay(
    samples, postfix, alpha=0.5
):
    """
    Plot multiple images with their corresponding masks and ground truth masks.
    """

    num_samples = len(samples)
    fig, axs = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axs = np.expand_dims(axs, 0)

    for row, (image, mask_gt, mask1, mask2) in enumerate(samples):
        blended = []
        for mask in [mask1, mask2]:
            instance_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids != 0]

            # fix color
            np.random.seed(42)
            for i in unique_ids:
                color = np.random.randint(0, 255, size=3)
                instance_mask[mask == i] = color

            if image.ndim == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image.copy()
            blended.append(
                (image_rgb * (1 - alpha) + instance_mask * alpha).astype(np.uint8)
            )

        # Ground Truth
        gt_mask = np.zeros((*mask_gt.shape, 3), dtype=np.uint8)
        unique_ids_gt = np.unique(mask_gt)
        unique_ids_gt = unique_ids_gt[unique_ids_gt != 0]
        np.random.seed(42)
        for i in unique_ids_gt:
            color = np.random.randint(0, 255, size=3)
            gt_mask[mask_gt == i] = color
        blended_gt = (image_rgb * (1 - alpha) + gt_mask * alpha).astype(np.uint8)

        axs[row, 0].imshow(image_rgb)
        axs[row, 0].set_title("Original Image" if row == 0 else "")
        axs[row, 0].axis("off")

        axs[row, 1].imshow(blended_gt)
        axs[row, 1].set_title("Mask Ground Truth" if row == 0 else "")
        axs[row, 1].axis("off")

        axs[row, 2].imshow(blended[0])
        axs[row, 2].set_title("Full-size Training" if row == 0 else "")
        axs[row, 2].axis("off")

        axs[row, 3].imshow(blended[1])
        axs[row, 3].set_title("Patch-based Training" if row == 0 else "")
        axs[row, 3].axis("off")

    plt.tight_layout(h_pad=1.0)
    plt.savefig(f"mask_overlay_{postfix}_barry.png")
    print(f"mask_overlay_{postfix}.png saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize CellposeSAM model predictions and compare with different training settings."
    )
    parser.add_argument(
        "--model_path1",
        type=str,
        default="checkpoints/fullres_epoch_0035",
        help="Path to the first model checkpoint",
    )
    parser.add_argument(
        "--model_path2",
        type=str,
        default="checkpoints/patch_best335",
        help="Path to the second model checkpoint",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="sartorius-cell-instance-segmentation/bbox_val_complete/",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--bbox_json",
        type=str,
        default="yolo_bbox/predictions.json",
        help="Path to the JSON file containing bounding box predictions",
    )
    parser.add_argument(
        "--masks_ext", type=str, default="_seg.npy", help="Extension for the mask files"
    )
    args = parser.parse_args()

    # --------------------- set up ----------------------- #
    io.logger_setup()  # run this to get printing of progress
    # Check if colab notebook instance has GPU access
    if core.use_gpu() == False:
        raise ImportError("No GPU access, change your runtime")
    # ------------------------ params ------------------------ #
    model_paths = {
        1: args.model_path1,  # Full-size training
        2: args.model_path2,  # Patch-based training
    }

    test_dir = args.test_dir
    bbox_json = args.bbox_json
    masks_ext = args.masks_ext

    min_sizes = {1: 59, 2: 136, 3: 74}  # shsy5y  # astro  # cort

    # -------------------------- load data ------------------------- #
    # read test data
    output = io.load_train_test_data(test_dir, mask_filter=masks_ext)
    test_data, test_labels, test_image_name, _, _, _ = output

    # ----------------- load model and test ----------------------- #
    dif_models = {
        cat_id: models.CellposeModel(pretrained_model=path)
        for cat_id, path in model_paths.items()
    }

    # open bbox jason file
    with open(bbox_json, "r") as f:
        bbox_data = json.load(f)

    # use defaultdict to collect bboxes and category_ids
    bboxes_by_image = defaultdict(list)
    category_count_by_image = defaultdict(list)

    # collect all the category_ids for each image_id
    for cnt, ann in enumerate(bbox_data["annotations"]):
        image_id = ann["image_id"]
        bbox = ann["bbox"]
        category_id = ann["category_id"]

        bboxes_by_image[image_id].append(bbox)
        category_count_by_image[image_id].append(category_id)

    # calculate the majority category for each image as the image's category
    image_to_major_category = {}

    for image_id, category_list in category_count_by_image.items():
        category_counter = Counter(category_list)
        majority_category_id = category_counter.most_common(1)[0][0]
        image_to_major_category[image_id] = majority_category_id

    imgs1 = []
    for idx in tqdm(range(len(test_data)), desc="Processing images", unit="image"):
        img = test_data[idx]
        gt = test_labels[idx]

        # get base image name from test_image_name
        img_name = os.path.basename(test_image_name[idx])
        img_name = img_name.split(".")[0]

        H, W = img.shape[:2]
        min_size = min_sizes[image_to_major_category[img_name]]

        pred_mask = dif_models[1].eval(img, batch_size=32, min_size=min_size)[0]

        imgs1.append(pred_mask)

    imgs2 = []
    for idx in tqdm(range(len(test_data)), desc="Processing images", unit="image"):
        img = test_data[idx]
        gt = test_labels[idx]

        # get base image name from test_image_name
        img_name = os.path.basename(test_image_name[idx])
        img_name = img_name.split(".")[0]

        H, W = img.shape[:2]
        min_size = min_sizes[image_to_major_category[img_name]]

        pred_mask = dif_models[2].eval(img, batch_size=32, min_size=min_size)[0]

        imgs2.append(pred_mask)

    # only draw first 3 samples
    samples = []
    for idx in range(3):
        samples.append((test_data[idx], test_labels[idx], imgs1[idx], imgs2[idx]))

    plot_multiple_image_and_mask_overlay(
        samples, postfix="pretrained_livecell", alpha=0.5
    )
