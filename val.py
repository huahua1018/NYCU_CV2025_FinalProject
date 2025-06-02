from cellpose import models, core, metrics, dynamics
from cellpose import io as cpio
import skimage.io as skio
import numpy as np
import cv2
import torch
import pycocotools.mask
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging

# Suppress Cellpose logs below ERROR
logging.getLogger('cellpose').setLevel(logging.ERROR)
logging.getLogger('cellpose.io').setLevel(logging.ERROR)
logging.getLogger('cellpose.core').setLevel(logging.ERROR)
logging.getLogger('cellpose.dynamics').setLevel(logging.ERROR)

# Disable autograd for faster inference
torch.set_grad_enabled(False)

# =============================
# 1. Helper functions
# =============================

def remove_overlap_naive(masks):
    """
    Greedily remove overlapping masks:
    Encode masks as RLE, compute IoU, then for each overlapping group,
    subtract the union of previous masks from the next.
    """
    if masks.size == 0:
        return masks

    # Encode masks and compute IoU matrix
    rles = [pycocotools.mask.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
    ious = pycocotools.mask.iou(rles, rles, [0] * len(rles))
    np.fill_diagonal(ious, 0)
    toproc = np.where(ious.sum(axis=0) > 0)[0]
    if len(toproc) == 0:
        return masks

    mt = torch.from_numpy(masks.astype(np.uint8)).cuda()
    prev = mt[toproc[0]].clone()
    for idx, i in enumerate(toproc[1:], start=1):
        prev = torch.max(prev, mt[toproc[idx - 1]])
        mt[i] *= (~prev)
    return mt.cpu().numpy()

def instmap_to_masks_boxes(inst_map):
    """
    Convert 2D instance map to binary masks and bounding boxes:
    Returns masks array and boxes array (x0, y0, x1, y1, area, score=0.0).
    """
    masks, boxes = [], []
    for lab in np.unique(inst_map)[1:]:
        m = (inst_map == lab).astype(np.uint8)
        ys, xs = np.where(m)
        if ys.size == 0:
            continue
        area = int(m.sum())
        masks.append(m)
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max(), area, 0.0])
    if not masks:
        return None, None
    return np.stack(masks, axis=0), np.array(boxes, dtype=float)

def weighted_mask_fusion_nmw(masks, boxes, scores, iou_thr=0.15, score_coef=0.8):
    """
    Non-Maximum Weighted fusion of masks:
    1. Compute IoU matrix.
    2. Group masks with IoU > iou_thr.
    3. For each group, compute weights = (scores * score_coef) / sum.
    4. Compute soft_map = weighted sum of group masks.
    5. Binarize at 0.5 or fallback to highest-score mask if empty.
    """
    N = masks.shape[0]
    if N == 0:
        return np.zeros((0, masks.shape[1], masks.shape[2]), dtype=np.uint8), np.zeros((0, 6), dtype=float)

    rles = [pycocotools.mask.encode(np.asfortranarray(m)) for m in masks]
    ious = pycocotools.mask.iou(rles, rles, [0] * N)
    used = set()
    fused_masks, fused_boxes = [], []

    for i in range(N):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, N):
            if ious[i, j] > iou_thr:
                group.append(j)
        used.update(group)

        sub_masks = masks[group]
        sub_boxes = boxes[group]
        sub_scores = scores[group].astype(float) * score_coef

        if len(group) == 1:
            fused_masks.append(sub_masks[0])
            fused_boxes.append(sub_boxes[0])
            continue

        weights = sub_scores / sub_scores.sum()
        soft_map = np.tensordot(weights, sub_masks, axes=(0, 0))
        bin_mask = (soft_map >= 0.5).astype(np.uint8)
        ys, xs = np.where(bin_mask)
        if ys.size == 0:
            best_idx = group[np.argmax(sub_scores)]
            bin_mask = masks[best_idx]
            ys, xs = np.where(bin_mask)
            fused_masks.append(bin_mask)
            fused_boxes.append([xs.min(), ys.min(), xs.max(), ys.max(), int(bin_mask.sum()), 0.0])
        else:
            fused_masks.append(bin_mask)
            fused_boxes.append([xs.min(), ys.min(), xs.max(), ys.max(), int(bin_mask.sum()), 0.0])

    if not fused_masks:
        return np.zeros((0, masks.shape[1], masks.shape[2]), dtype=np.uint8), np.zeros((0, 6), dtype=float)
    return np.stack(fused_masks, axis=0), np.array(fused_boxes, dtype=float)

def rle_encode(binary_mask):
    """
    Encode a binary mask (uint8) into COCO RLE string.
    """
    pixels = binary_mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def tta_predict_probability_and_flows(model, image, image_id):
    """
    Apply TTA (original + horizontal flip) to predict probability and flow maps.
    Returns averaged probability map and dp map (shape=(2, H, W)).
    """
    images = [image, np.flip(image, axis=1)]
    inv_prob = [lambda x: x, lambda x: np.flip(x, axis=1)]
    inv_dp = [
        lambda dp: dp,
        lambda dp: np.stack([np.flip(dp[0], axis=1), -np.flip(dp[1], axis=1)], axis=0)
    ]

    all_prob, all_dp = [], []
    H, W = image.shape[:2]

    for idx, aug_img in enumerate(images):
        _, eval_flows, _ = model.eval(aug_img, compute_masks=False)
        dp_raw = eval_flows[1].astype(np.float32)
        prob_raw = eval_flows[2].astype(np.float32)

        prob = inv_prob[idx](prob_raw)
        dp = inv_dp[idx](dp_raw)

        # Resize if necessary
        if prob.shape[:2] != (H, W):
            prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        if dp.shape[1:] != (H, W):
            dp_x = cv2.resize(dp[0], (W, H), interpolation=cv2.INTER_LINEAR)
            dp_y = cv2.resize(dp[1], (W, H), interpolation=cv2.INTER_LINEAR)
            dp = np.stack([dp_x, dp_y], axis=0)

        all_prob.append(prob)
        all_dp.append(dp)

    return np.mean(all_prob, axis=0), np.mean(all_dp, axis=0)

# =============================
# 2. Parameters
# =============================
CKPT_PATHS = [
    "fullres/fullres/fullres_epoch_0035",  # 0.3048 mAP
    "patch_best335",                       # 0.3056 mAP
    # Add more checkpoints as needed
]

VAL_DIR     = "sartorius-cell-instance-segmentation/myval/"
MASK_EXT    = "_seg.npy"

decode_flow_th = 0.4   # flow threshold for decoding
iou_thr_nmw    = 0.45  # IoU threshold for WMF grouping
score_coef     = 0.8   # score scaling factor
min_size       = 75    # minimum mask area to retain
corrupt        = True  # apply convex hull correction for astrocytes
cell_type      = 0     # 1 = astrocyte, 0 = skip hull

# =============================
# 3. Load data & models
# =============================
cpio.logger_setup()
assert core.use_gpu(), "GPU is required for inference"

# Load validation images and ground-truth masks
imgs, gts, *_ = cpio.load_train_test_data(VAL_DIR, mask_filter=MASK_EXT)
H, W = imgs[0].shape[:2]

# Load all models
models_list = []
for ckpt in CKPT_PATHS:
    mdl = models.CellposeModel(gpu=True, pretrained_model=ckpt)
    models_list.append(mdl)

# =============================
# 4. Inference + Ensemble WMF
# =============================
proc_maps = []

for img in tqdm(imgs, desc="postproc + TTA + Ensemble WMF"):
    all_masks, all_boxes, all_scores = [], [], []

    for mdl in models_list:
        # Predict with TTA
        avg_prob_map, avg_dp_map = tta_predict_probability_and_flows(mdl, img, "")

        # Decode to dense instance map
        combined_mask = dynamics.resize_and_compute_masks(
            avg_dp_map,
            avg_prob_map,
            cellprob_threshold=0.0,
            flow_threshold=decode_flow_th,
            min_size=20,     # initial filter
            resize=(H, W),
            device=mdl.device
        )

        # Skip if no detection
        if combined_mask.max() == 0:
            continue

        # Convert to binary masks and boxes
        m_bin, bxs = instmap_to_masks_boxes(combined_mask)
        if m_bin is None:
            continue

        # Compute instance scores as mean probability over mask
        n_inst = m_bin.shape[0]
        inst_scores = []
        for k in range(n_inst):
            mask_k = m_bin[k].astype(bool)
            inst_scores.append(float(avg_prob_map[mask_k].mean()) if mask_k.sum() else 0.0)

        all_masks.append(m_bin)
        all_boxes.append(bxs)
        all_scores.append(np.array(inst_scores))

    # If no masks from any model, append empty map
    if not all_masks:
        proc_maps.append(np.zeros((H, W), dtype=np.int32))
        continue

    # Concatenate all model results
    masks_concat  = np.concatenate(all_masks, axis=0)
    boxes_concat  = np.concatenate(all_boxes, axis=0)
    scores_concat = np.concatenate(all_scores, axis=0)

    # Fuse masks using weighted mask fusion (NMW)
    fused_masks, fused_boxes = weighted_mask_fusion_nmw(
        masks_concat, boxes_concat, scores_concat,
        iou_thr=iou_thr_nmw, score_coef=score_coef
    )

    # Filter small masks
    if fused_masks.shape[0] > 0:
        areas = fused_masks.sum(axis=(1, 2))
        keep = np.where(areas > min_size)[0]
        fused_masks = fused_masks[keep]
        fused_boxes = fused_boxes[keep]

    # Apply convex hull for astrocytes if needed
    if corrupt and cell_type == 1 and fused_masks.shape[0] > 0:
        corrected = []
        for m in fused_masks:
            conts, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cvx_mask = np.zeros_like(m, dtype=np.uint8)
            for cnt in conts:
                hull = cv2.convexHull(cnt)
                cvx_mask = cv2.fillConvexPoly(cvx_mask, hull, 1)
            corrected.append(cvx_mask)
        fused_masks = np.stack(corrected, axis=0)

    # Final greedy removal of overlapping masks
    if fused_masks.shape[0] > 0:
        fused_masks = remove_overlap_naive(fused_masks)

    # Build final instance map
    canvas = np.zeros((H, W), dtype=np.int32)
    for m in fused_masks:
        canvas[m.astype(bool)] = canvas.max() + 1

    proc_maps.append(canvas)

# =============================
# 5. Evaluation (mAP)
# =============================
ths = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
aps = metrics.average_precision(gts, proc_maps, threshold=ths)[0]

print("\n>>> AP @ IoU 0.50–0.95")
for t, idx in zip(ths, range(len(ths))):
    print(f" IoU {t:.2f}: {aps[:, idx].mean():.3f}")
print(f"\n>>> final mAP = {aps.mean():.4f}")
