import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold

# ref : https://github.com/CarnoZhao/mmdetection/blob/sartorius_solution/sartorius/data.ipynb
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint16)
    for m, enc in enumerate(encs):
        if isinstance(enc, float) and np.isnan(enc):
            continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2*i]) - 1
            length = int(s[2*i + 1])
            img[start : start + length] = 1 + m  # 1-based indexing
    return img.reshape(shape)

def produce_seg_npy(df, output_dir, image_dir="sartorius-cell-instance-segmentation/train", save_image=True):
    os.makedirs(output_dir, exist_ok=True)

    grouped = df.groupby('id')
    for image_id, group in tqdm(grouped, desc=f"Generating masks ({os.path.basename(output_dir)})"):
        height = group.iloc[0]['height']
        width = group.iloc[0]['width']
        rle_list = group['annotation'].tolist()  

        full_mask = enc2mask(rle_list, (height, width))
        # print(f"full_mask shape: {full_mask.shape}")

       
        mask_dict = {'masks': full_mask}

        # save as .npy
        out_path = os.path.join(output_dir, f"{image_id}_seg.npy")
        with open(out_path, 'wb') as f:
            np.save(f, mask_dict, allow_pickle=True)

        if save_image:
            img_path = os.path.join(image_dir, f"{image_id}.png")
            img = cv2.imread(img_path)
            if img is not None:
                out_img_path = os.path.join(output_dir, f"{image_id}.png")
                cv2.imwrite(out_img_path, img)
            else:
                print(f"[警告] 無法讀取圖片 {img_path}")


if __name__ == "__main__":
    # ----------------------------------------------------
    # 1. 讀入原始檔案與標籤
    # ----------------------------------------------------
    labels = pd.read_csv("sartorius-cell-instance-segmentation/train.csv")
    image_dir = "sartorius-cell-instance-segmentation/train"
    # 所有圖片的 id 清單（用來確認哪些 id 是存在的）
    file_names = glob.glob(os.path.join(image_dir, "*.png"))

    # ----------------------------------------------------
    # 2. 先取得每張圖片的 cell_type, width, height
    #    做為分層依據
    # ----------------------------------------------------
    image_info = labels.groupby("id").agg({
        "cell_type": lambda x: x.mode()[0],  # 多數類別當作該張圖的 cell_type
        "width": "first",
        "height": "first"
    }).reset_index()

    ids = image_info["id"].values
    types = image_info["cell_type"].values

    # ----------------------------------------------------
    # 3. 建立 StratifiedKFold（5-Fold）
    # ----------------------------------------------------
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(ids, types)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")

        # 取出 train_ids / val_ids
        train_ids = ids[train_idx]
        val_ids = ids[val_idx]

        # 依照 train_ids / val_ids 篩選原始 labels DataFrame
        train_df = labels[labels["id"].isin(train_ids)].reset_index(drop=True)
        val_df   = labels[labels["id"].isin(val_ids)].reset_index(drop=True)

        # 輸出該 fold 的統計資訊
        print(f"Fold {fold+1} - Train images：{train_df['id'].nunique()} 張，Val images：{val_df['id'].nunique()} 張")
        print("  Train cell_types 分佈：")
        print(train_df["cell_type"].value_counts().to_string())
        print("  Val cell_types 分佈：")
        print(val_df["cell_type"].value_counts().to_string())

        # ------------------------------------------------
        # 4. 產生 .npy 檔案（可針對每個 fold 建不同資料夾）
        # ------------------------------------------------
        # 比如： mytrain_fold0, myval_fold0, mytrain_fold1, myval_fold1, ...
        train_dir = f"sartorius-cell-instance-segmentation/mytrain_fold{fold}"
        val_dir   = f"sartorius-cell-instance-segmentation/myval_fold{fold}"

        produce_seg_npy(train_df, train_dir, image_dir)
        produce_seg_npy(val_df, val_dir, image_dir)
