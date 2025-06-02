import os
import numpy as np
import torch
from pathlib import Path
from cellpose import io, models, core, metrics
from torch.utils.tensorboard import SummaryWriter
from custom_train import train_seg_with_callback
from check_bad_patch import filter_invalid_patches
from augment import Augmenter  

# ---------------------- the ckpt that u wanna load ----------------------
resume_weights = "LiveCell_epoch50_bs2_norm_part2"
# ----------------------------------------------------------------------------

# --------------------------- helper: split into patches ----------------------------- #
def split_into_patches(img_list, mask_list, pw=352, ph=260):
    patches_img, patches_msk = [], []
    for img, msk in zip(img_list, mask_list):
        H, W = img.shape[:2]
        ys = [0, H - ph]; xs = [0, W - pw]
        for y0 in ys:
            for x0 in xs:
                patches_img.append(img[y0:y0 + ph, x0:x0 + pw])
                patches_msk.append(msk[y0:y0 + ph, x0:x0 + pw])
    return patches_img, patches_msk

# --------------------------- environment setup ----------------------------- #
io.logger_setup()
if not core.use_gpu():
    raise ImportError("No GPU detected; please switch to a GPU runtime")

model_name = "patch_train"
save_path = Path("based_on_livecell")
save_path.mkdir(exist_ok=True)

# Training hyperparameters
n_epochs      = 100
learning_rate = 1e-5
weight_decay  = 0.1
batch_size    = 2

train_dir = Path("sartorius-cell-instance-segmentation/mytrain")
test_dir  = Path("sartorius-cell-instance-segmentation/myval")
masks_ext = "_seg.npy"

if not train_dir.exists() or not test_dir.exists():
    raise FileNotFoundError("train_dir or test_dir not found")

# --------------------------- load full-size images & masks ----------------------------- #
train_imgs, train_msks, _, val_imgs, val_msks, _ = io.load_train_test_data(
    str(train_dir), str(test_dir),
    mask_filter=masks_ext,
    image_filter=""
)
print(f"Loaded full images → train: {len(train_imgs)}, val: {len(val_imgs)}")

# --------------------------- slice only training set into patches ----------------------------- #
train_data, train_labels = split_into_patches(train_imgs, train_msks, 352, 260)
print(f"Sliced training into patches → train: {len(train_data)}")

# --------------------------- keep validation set as full images ----------------------------- #
val_data, val_labels = val_imgs, val_msks

# --------------------------- filter out invalid training patches ----------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data, train_labels, bad_train = filter_invalid_patches(
    train_data, train_labels, prefix='train', device=device
)
print(f"After filtering → train: {len(train_data)} (removed {len(bad_train)})")
if len(train_data) == 0:
    raise RuntimeError("No training data left after filtering!")

# ---------------------- initialize Augmenter ----------------------
train_aug = Augmenter(mosaic_prob=0.0)

def scheduled_augment(imgs, masks, epoch=0):
    if epoch < 50:
        return train_aug(imgs, masks, epoch=epoch)
    # return imgs, masks
    return train_aug(imgs, masks, epoch=epoch)  

# --------------------------- set up TensorBoard callback ----------------------------- #
log_dir = Path("runs") / model_name
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(str(log_dir))

class MAPCallback:
    def __init__(self, model, val_imgs, val_msks, writer):
        self.model    = model
        self.val_imgs = val_imgs
        self.val_msks = val_msks
        self.writer   = writer

    def on_epoch_end(self, epoch, train_loss, val_loss, lr):
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val",   val_loss,   epoch)
        self.writer.add_scalar("LR",         lr,         epoch)
        with torch.no_grad():
            preds_list, *_ = self.model.eval(
                self.val_imgs, batch_size=1, diameter=None
            )
        ap_array = metrics.average_precision(self.val_msks, preds_list)[0]
        self.writer.add_scalar("mAP50/val",    ap_array[:,0].mean(), epoch)
        self.writer.add_scalar("mAP50_95/val", ap_array.mean(),      epoch)

# --------------------------- build model & register callback ----------------------------- #
if resume_weights:
    model = models.CellposeModel(gpu=True, pretrained_model=resume_weights)
    print(f"Loaded pretrained weights from {resume_weights}")
else:
    model = models.CellposeModel(gpu=True)

net   = model.net.to(device)
map_cb = MAPCallback(model, val_data, val_labels, writer)

# --------------------------- run training ----------------------------- #
new_model_path, train_losses, val_losses = train_seg_with_callback(
    net,
    train_data=train_data,
    train_labels=train_labels,
    test_data=val_data,
    test_labels=val_labels,
    batch_size=batch_size,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    nimg_per_epoch=len(train_data),
    save_every=1,
    save_each=True,
    save_path=save_path,
    model_name=model_name,
    callback=map_cb,
    validate_every=1,
    min_train_masks=0,
    normalize=True,
    augment_fn=scheduled_augment,  # ← augment_fn
)

print("✅ Training complete. Model saved to:", new_model_path)
writer.close()
