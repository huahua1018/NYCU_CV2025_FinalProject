import os
import numpy as np
import torch
from pathlib import Path
from cellpose import io, models, core, metrics
from torch.utils.tensorboard import SummaryWriter
from custom_train import train_seg_with_callback
from check_bad_patch import filter_invalid_patches  # remove if not filtering
from augment import Augmenter  # remove if not using patch-based augment

# Setup environment
io.logger_setup()
if not core.use_gpu():
    raise ImportError("No GPU detected; please use GPU runtime")

model_name = "fullres_train"
save_path = Path("fullres_ckpt")
save_path.mkdir(exist_ok=True)

# Training parameters
n_epochs = 100
learning_rate = 1e-5
weight_decay = 0.1
batch_size = 2

train_dir = Path("sartorius-cell-instance-segmentation/mytrain")
test_dir = Path("sartorius-cell-instance-segmentation/myval")
masks_ext = "_seg.npy"

if not train_dir.exists() or not test_dir.exists():
    raise FileNotFoundError("train_dir or test_dir not found")

# Load full-size images and masks
train_imgs, train_msks, _, val_imgs, val_msks, _ = io.load_train_test_data(
    str(train_dir), str(test_dir),
    mask_filter=masks_ext,
    image_filter=""
)
print(f"Loaded images → train: {len(train_imgs)}, val: {len(val_imgs)}")

# Use full images and masks directly
train_data, train_labels = train_imgs, train_msks
val_data, val_labels = val_imgs, val_msks

# Filter invalid masks if desired
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data, train_labels, bad_train = filter_invalid_patches(
    train_data, train_labels, prefix='train', device=device
)
print(f"After filtering → train: {len(train_data)} (removed {len(bad_train)})")
if len(train_data) == 0:
    raise RuntimeError("No training data left after filtering!")

# Initialize augment function (no mosaic used here)
train_aug = Augmenter(mosaic_prob=0.0)

def scheduled_augment(imgs, masks, epoch=0):
    # No mosaic; can add other augment before epoch 50 if needed
    if epoch >= 0:
        return imgs, masks
    return train_aug(imgs, masks, epoch=epoch)

# Setup TensorBoard callback
log_dir = Path("runs") / model_name
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(str(log_dir))

class MAPCallback:
    """Log losses, LR, and compute mAP on full-res validation set."""
    def __init__(self, model, val_imgs, val_msks, writer):
        self.model = model
        self.val_imgs = val_imgs
        self.val_msks = val_msks
        self.writer = writer

    def on_epoch_end(self, epoch, train_loss, val_loss, lr):
        # Log train/val loss and learning rate
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("LR", lr, epoch)

        # Inference on validation set
        with torch.no_grad():
            preds_list, *_ = self.model.eval(self.val_imgs, batch_size=1, diameter=None)

        # Compute COCO-style average precision
        ap_array = metrics.average_precision(self.val_msks, preds_list)[0]
        # Log mAP@0.50 and mAP@[0.50:0.95]
        self.writer.add_scalar("mAP50/val", ap_array[:, 0].mean(), epoch)
        self.writer.add_scalar("mAP50_95/val", ap_array.mean(), epoch)

# Build model and register callback
model = models.CellposeModel(gpu=True)
net = model.net.to(device)
map_cb = MAPCallback(model, val_data, val_labels, writer)

# Run training
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
    augment_fn=scheduled_augment,
)

print(" Training complete. Model saved to:", new_model_path)
writer.close()
