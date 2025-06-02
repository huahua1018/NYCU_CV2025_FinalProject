import time
import os
import logging
import numpy as np
from pathlib import Path
from tqdm import trange

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from cellpose import io, utils, models, dynamics, train
from cellpose.transforms import normalize_img, random_rotate_and_resize

train_logger = logging.getLogger(__name__)

def train_seg_with_callback(
        net,
        train_data=None, train_labels=None, train_files=None,
        train_labels_files=None, train_probs=None,
        test_data=None, test_labels=None, test_files=None,
        test_labels_files=None, test_probs=None,
        channel_axis=None, load_files=True,
        batch_size=1, learning_rate=5e-5, SGD=False,
        n_epochs=100, weight_decay=0.1,
        normalize=True, compute_flows=False,
        save_path=None, save_every=100, save_each=False,
        nimg_per_epoch=None, nimg_test_per_epoch=None,
        rescale=False, scale_range=None, bsize=256,
        min_train_masks=5, model_name=None, class_weights=None,
        callback=None, validate_every=5, tb_writer=None,
        augment_fn=None, use_amp=True
    ):
    """
    Cellpose segmentation training with:
      • custom callback
      • AMP support (set use_amp=False to disable)
    """
    if SGD:
        train_logger.warning("SGD is deprecated; using AdamW")

    device = net.device
    scale_range = 0.5 if scale_range is None else scale_range

    # Prepare normalization parameters
    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif isinstance(normalize, bool):
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize
    else:
        raise ValueError("normalize must be bool or dict")

    # Preprocess data
    result = train._process_train_test(
        train_data=train_data, train_labels=train_labels,
        train_files=train_files, train_labels_files=train_labels_files,
        train_probs=train_probs,
        test_data=test_data, test_labels=test_labels,
        test_files=test_files, test_labels_files=test_labels_files,
        test_probs=test_probs,
        load_files=load_files, min_train_masks=min_train_masks,
        compute_flows=compute_flows, channel_axis=channel_axis,
        normalize_params=normalize_params, device=device
    )
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = result

    kwargs = {} if normed else {"normalize_params": normalize_params,
                                "channel_axis": channel_axis}

    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    if class_weights is not None and isinstance(class_weights, (list, np.ndarray, tuple)):
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = (len(test_data) if test_data is not None else
                 (len(test_files) if test_files is not None else None))
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # Learning rate schedule: warm-up + cosine decay
    warmup_epochs = 5
    base_lr = learning_rate
    eta_min = 1e-7

    LR = np.zeros(n_epochs, dtype=np.float32)
    for e in range(min(warmup_epochs, n_epochs)):
        LR[e] = base_lr * (e + 1) / warmup_epochs

    if n_epochs > warmup_epochs:
        t = np.arange(0, n_epochs - warmup_epochs)
        cos_part = eta_min + 0.5 * (base_lr - eta_min) * \
                   (1 + np.cos(np.pi * t / (n_epochs - warmup_epochs)))
        LR[warmup_epochs:] = cos_part

    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler(enabled=use_amp)

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    model_dir = save_path / model_name
    model_dir.mkdir(exist_ok=True)
    filename = model_dir / model_name
    train_logger.info(f"Saving checkpoints to {filename}")

    train_losses = np.zeros(n_epochs, dtype=np.float32)
    test_losses = np.zeros(n_epochs, dtype=np.float32)

    for iepoch in trange(n_epochs, desc="Epoch", ncols=100):
        np.random.seed(iepoch)
        rperm = (np.random.choice(np.arange(nimg), nimg_per_epoch, p=train_probs)
                 if nimg != nimg_per_epoch else np.random.permutation(np.arange(nimg)))

        for pg in optimizer.param_groups:
            pg["lr"] = LR[iepoch]

        net.train()
        epoch_train_loss, nsamples = 0.0, 0

        for k in trange(0, nimg_per_epoch, batch_size, leave=False, desc="Train", ncols=100):
            inds = rperm[k:k+batch_size]
            imgs, lbls = train._get_batch(inds,
                                          data=train_data, labels=train_labels,
                                          files=train_files, labels_files=train_labels_files,
                                          **kwargs)
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item() if rescale else np.ones_like(diams)

            if augment_fn is not None:
                imgs, lbls = augment_fn(imgs, lbls, epoch=iepoch)

            imgi, lbl = random_rotate_and_resize(imgs, Y=lbls,
                                                 rescale=rsc, scale_range=scale_range,
                                                 xy=(bsize, bsize))[:2]

            X = torch.from_numpy(imgi).to(device)
            lbl = torch.from_numpy(lbl).to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                y = net(X)[0]
                loss = train._loss_fn_seg(lbl, y, device)
                if y.shape[1] > 3:
                    loss += train._loss_fn_class(lbl, y, class_weights=class_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item() * len(imgi)
            epoch_train_loss += batch_loss
            nsamples += len(imgi)

        train_losses[iepoch] = epoch_train_loss / nsamples

        val_loss_mean = 0.0
        if (test_data is not None or test_files is not None) and \
           (iepoch % validate_every == 0 or iepoch == n_epochs - 1):

            np.random.seed(42)
            rperm = (np.random.choice(np.arange(nimg_test), nimg_test_per_epoch, p=test_probs)
                     if nimg_test != nimg_test_per_epoch else np.random.permutation(np.arange(nimg_test)))

            net.eval()
            val_sum, v_nsamp = 0.0, 0
            with torch.no_grad():
                for k in trange(0, len(rperm), batch_size, leave=False, desc="Val", ncols=100):
                    inds = rperm[k:k+batch_size]
                    imgs, lbls = train._get_batch(inds,
                                                  data=test_data, labels=test_labels,
                                                  files=test_files, labels_files=test_labels_files,
                                                  **kwargs)
                    diams = np.array([diam_test[i] for i in inds])
                    rsc = diams / net.diam_mean.item() if rescale else np.ones_like(diams)

                    imgi, lbl = random_rotate_and_resize(imgs, Y=lbls,
                                                         rescale=rsc, scale_range=scale_range,
                                                         xy=(bsize, bsize))[:2]
                    X = torch.from_numpy(imgi).to(device)
                    lbl = torch.from_numpy(lbl).to(device)

                    with autocast(enabled=use_amp):
                        y = net(X)[0]
                        vloss = train._loss_fn_seg(lbl, y, device)
                        if y.shape[1] > 3:
                            vloss += train._loss_fn_class(lbl, y, class_weights=class_weights)
                    val_sum += vloss.item() * len(imgi)
                    v_nsamp += len(imgi)

            val_loss_mean = val_sum / v_nsamp
            test_losses[iepoch] = val_loss_mean
        else:
            test_losses[iepoch] = test_losses[iepoch - 1] if iepoch else 0

        if callback is not None:
            callback.on_epoch_end(iepoch, train_losses[iepoch],
                                  test_losses[iepoch], LR[iepoch])

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_losses[iepoch], iepoch)
            tb_writer.add_scalar("Loss/valid", test_losses[iepoch], iepoch)
            tb_writer.add_scalar("LR", LR[iepoch], iepoch)

        if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch):
            ckpt_name = (f"{filename}_epoch_{iepoch:04d}"
                         if (save_each and iepoch != n_epochs - 1) else filename)
            net.save_model(str(ckpt_name))
            train_logger.info(f"Saved model to {ckpt_name}")

    net.save_model(str(filename))
    train_logger.info(f"Final model saved to {filename}")
    return str(filename), train_losses, test_losses
