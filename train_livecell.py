import os
import argparse
from cellpose import models, core, io, plot, train
from pathlib import Path
from tqdm import trange
from custom_train import train_seg_with_callback
from torch.utils.tensorboard import SummaryWriter
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CellposeSAM on LiveCell dataset with custom training parameters."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="LiveCell_part1",
        help="Name of the model to save.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints",
        help="Path to save the model checkpoints.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=50,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="sartorius-cell-instance-segmentation/livecell_processed/train1/",
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="sartorius-cell-instance-segmentation/livecell_processed/val/",
        help="Directory containing validation images.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="Path to the pre-trained model checkpoints.",
    )
    parser.add_argument(
        "--masks_ext",
        type=str,
        default="_seg.npy",
        help="Extension for the mask files.",
    )
    args = parser.parse_args()
    # --------------------------- set up ----------------------------- #
    io.logger_setup()  # run this to get printing of progress

    # Check if colab notebook instance has GPU access
    if core.use_gpu() == False:
        raise ImportError("No GPU access, change your runtime")

    # ------------------------- training params ------------------------- #
    model_name = args.model_name
    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # default training params
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size

    train_dir = args.train_dir
    test_dir = args.test_dir
    if not Path(train_dir).exists():
        raise FileNotFoundError("directory does not exist")
    checkpoints_dir = args.checkpoints_dir
    masks_ext = "_seg.npy"

    # list all files
    files = [
        f
        for f in Path(train_dir).glob("*")
        if "_masks" not in f.name and "_flows" not in f.name and "_seg" not in f.name
    ]
    if len(files) == 0:
        raise FileNotFoundError(
            "no files found, did you specify the correct folder and extension?"
        )
    else:
        print(f"{len(files)} files in folder:")

    # --------------------------- set up tensorboard ----------------------------- #
    log_dir = os.path.join("runs", model_name)
    # create folder if not exists
    os.makedirs(log_dir, exist_ok=True)

    # create tensorboard writer
    writer = SummaryWriter(log_dir)

    # --------------------------- train model ----------------------------- #
    if checkpoints_dir is None:
        model = models.CellposeModel(gpu=True)
    else:
        model = models.CellposeModel(gpu=True, pretrained_model=checkpoints_dir)
    # get files
    output = io.load_train_test_data(train_dir, test_dir, mask_filter=masks_ext)
    train_data, train_labels, _, test_data, test_labels, _ = output

    new_model_path, train_losses, test_losses = train_seg_with_callback(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        nimg_per_epoch=max(2, len(train_data)),  # can change this
        save_every=5,
        save_path=save_path,
        save_each=True,
        model_name=model_name,
        tb_writer=writer,
        validate_every=1,
        min_train_masks=0,
    )
    print(f"model saved to {new_model_path}")
    # close tensorboard
    writer.close()
