# NYCU Computer Vision 2025 Spring FinalProject
Team member: 313551093 盧品樺, 313551052 王嘉羽, 313551127 王翔, 313553024 蘇柏叡

## Introduction

For our final project, we participated in the [Sartorius - Cell Instance Segmentation Kaggle competition](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/overview), which focuses on segmenting neuronal cells in microscopy images. The evaluation metric used in the competition is mean Average Precision (mAP). In our project, we primarily used **Cellpose-SAM**, the latest version of Cellpose, and integrated techniques such as **ensemble learning, Test-Time Augmentation (TTA)**, and **Weighted Mask Fusion (WMF)** to improve the final mAP score.

## Setup and Preparation Process
### Step 1 : Environment Setup
Our project leverages **cellposeSAM**. 
For installation guidance, please refer to the official instructions provided in the [cellpose](https://github.com/MouseLand/cellpose) repository.

Other required dependencies can be installed using one of the following methods:
#### 1. Use conda (Optimal)
```
conda env create -f environment.yml 
```

#### 2. Use pip
```
pip install -r requirements.txt
```

### Step 2 : Dataset Preparation
Download the dataset from [Sartorius - Cell Instance Segmentation](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data) and 
place the extracted **sartorius-cell-instance-segmentation** folder inside the **NYCU_CV2025_FinalProject** directory as shown below:

> NYCU_CV2025_FinalProject / <br>
>├── sartorius-cell-instance-segmentation / <br>

## Usage
### Preprocess
#### Pretrained on LiveCell dataset(Optional)
Some of our methods benefit from being pretrained on the LiveCell dataset before training on the Sartorius - Cell Instance Segmentation dataset, while others may perform worse after pretraining.
Therefore, pretraining is optional and can be enabled or disabled depending on the method used.
##### Step 1: Prepare LiveCell dataset
Since CellposeSAM requires masks to be saved in the {filename}_seg.npy format, you need to preprocess the LIVECell dataset by running the following script:

```
python prepare_livecell_dataset.py --base_dir <Path to the LIVECell dataset> --output_dir <Path to save processed dataset>               
```

##### Step 2: Split training data into two part
Because the LIVECell dataset is too large to train CellposeSAM efficiently, we split it into two smaller parts using the following script:

```
python prepare_livecell_dataset.py --data_dir <Path to the processed LIVECell dataset>  --part1_dir <Path to save the first part of the dataset>  --part2_dir <Path to save the second part of the dataset>        
```

##### Step 3: Pretrained on LiveCell dataset

```
python train_livecell.py --train_dir <Path to training data> --test_dir<Path to validation data> --checkpoints_dir <Path to the pre-trained model checkpoints> --model_name <Name of the model to save> --save_path <Path to save the model checkpoints> 
```

#### Prepare training dataset
Since CellposeSAM requires masks to be saved in the {filename}_seg.npy format,
and the official training data does not include a validation split,
we need to run the following script to generate the required files:

```
python preprocess.py --img_dir <Path to training data> --label_csv<Path to train.csv> --train_dir <Path tosave training set> --val_dir <Path to save validation set>
```

### Train

Run the following command to split the images into patches and start training:

```
python patch_train.py
```

If you want to train on full image instead of patches, you can run following script:

```
python fullres_train.py
```

### Test

### Visualization
We provide a Python script that displays the original image, ground truth, and predictions from two different models side by side for easy comparison：

```
python visualization_compare.py --model_path1 <Path to the first model checkpoint> --model_path1 <Path to the second model checkpoint> --test_dir <Path to validation set> 
```

You will get an output like the image below:
![mask_overlay_pretrained_livecell_barry](https://github.com/user-attachments/assets/8b126c64-1fe0-404f-b396-936b9842bfd0)


## Performance snapshot
