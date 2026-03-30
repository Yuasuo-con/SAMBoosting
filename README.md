# Title

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Checkpoints](#model-checkpoints)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)

## Installation

### Requirements

**Core Dependencies:**
```bash
# Deep Learning Framework
pip install torch torchvision

# Medical Image Processing
pip install SimpleITK monai

# Utilities & Visualization
pip install numpy scipy tqdm tensorboardX matplotlib opencv-python Pillow

# 3D Connected Components Analysis
pip install connected-components-3d

# Transformer-based Models (for TransUNet and SwinUNet)
pip install einops timm ml_collections sympy
```

**Install SAM2 from Source:**

SAM2 (Segment Anything Model 2) be installed from source:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

**Note:** SAM2 requires compilation and may take a few minutes to install. Ensure you have CUDA toolkit installed for GPU support.

## Data Preparation

### For 3D CT/MR Datasets (e.g., Qilu-Spine)

1. **Organize your data** in the following structure:

```
data/
├── QiluSpine/
│   ├── images/
│   │   ├── _0000.nii.gz
│   │   └── ...
│   └── labels/
│       ├── .nii.gz
│       └── ...
```

2. **Step 1: Preprocess the 3D volumes to npz format**:

```bash
python pre_CT_MR.py \
    -modality CT \
    -anatomy Spine \
    -img_name_prefix "_0000.nii.gz" \
    -gt_name_prefix ".nii.gz" \
    -img_path "data/QiluSpine/images" \
    -gt_path "data/QiluSpine/labels" \
    -output_path "data/npz/QiluSpine" \
    -num_workers 4 \
    -window_level 40 \
    -window_width 400 \
    --save_nii
```

**Parameters:**

- `-modality`: CT or MR
- `-anatomy`: Anatomy + dataset name (e.g., Spine for Qilu-Spine)
- `-img_name_prefix`: Prefix of image files
- `-gt_name_prefix`: Prefix of ground truth files
- `-img_path`: Path to nii images
- `-gt_path`: Path to ground truth labels
- `-output_path`: Output path for preprocessed npz files
- `-num_workers`: Number of workers for preprocessing
- `-window_level`: CT window level (default: 40)
- `-window_width`: CT window width (default: 400)
- `--save_nii`: Save as nii files for sanity check (optional)

3. **Step 2: Convert npz files to npy format for training**:

For 3D datasets, you need to convert the preprocessed npz files to npy format by extracting slices:

```bash
python npz_to_npy.py \
    -npz_dir "data/npz/QiluSpine/npz_train/CT_Spine" \
    -npy_dir "data/npy/QiluSpine_train" \
    -num_workers 4
```

**Parameters:**

- `-npz_dir`: Path to the directory containing preprocessed npz files (training set)
- `-npy_dir`: Path to save the extracted npy files for training
- `-num_workers`: Number of workers for conversion

This script will:

- Extract each 3D volume into individual 2D slices
- Save images as 3-channel npy files in `data/npy/QiluSpine_train/imgs/`
- Save corresponding masks as single-channel npy files in `data/npy/QiluSpine_train/gts/`
- Resize all slices to 256x256 resolution

**Note:** The training pipeline expects data in npy format. Make sure to run this conversion step before training.

### For 2D Datasets (e.g., ISIC-2017)

Organize your ISIC-2017 dataset in the following structure:

```
/data1/dataset/cax/ISIC-2017/
├── train/
├── val/
└── test/
```

No preprocessing is needed for 2D datasets. Use the original data directly during training.

## Model Checkpoints

### Download SAM2 Checkpoints

```bash
cd checkpoints
bash download_ckpts.sh
```

This will download the SAM2 Hiera Large checkpoint. You can modify the script to download other variants:
- `sam2_hiera_tiny.pt`
- `sam2_hiera_small.pt`
- `sam2_hiera_base_plus.pt`
- `sam2_hiera_large.pt` (default)

### Download TransUNet Pretrained Weights

TransUNet requires ImageNet pretrained weights. Download them manually:

**For ViT-B/16:**
```bash
mkdir -p vit_checkpoint/imagenet21k
cd vit_checkpoint/imagenet21k
# Download from: https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

**For ResNet50 + ViT-B/16 (default):**
```bash
mkdir -p vit_checkpoint/imagenet21k
cd vit_checkpoint/imagenet21k
# Download from: https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
wget https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz -O "R50+ViT-B_16.npz"
```

### Download SwinUNet Pretrained Weights (Optional)

SwinUNet can use pretrained weights but also works without them:

```bash
mkdir -p swin_checkpoint
cd swin_checkpoint
# Download Swin Transformer pretrained model (optional)
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
```

### Directory Structure
After downloading, your directory structure should look like:
```
checkpoints/
└── sam2_hiera_large.pt

vit_checkpoint/
└── imagenet21k/
    ├── ViT-B_16.npz
    └── R50+ViT-B_16.npz

swin_checkpoint/
└── swin_tiny_patch4_window7_224.pth (optional)
```

## SAM2 Fine-Tuning

Before semi-supervised training, you need to fine-tune SAM2 on your target dataset using [sam2_trainer.py](file:///root/data/private/cax/sam_boosting/sam2_trainer.py). This step adapts the pre-trained SAM2 model to your specific medical image domain.

### Fine-Tuning on Qilu-Spine Dataset and ISIC-2017 Dataset

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    sam2_trainer.py \
    --tr_npy_path "data/npy/QiluSpine_train" \
    --val_npy_path "data/npy/QiluSpine_val" \
    --model_cfg "sam2_configs/sam2_hiera_l.yaml" \
    --pretrain_model_path "checkpoints/sam2_hiera_large.pt" \
    --task_name "MedSAM2_QiluSpine" \
    --work_dir "work_dir/sam2_finetuning" \
    --num_epochs 30 \
    --batch_size 8 \
    --lr 6e-5 \
    --weight_decay 0.01 \
    --bbox_shift 5 \
    --labeled_data 200 \
    --dataset "spine"
```

**Key Parameters:**

- `--tr_npy_path`: Path to training npy files (imgs/ and gts/ subfolders required)
- `--val_npy_path`: Path to validation npy files
- `--model_cfg`: SAM2 model configuration file
- `--pretrain_model_path`: Path to pre-trained SAM2 checkpoint
- `--task_name`: Task name for saving checkpoints
- `--work_dir`: Working directory to save fine-tuned models
- `--num_epochs`: Number of epochs for fine-tuning (default: 30)
- `--batch_size`: Batch size per GPU (default: 8)
- `--lr`: Learning rate (default: 6e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--bbox_shift`: Bounding box perturbation range (default: 5)
- `--labeled_data`: Number of labeled samples to use (spine: 1/2/4/8, skin: 20/50/100/200)
- `--dataset`: Dataset type (`spine` or `skin`)
- `--resume`: Path to resume training from checkpoint (optional)

**Notes:**

- The fine-tuning process uses distributed training with multiple GPUs
- Only the image encoder and mask decoder parameters are trainable (prompt encoder is frozen)
- Uses Dice + Cross Entropy loss for optimization
- Training progress and loss curves are saved in the work directory
- Best model (based on validation loss) is saved as `medsam_model_best.pth`

### Preparing Validation Data

For validation, you should split a small portion of your data:

```bash
# Example: Create validation set from training data
mkdir -p data/npy/QiluSpine_val/imgs
mkdir -p data/npy/QiluSpine_val/gts
cp data/npy/QiluSpine_train/imgs/*.npy data/npy/QiluSpine_val/imgs/ | head -20
cp data/npy/QiluSpine_train/gts/*.npy data/npy/QiluSpine_val/gts/ | head -20
```

### Using Fine-Tuned Model in Semi-Supervised Training

After fine-tuning, use the fine-tuned model as the pre-trained weight for semi-supervised training:

```bash
# Use the fine-tuned model instead of the original SAM2 checkpoint
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    semi_train.py \
    ... \
    --pretrain_model_path "work_dir/sam2_finetuning/MedSAM2_QiluSpine-*/medsam_model_best.pth" \
    ...
```

## Training

### Semi-Supervised Training with SAM2 Boosting

Train with labeled and unlabeled data:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    semi_train.py \
    --model transUNet \
    --labeled_ratio 0.01 \
    --batch_size 24 \
    --base_lr 0.01 \
    --max_epochs 200 \
    --data_path "data/npy/QiluSpine_train" \
    --pretrain_model_path "checkpoints/sam2_hiera_large.pt" \
    --model_cfg "sam2_configs/sam2_hiera_l.yaml" \
    --work_dir "work_dir/transUNet_QiluSpine"
```

**Key Parameters:**
- `--model`: Segmentation model type (`UNet`, `transUNet`, `SwinUNet`)
- `--labeled_ratio`: Ratio of labeled data (e.g., 0.01, 0.05, 0.1)
- `--batch_size`: Batch size per GPU
- `--base_lr`: Base learning rate
- `--max_epochs`: Maximum number of training epochs
- `--data_path`: Path to preprocessed npy data (output from Step 2)
- `--pretrain_model_path`: Path to SAM2 checkpoint (or fine-tuned MedSAM2 model)
- `--model_cfg`: SAM2 model configuration file
- `--work_dir`: Working directory to save checkpoints and logs
- `--alpha`: EMA momentum (default: 0.99)
- `--consistency`: Consistency loss weight (default: 0.1)
- `--consistency_rampup`: Consistency ramp-up epochs (default: 50)

**Important Notes:**
- **For TransUNet**: Make sure you have downloaded the ImageNet pretrained weights (`R50+ViT-B_16.npz`) in `vit_checkpoint/imagenet21k/` directory. The model will automatically load these weights during initialization.
- **For SwinUNet**: Pretrained weights are optional. The model can be trained from scratch.
- **For UNet**: No pretrained weights needed, trains from scratch.
- If using a fine-tuned MedSAM2 model, replace `--pretrain_model_path` with the path to your fine-tuned checkpoint.

### Training without SAM2 (Baseline)

```bash
python semi_train_without_sam.py \
    --model transUNet \
    --labeled_ratio 0.01 \
    --batch_size 24 \
    --base_lr 0.01 \
    --max_epochs 200 \
    --data_path "data/npy/QiluSpine_train" \
    --work_dir "work_dir_without_sam2/transUNet_QiluSpine"
```

## Inference

### Test on 3D CT/MR Datasets (Qilu-Spine)

```bash
python semi_test.py \
    --model transUNet \
    --checkpoint_path "work_dir/transUNet_QiluSpine/_latest_0.01.pth" \
    --data_path "data/npz/QiluSpine/npz_test/CT_Spine" \
    --device "cuda:0" \
    --save_dir "results/QiluSpine_predictions"
```

**Parameters:**

- `--model`: Model type used for training
- `--checkpoint_path`: Path to trained checkpoint
- `--data_path`: Path to test data in npz format (from Step 1)
- `--device`: CUDA device (e.g., "cuda:0")
- `--save_dir`: Directory to save predictions

### Test on ISIC-2017 Dataset

```bash
python semi_testISIC.py \
    --model transUNet \
    --checkpoint_path "work_dir_without_sam2/transUNet/50case_labeled_large_trian/_latest_0.01.pth" \
    --save_dir "/data1/dataset/cax/ISIC-2017/TransUNet_SAMBoosting_50case_labeled" \
    --device "cuda:0"
```

## Evaluation

The evaluation metrics are automatically computed during inference:

- **Dice Score (DSC)**
- **Intersection over Union (IoU)**
- **Hausdorff Distance 95% (HD95)**
- **Average Surface Distance (ASD)**

Results will be printed to console and saved in the specified output directory.

## Supported Models

- **UNet**: Standard 2D U-Net
- **TransUNet**: Vision Transformer-based U-Net
- **SwinUNet**: Swin Transformer-based U-Net

