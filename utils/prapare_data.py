import numpy as np
import SimpleITK as sitk
import os
join = os.path.join
import test
from tqdm import tqdm
import cc3d
import multiprocessing as mp
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-img_dir",type=str,default="data/CT-FENGE/original_data")
parser.add_argument("-label_dir",type=str,default="data/CT-FENGE/3classes_mask")
parser.add_argument("-train_data_dir",type=str,default="data/CT-FENGE/train")
parser.add_argument("-test_data_dir",type=str,default="data/CT-FENGE/test")
args = parser.parse_args()

if os.path.exists(args.train_data_dir):
    print("train data dir exists")
else:
    os.makedirs(args.train_data_dir)

if os.path.exists(args.test_data_dir):
    print("test data dir exists")
else:
    os.makedirs(args.test_data_dir)

img_files = sorted(os.listdir(args.img_dir))
print("file numbers:",img_files.__len__())
img_files = [
    os.path.join(args.img_dir,f) for f in img_files
]
gt_file = sorted(os.listdir(args.label_dir))
gt_files = [
    os.path.join(args.label_dir,f) for f in gt_file
]

train_img_files = img_files[:int(img_files.__len__()*0.7)]
train_gt_files = gt_files[:int(gt_files.__len__()*0.7)]
# process train data
for img_file,gt_file in tqdm(zip(train_img_files,train_gt_files)):
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
    gt = gt.astype(np.uint8)
    # exclude the objects with less than 1000 pixels in 3D
    gt = cc3d.dust(
        gt, threshold=1000, connectivity=26, in_place=True
    )

    # remove small objects with less than 100 pixels in 2D slices
    for slice_i in range(gt.shape[0]):
        gt_i = gt[slice_i, :, :]
        # remove small objects with less than 100 pixels
        # reason: fro such small objects, the main challenge is detection rather than segmentation
        gt[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=100, connectivity=8, in_place=True
        )
    
    img_sitk = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(img_sitk)
    lower_bound, upper_bound = np.percentile(
        img[img > 0], 0.5
    ), np.percentile(img[img > 0], 99.5)
    image_data_pre = np.clip(img, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / (np.max(image_data_pre) - np.min(image_data_pre))
        * 255.0
    )
    image_data_pre[img == 0] = 0
    image_data_pre = np.uint8(image_data_pre)
    np.savez_compressed(join(args.train_data_dir, os.path.basename(img_file).replace(".nii.gz",".npz")), imgs=image_data_pre, gts=gt, spacing=img_sitk.GetSpacing())
    
# process test data
test_img_files = img_files[int(img_files.__len__()*0.7):]
test_gt_files = gt_files[int(gt_files.__len__()*0.7):]
for img_file,gt_file in tqdm(zip(test_img_files,test_gt_files)):
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
    gt = gt.astype(np.uint8)
    # exclude the objects with less than 1000 pixels in 3D
    gt = cc3d.dust(
        gt, threshold=1000, connectivity=26, in_place=True
    )

    # remove small objects with less than 100 pixels in 2D slices
    for slice_i in range(gt.shape[0]):
        gt_i = gt[slice_i, :, :]
        # remove small objects with less than 100 pixels
        # reason: fro such small objects, the main challenge is detection rather than segmentation
        gt[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=100, connectivity=8, in_place=True
        )
    
    img_sitk = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(img_sitk)
    lower_bound, upper_bound = np.percentile(
        img[img > 0], 0.5
    ), np.percentile(img[img > 0], 99.5)
    image_data_pre = np.clip(img, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / (np.max(image_data_pre) - np.min(image_data_pre))
        * 255.0
    )
    image_data_pre[img == 0] = 0
    image_data_pre = np.uint8(image_data_pre)
    np.savez_compressed(join(args.test_data_dir, os.path.basename(img_file).replace(".nii.gz",".npz")), imgs=image_data_pre, gts=gt, spacing=img_sitk.GetSpacing())