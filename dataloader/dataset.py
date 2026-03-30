import  numpy as np
import glob
from torch.utils.data import Dataset
from os.path import join
import os
import random
import torch
from sam2.utils.transforms import SAM2Transforms
from torch.nn import functional as F
import cv2
from PIL import Image


class NpyDataset(Dataset):
    def __init__(self, data_root, file_list=None, bbox_shift=20,):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "*.npy"), recursive=True)
        )
        self.gt_path_files = self.gt_path_files
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        
        self.gt_path_files = self.gt_path_files[:len(self.gt_path_files * 1)]
        self.bbox_shift = bbox_shift
        self._transform = SAM2Transforms(resolution=1024, mask_threshold=0)
        print(f"number of images: {len(self.gt_path_files)}")


    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = self._transform(img.copy())
        img_256 = F.interpolate(img_1024.view(-1,3,1024,1024), size=(256,256), mode='bilinear', align_corners=False).view(-1,256,256)
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        assert gt.shape == (256, 256), "ground truth should be 256x256"
        label_ids = np.unique(gt)[1:]
        gt2D = gt
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))

        bboxes = np.array([x_min, y_min, x_max, y_max])*4 ## scale bbox from 256 to 1024

        return (
            img_1024, ## [3, 1024, 1024]
            img_256,
            torch.tensor(gt2D).long(), ## [1, 256, 256]
            torch.tensor(bboxes).float(), 
            img_name,
        )
    
    
class ISICDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20, labeled_ratio=0.1):
        self.data_root = data_root
        self.labeled_ratio = labeled_ratio
        self.gt_path = join(data_root, "label")
        self.img_path = join(data_root, "image")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "*.png"), recursive=True)
        )

        # self.gt_path_files = self.gt_path_files[:20]
        self.bbox_shift = bbox_shift
        self._transform = SAM2Transforms(resolution=1024, mask_threshold=0)
        self._transform256 = SAM2Transforms(resolution=256, mask_threshold=0)
        print(f"number of images: {len(self.gt_path_files)}")
    
    def __len__(self):
        return len(self.gt_path_files)
    
    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index]).replace("_segmentation.png", ".jpg")
        img = Image.open(join(self.img_path, img_name))
        # convert the shape to (3, H, W)
        img_1024 = self._transform(img.copy())
        img_256  = self._transform256(img.copy())
        gt = np.array(Image.open(self.gt_path_files[index]))
        gt[gt == 255] = 1
        gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
        assert img_name == os.path.basename(self.gt_path_files[index]).replace("_segmentation.png", ".jpg"), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        assert gt.shape == (256, 256), "ground truth should be 256x256"
        gt2D = gt
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))

        bboxes = np.array([x_min, y_min, x_max, y_max])*4 ## scale bbox from 256 to 1024

        return (
            img_1024, ## [3, 1024, 1024]
            img_256, ## [3, 256, 256]
            torch.tensor(gt2D).long(), ## [1, 256, 256]
            torch.tensor(bboxes).float(), 
            img_name,
        )
