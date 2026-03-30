from UNet.UNet import UNet
from transUnet.vit_seg_modeling import VisionTransformer as transUNet
from transUnet.vit_seg_configs import get_r50_b16_config
from SwinUNet.vision_transformer import SwinUnet as SwinUNet
from dataloader.dataset import ISICDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils.util_2D import *
import numpy as np
import argparse
import cv2
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default="transUNet", help="model name")
parser.add_argument("-checkpoint_path", type=str, default="work_dir_without_sam2/transUNet/50case_labeled_large_trian/_latest_0.01.pth", help="checkpoint path")
parser.add_argument("-save_dir", type=str, default="work_dir/ISIC_predictions", help="Directory to save predictions")
parser.add_argument("-device", type=str, default="cuda:0")
args = parser.parse_args()

if args.model == 'UNet':
    model = UNet(3, 2)
elif args.model == 'transUNet':
    config_vit = get_r50_b16_config()
    config_vit.n_classes = 2
    config_vit.n_skip = 3
    config_vit.patches.grid = (16, 16)
    model = transUNet(config_vit, img_size=256, num_classes=2)

    # load VIT pretrained weights
    model.load_from(weights=np.load(config_vit.pretrained_path))
elif args.model == 'SwinUNet':
    config = None
    model = SwinUNet(config, img_size=256, num_classes=2)
else:
    raise NotImplementedError

checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(checkpoint)
dataset = ISICDataset(data_root='data/ISIC-2017/test')
dataloader = DataLoader(dataset=dataset, batch_size=24, shuffle=True)
device = args.device
save_dir = args.save_dir

iou_scores = []
dice_scores = []
hd95_scores = []
asd_scores = []

count = 0

os.makedirs(os.path.join(save_dir, 'pred'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)

model.eval()
model.to(device)
for i, (_, img, gt, _, _) in tqdm(enumerate(dataloader)):
    img = img.to(device)
    gt = gt.to(device)
    out = model(img)
    out = torch.argmax(out, dim=1)
    img = img.to('cpu').numpy()
    out = out.to('cpu').numpy()
    gt = gt.to('cpu').numpy()
    batch = out.shape[0]
    for j in range(batch):
        if gt[j].max() != 0 and out[j].max() != 0:
            pred = out[j] * 255
            pred = pred.astype(np.uint8)
            ground_truth = gt[j] * 255
            ground_truth = ground_truth.astype(np.uint8)

            pred_mask = Image.fromarray(pred, mode='L')  # 'L' 表示灰度图
            gt_mask = Image.fromarray(ground_truth, mode='L')

            pred_mask.save(os.path.join(save_dir, 'pred', f'{count}.png'))
            gt_mask.save(os.path.join(save_dir, 'gt', f'{count}.png'))

            count = count + 1

            # image = cv2.imread('data/ISIC-2017/one_test/image/ISIC_0013457.jpg')       # BGR
            # img256 = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            # # img: (H, W, 3) uint8   mask: (H, W) 0/255
            # mask_color = np.array([0, 0, 255], dtype=np.uint8)  # 红色
            # mask_rgb   = mask_color * (pred > 0)[..., None]      # H×W×3

            # alpha = 0.4
            # overlay = cv2.addWeighted(img256, 1-alpha, mask_rgb, alpha, 0)

            # # overlay 就是叠加后的图，直接保存或 imshow
            # cv2.imwrite('overlay.png', overlay)
#             iou_scores.append(compute_iou_scores(out[j], gt[j]))
#             dice_scores.append(compute_dice_scores(out[j], gt[j]))
#             hd95_scores.append(hausdorff_distance(out[j], gt[j]))
#             asd_scores.append(average_symmetric_surface_distance(out[j], gt[j]))
# iou_scores = np.array(iou_scores).flatten()
# dice_scores = np.array(dice_scores).flatten()
# hd95_scores = np.array(hd95_scores).flatten()
# asd_scores = np.array(asd_scores).flatten()

# print(args.model)
# print(args.checkpoint_path)

# print(f'average dice = {np.mean(dice_scores):.5f}')
# print(f'std dice = {np.std(dice_scores):.5f}')
# print(f'average iou = {np.mean(iou_scores):.5f}')
# print(f'std iou = {np.std(iou_scores):.5f}')
# print(f'average asd = {np.mean(asd_scores):.5f}')
# print(f'std asd = {np.std(asd_scores):.5f}')
# print(f'average hd95 = {np.mean(hd95_scores):.5f}')
# print(f'std hd95 = {np.std(hd95_scores):.5f}')




    