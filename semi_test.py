from UNet.UNet import UNet
from transUnet.vit_seg_modeling import VisionTransformer as transUNet
from transUnet.vit_seg_configs import get_r50_b16_config
from SwinUNet.vision_transformer import SwinUnet as SwinUNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from semi_train import set_seed
from tqdm import tqdm
from os.path import join
from os import listdir
import SimpleITK as sitk 
import argparse
from PIL import Image
from sam2.utils.transforms import SAM2Transforms
from utils.util import compute_iou_scores, compute_dice_scores, compute_asd_all_classes, compute_hd95_all_classes, remove_all_small_regions

def s_model_inference(model:nn.Module, image:torch.Tensor, device:str)-> np.ndarray:

    image_256 = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
    image_256 = image_256.to(device)
    model.eval()
    with torch.no_grad():
        pred_logit_256 = model(image_256)
        pred_logit_256 = torch.softmax(pred_logit_256, dim=1)

        # upsample to original resolution
        pred_logit_512 = F.interpolate(
        pred_logit_256,
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
        )  # (1, 4, 512, 512)

        # pred_logit_256 = F.softmax(pred_logit_256, dim=1)
        # sample = torch.argmax(pred_logit_256[0], dim=0).cpu().numpy().astype(np.uint8)
        # get_image = Image.fromarray(sample*80)
        # get_image.save("test_s.png")

        pred_logit_512 = pred_logit_512.squeeze().cpu().numpy()
        pred_512 = np.argmax(pred_logit_512, axis=0).astype(np.uint8) # (512, 512)

        # get_image = Image.fromarray(pred_512*80)
        # get_image.save("test_s.png")

        return pred_512

def main(args):
    set_seed(2024)
    device = args.device
    if args.model == "UNet":
        model = UNet(in_channels=3, out_channels=4)
    elif args.model == "transUNet":
        config_vit = get_r50_b16_config()
        config_vit.n_classes = 4
        config_vit.n_skip = 3
        config_vit.patches.grid = (16, 16)
        model = transUNet(config_vit, img_size=256, num_classes=4)

        # load VIT pretrained weights
        # model.load_from(weights=np.load(config_vit.pretrained_path))
    elif args.model == "SwinUNet":
        config = None
        model = SwinUNet(config, img_size=256, num_classes=4)

    else:
        raise NotImplementedError
    model = model.to(device)

    ckpt = torch.load(args.checkpoint_path, map_location=torch.device(device), weights_only=True)#['model']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    print("Loaded checkpoint sucessfully")

    _transfer = SAM2Transforms(resolution=1024, mask_threshold=0)


    _names = sorted(listdir(args.data_root))
    names = [name for name in _names if name.endswith('.npz')]
    # names = names[:1]
    # names = ['MR_Abd_3classesmask_case50.npz']

    iou_scores = []
    dice_scores = []
    asd_results = []
    hd95_results = []
    for name in tqdm(names):
        npz = np.load(join(args.data_root, name), allow_pickle=True)
        spacing = npz['spacing']
        spacing_xyz = spacing[::-1]
        img_3D = npz['imgs']
        gt_3D = npz['gts']
        pred_3D = np.zeros_like(img_3D, dtype=np.uint8)

        for z in range(img_3D.shape[0]):
            img_2D = img_3D[z, :, :]
            img_3c = np.repeat(img_2D[:, :, None], 3, axis=-1)
            img_3c = _transfer(img_3c.copy())[None, ...]
            # img_3c = img_3c.view(1, 3, img_2D.shape[0], img_2D.shape[1])
            pred_2D = s_model_inference(model=model, image=img_3c, device=device)
            pred_3D[z, :, :] = pred_2D
        
        pred_3D = remove_all_small_regions(pred_3D, min_size=200)
        iou_scores.append(compute_iou_scores(pred_3D, gt_3D))
        dice_scores.append(compute_dice_scores(pred_3D, gt_3D))
        # asd_results.append(compute_asd_all_classes(pred_3D, gt_3D, spacing=spacing_xyz))
        # hd95_results.append(compute_hd95_all_classes(pred_3D, gt_3D, spacing=spacing_xyz))

        if args.visualize:
            seg_sitk = sitk.GetImageFromArray(pred_3D)
            seg_sitk.SetSpacing(spacing)
            sitk.WriteImage(seg_sitk, join(args.pred_save_dir, name.replace('.npz', f'_pred.nii.gz')))
            print(f'{name} done')

    iou_scores = np.array(iou_scores).flatten()
    dice_scores = np.array(dice_scores).flatten()
    # asd_results = np.array(asd_results).flatten()
    # hd95_results = np.array(hd95_results).flatten()
    print(args.model)
    print(args.checkpoint_path)
    print(f'average dice = {np.mean(dice_scores):.5f}')
    print(f'std dice = {np.std(dice_scores):.5f}')
    print(f'average iou = {np.mean(iou_scores):.5f}')
    print(f'std iou = {np.std(iou_scores):.5f}')
    # print(f'average asd = {np.mean(asd_results):.5f}')
    # print(f'std asd = {np.std(asd_results):.5f}')
    # print(f'average hd95 = {np.mean(hd95_results):.5f}')
    # print(f'std hd95 = {np.std(hd95_results):.5f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-device",
        type=str,
        default="cuda:0",
        help="the device to run the code on",
    )
    parser.add_argument(
        "-checkpoint_path",
        type=str,
        default="work_dir/UNet/_latest.pth",
        help="path to the config file for the model",
    ) 
    parser.add_argument(
        "-data_root",
        type=str,
        default="data/npz_test/MR_Abd",
        help="path to the data root",
    )
    parser.add_argument(
        "-batch_size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument("-pred_save_dir", type=str, default="work_dir", help="Path to save the segmentation results")
    parser.add_argument("--visualize", action="store_true", default=False, help="Save the .nii.gz segmentation results")
    parser.add_argument("--model", type=str, default="UNet", help="the model need to test")
    agrs = parser.parse_args()
    main(agrs)