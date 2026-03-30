import torch
import numpy as np
import os
import torch.distributed
from tqdm import tqdm

join = os.path.join
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import monai
import argparse
import random
from datetime import datetime
import glob
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist 
from utils.util import compute_iou_scores
import logging
import PIL.Image as Image
import cv2

# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20, labeled_data=None):
        self.data_root = data_root
        if labeled_data is not None:
            if labeled_data == 1:
                self.labeled_data = 274
            elif labeled_data == 2:
                self.labeled_data = 509
            elif labeled_data == 4:
                self.labeled_data = 1027
            elif labeled_data == 8:
                self.labeled_data = 2097
            else:
                self.labeled_data = labeled_data
        else:
            self.labeled_data = None
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "*.npy"), recursive=True)
        )

        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        
        # self.gt_path_files = self.gt_path_files[:int(len(self.gt_path_files) * self.labeled_ratio)]
        if self.labeled_data is not None:
            self.gt_path_files = self.gt_path_files[:self.labeled_data]
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
            torch.tensor(gt2D[None, :, :]).long(), ## [1, 256, 256]
            torch.tensor(bboxes).float(), 
            img_name,
        )

class ISICDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20, labeled_data=None):
        self.data_root = data_root
        self.labeled_data = labeled_data
        self.gt_path = join(data_root, "label")
        self.img_path = join(data_root, "image")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "*.png"), recursive=True)
        )
        
        if self.labeled_data is not None:
            self.gt_path_files = self.gt_path_files[:self.labeled_data]
        self.bbox_shift = bbox_shift
        self._transform = SAM2Transforms(resolution=1024, mask_threshold=0)
        print(f"number of images: {len(self.gt_path_files)}")
    
    def __len__(self):
        return len(self.gt_path_files)
    
    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index]).replace("_segmentation.png", ".jpg")
        img = Image.open(join(self.img_path, img_name))
        # convert the shape to (3, H, W)
        img_1024 = self._transform(img.copy())
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
            torch.tensor(gt2D[None, :, :]).long(), ## [1, 256, 256]
            torch.tensor(bboxes).float(), 
            img_name,
        )


class MedSAM2(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam2_model = model
        # freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
        # freeze image encoder
        for param in self.sam2_model.image_encoder.parameters():
            param.requires_grad = False
        

    def forward(self, image, box):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits
    
    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features

class Trainer:
    def __init__(
        self,
        model:torch.nn.Module,
        train_loader:DataLoader,
        optimizer:torch.optim,
        val_loader:DataLoader=None,
        scheduler=None,
        device=None,
        num_epochs=None,
        save_path=None,
        save_name=None,
        resume=None,
        log_interval=10,
        val_interval=2,
        best_val_loss=1,
        best_val_dice=0,
        best_val_iou=0,
        lo=nn.CrossEntropyLoss(ignore_index=255),
    ):
        super().__init__()
        self.model = DDP(model,device_ids=[device],output_device=device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.save_name = save_name
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.best_val_loss = best_val_loss
        self.best_val_dice = best_val_dice
        self.best_val_iou = best_val_iou
        self.lo = lo
        self.resume = resume

    def _run_batch(self, image, gt2D, boxes, mode="train"):
        if mode == "train":
            self.optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            medsam_pred = self.model(image, boxes_np)
            loss = self.lo(medsam_pred, gt2D)       
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        elif mode == "val":
            boxes_np = boxes.detach().cpu().numpy()
            medsam_pred = self.model(image, boxes_np)
            loss = self.lo(medsam_pred, gt2D)
        else:
            return None    
        return loss.item()
    
    def _iou(self, mask_pred, mask_gt):
        mask_pred = mask_pred.detach().cpu().numpy()
        mask_gt = mask_gt.detach().cpu().numpy()
        mask_pred = np.argmax(mask_pred, axis=1, keepdims=True)
        return compute_iou_scores(mask_pred, mask_gt)
    def _run_epoch(self, epoch):
        """
        Args:
            epoch: current epoch number
            mode: "train" or "val"
        """
        self.model.train()
        loader = tqdm(self.train_loader,desc=f"train epoch {epoch}")
        # set the sampler to the current epoch, so that the sampler is consistent across all processes
        self.train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(loader):
            image = image.to(self.device)
            gt2D = gt2D.to(self.device)
            boxes = boxes.to(self.device)
            loss = self._run_batch(image, gt2D, boxes, mode="train")
            epoch_loss += loss
        
        epoch_loss /= step
        reduced_loss = torch.tensor([epoch_loss],device=self.device)
        dist.all_reduce(reduced_loss)
        reduced_loss /= dist.get_world_size()
        # only the master process prints the loss and saves the checkpoint
        if dist.get_rank() == 0:
            print(
                f'Epoch: {epoch}, Loss: {reduced_loss.item()}'
            )
            self.save_checkpoint(epoch)
        return reduced_loss.item()
    def val(self, epoch):
        val_loss = 0
        self.model.eval()
        loader = tqdm(self.val_loader,desc=f"val epoch {epoch}")
        # set the sampler to the current epoch, so that the sampler is consistent across all processes
        self.val_loader.sampler.set_epoch(epoch)
        for step, (image, gt2D, boxes, _) in enumerate(loader):
            image = image.to(self.device)
            gt2D = gt2D.to(self.device)
            boxes_np = boxes.to(self.device)
            val_loss += self._run_batch(image, gt2D, boxes_np, mode="val")
        
        val_loss /= step
        # reduce the loss across all processes
        val_reduced_loss = torch.tensor([val_loss],device=self.device)
        dist.all_reduce(val_reduced_loss)
        val_reduced_loss /= dist.get_world_size()
        # only the master process prints the loss and saves the checkpoint
        if dist.get_rank() == 0:
            print(
                f'Epoch: {epoch}, Val Loss: {val_reduced_loss.item()}'
            )
            if val_reduced_loss < self.best_val_loss:
                self.best_val_loss = val_reduced_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"Best val loss: {val_reduced_loss.item()}, best val loss epoch: {epoch}")
        return val_reduced_loss.item()
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
        }
        if is_best:
            torch.save(checkpoint, join(self.save_path, "medsam_model_best.pth"))
        else:
            torch.save(checkpoint, join(self.save_path, "medsam_model_latest.pth"))
    def train(self, max_epochs):
        start_epoch = 0
        epoch_loss_list = []
        val_loss_list =  []
        if self.resume is not None:
            if os.path.isfile(self.resume):
                ## Map model to be loaded to specified single GPU
                print("=> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume, map_location=f'cuda:{self.device}')
                # checkpoint["model"] = {'module.'+k : v for k, v in checkpoint["model"].items()}
                start_epoch = checkpoint["epoch"] + 1
                self.model.load_state_dict(checkpoint["model"], strict=True)
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.best_val_loss = checkpoint["best_val_loss"]
        for epoch in range(start_epoch, max_epochs):
            epoch_loss_list.append(self._run_epoch(epoch))
            if (epoch % self.val_interval == 0) and (epoch > 0):
                val_loss_list.append(self.val(epoch))
            
            # plot loss
            if dist.get_rank() == 0:
                plt.plot(epoch_loss_list)
                plt.plot(val_loss_list)
                plt.title("Dice + Cross Entropy Loss: train + val loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend(["train_loss", "val_loss"])
                plt.savefig(join(self.save_path, "loss.png"))
                plt.close()
    
class DiceCELoss(nn.Module):
    def __init__(self, baison=0.5):
        super(DiceCELoss, self).__init__()
        self.baison = baison
        self.dice_loss = monai.losses.DiceLoss(softmax=True, squared_pred=True, reduction="mean", to_onehot_y=True)
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
    def forward(self, inputs, targets):
        return self.dice_loss(inputs, targets) + self.ce_loss(inputs, targets.view(-1,256,256))


def ddp_setup(rank, world_size):
    set_seed(2024)
    # initialize the process group, rank and world size
    init_process_group(backend="nccl", world_size=world_size, rank=rank, init_method="env://")
    # set the default device
    torch.cuda.set_device(rank)

def main(rank, args):
    ddp_setup(rank=rank, world_size=args.world_size)
    local_rank = torch.distributed.get_rank()
    if args.dataset == "spine":
        tr_dataset = NpyDataset(args.tr_npy_path, bbox_shift=args.bbox_shift, labeled_data=args.labeled_data)
        val_dataset = NpyDataset(args.val_npy_path,bbox_shift=args.bbox_shift)
    elif args.dataset == "skin":
        tr_dataset = ISICDataset(args.tr_npy_path, bbox_shift=args.bbox_shift, labeled_data=args.labeled_data)
        val_dataset = ISICDataset(args.val_npy_path,bbox_shift=args.bbox_shift)
    else:
        raise NotImplementedError
    # set the dataloader sampler, so that the sampler is consistent across all processes
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(tr_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(val_dataset))
    loss_fn = DiceCELoss()
    model_cfg = args.model_cfg
    sam2_checkpoint = args.pretrain_model_path
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=local_rank, apply_postprocessing=True)
    medsam_model = MedSAM2(model=sam2_model).to(local_rank)
    img_mask_encdec_params = list(medsam_model.sam2_model.image_encoder.parameters()) + list(
        medsam_model.sam2_model.sam_mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
    img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    # only rank 0 will print the number of parameters
    if local_rank == 0:
        print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
        ) 
    
    trainer = Trainer(
        model=medsam_model,
        train_loader=tr_dataloader,
        val_loader=val_dataloader,
        val_interval=10000,
        optimizer=optimizer,
        device=local_rank,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        save_name=args.task_name,
        lo=loss_fn,
        resume=args.resume,
    )

    trainer.train(args.num_epochs)
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default=None,
    help="path to training npy files; two subfolders: gts and imgs",
    )
    parser.add_argument("-task_name", type=str, default="MedSAM2-Fine-Tuning")
    parser.add_argument(
        "-model_cfg", type=str, default="sam2_hiera_l.yaml", help="model config file"
    )
    parser.add_argument("-pretrain_model_path",
                        type=str,
                        default=None,
    )
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    # train
    parser.add_argument("-num_epochs", type=int, default=30)
    parser.add_argument("-batch_size", type=int, default=8)
    parser.add_argument("-bbox_shift", type=int, default=5)
    parser.add_argument("-num_workers", type=int, default=0)
    # data
    parser.add_argument("-labeled_data", type=int, default=200, help="how many labeled data to use, spine: 1, 2, 4, 8 skin:20, 50, 100, 200")
    parser.add_argument("-dataset", choices=["spine", "skin"], default="spine", help="dataset to use")
    # Optimizer parameters
    parser.add_argument(
        "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
    )
    parser.add_argument(
        "-lr", type=float,
        default=6e-5,
        metavar="LR", help="learning rate (absolute lr)"
    )
    parser.add_argument(
        "-resume", type=str,
        default=None,
        help="Resuming training from checkpoint"
    )
    parser.add_argument(
        "--world-size", default=2, type=int, help="number of cuda for distributed training"
    )
    # 
    parser.add_argument("-val_npy_path", type=str, help="path to validation data")
    args, unknown = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "6"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "6"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    os.makedirs(model_save_path, exist_ok=True)
    args.save_path = model_save_path

    torch.multiprocessing.spawn(main, args=(args,), nprocs=2)
    
            