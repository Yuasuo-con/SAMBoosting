from ast import arg, mod
from genericpath import exists
import os
import glob
from sympy import im
import torch
import torch.multiprocessing as mp
import numpy as np
import argparse
from os.path import join
from torch.distributed import init_process_group,destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DataParallel as DDP
from torch.utils.data import DataLoader
from dataloader.dataset import NpyDataset, ISICDataset
from dataloader.sampler import TwoStreamBatchSampler, TwoStreamBatchSampler_distributed
from sam2.build_sam import build_sam2
from UNet.UNet import UNet
from transUnet.vit_seg_modeling import VisionTransformer as transUNet
from transUnet.vit_seg_configs import get_r50_b16_config
from SwinUNet.vision_transformer import SwinUnet as SwinUNet
from torch.nn import functional as F
from monai.losses import DiceLoss
import logging
import datetime

import utils.losses as losses
from utils import ramps
from tensorboardX import SummaryWriter
from medsam2.medsam2 import MedSAM2

# set seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# set the process group
def ddp_setup(rank, world_size):
    init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
    set_seed(2024)
    torch.cuda.set_device(rank)

def get_current_consistency_weight(consistency, epoch, consistency_rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model:torch.nn.Module, ema_model:torch.nn.Module, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # if global_step < 3000:
    #     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #         ema_param = param.clone()
    # else:
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(alpha).add_(param, alpha=1 - alpha)
def set_logger_queue(log_queue):
    """
        set the mutiprocessing queue for logger
    """
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(queue_handler)
    return logger

def main(rank, args, log_queue):
    logger = set_logger_queue(log_queue)
    lr_= args.lr
    labeled_ratio = args.labeled_ratio
    alpha = args.alpha

    ddp_setup(rank, args.world_size)
    local_rank = dist.get_rank()

    # init  model and ema model
    if args.model == 'UNet':
        model = UNet(3, 4)
        ema_model = UNet(3, 4)
    elif args.model == 'transUNet':
        config_vit = get_r50_b16_config()
        config_vit.n_classes = 4
        config_vit.n_skip = 3
        config_vit.patches.grid = (16, 16)
        model = transUNet(config_vit, img_size=256, num_classes=4)

        # load VIT pretrained weights
        model.load_from(weights=np.load(config_vit.pretrained_path))

        # freeze VIT weights
        # for params in model.transformer.parameters():
        #     params.requires_grad = False

        ema_model = transUNet(config_vit, img_size=256, num_classes=4)
    elif args.model == 'SwinUNet':
        config = None
        model = SwinUNet(config, img_size=256, num_classes=2)
        ema_model = SwinUNet(config, img_size=256, num_classes=2)
    
    else:
        raise NotImplementedError

    for param in ema_model.parameters():
        param.detach_()
    
    # move models to gpu

    model = DDP(model, device_ids=[local_rank])
    ema_model = DDP(ema_model, device_ids=[local_rank])
    model.to(local_rank)
    ema_model.to(local_rank)

    if os.path.exists(os.path.join(args.work_dir, '_latest_' + str(alpha) + '.pth')):
        model.load_state_dict(torch.load(os.path.join(args.work_dir, '_latest_' + str(alpha) + '.pth'), weights_only=True))
        print("model loaded")

    dice_loss = DiceLoss(to_onehot_y=True, softmax=True, reduction="mean")

    files = glob.glob(os.path.join(args.tr_npy_path+'/img', '*'))
    files_count = len(files)
    
    args.tr_npy_path
    if args.dataset == "spine":
        if args.labelled_num is not None:
            if args.labelled_num == 1:
                args.labelled_num = 274
            elif args.labelled_num == 2:
                args.labelled_num = 509
            elif args.labelled_num == 4:
                args.labelled_num = 1027
            elif args.labelled_num == 8:
                args.labelled_num = 2097
    labeled_idxs = list(range(args.labelled_num))
    unlabeled_idxs = list(range(args.labelled_num, files_count))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # print model parameter numbers
    logger.info("the total parameter numbers", sum(p.numel() for p in model.parameters() if p.requires_grad))    

    labeled_bs = args.labeled_bs
    batch_sampler = TwoStreamBatchSampler_distributed(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-labeled_bs)
    if args.dataset == "spine":
        train_dataset = NpyDataset(args.tr_npy_path, bbox_shift=args.bbox_shift)
        train_loader = DataLoader(train_dataset,
                                batch_sampler=batch_sampler,
                                num_workers=args.num_workers, 
                                pin_memory=True)
    elif args.dataset == "skin":
        train_dataset = ISICDataset(args.tr_npy_path, bbox_shift=args.bbox_shift)
        train_loader = DataLoader(train_dataset,
                                batch_sampler=batch_sampler,
                                num_workers=args.num_workers, 
                                pin_memory=True)
    
    if local_rank == 0:
        writer = SummaryWriter(args.work_dir+'/log')

    model.train()
    ema_model.train()
    if os.path.exists(os.path.join(args.work_dir, '_latest_' + str(alpha) + '.pth')):
        with open(os.path.join(args.work_dir, 'log.txt')) as log:
            line = log.readlines()[-1].strip()
            num = line.split(' ')[6].strip()
            iter_num = (int(num) // 1000) * 1000
    else:
        iter_num = 0
    max_epoch = args.max_iterations//len(train_loader)+1
    for epoch in range(max_epoch):
        for inputs_1024, inputs_256, targets, bbox, _ in train_loader:
            inputs_1024, inputs_256, targets = inputs_1024.to(local_rank), inputs_256.to(local_rank),targets.to(local_rank)
            
            outputs = model(inputs_256)
            # outputs_soft = F.softmax(outputs, dim=1)
            
            noise = torch.clamp(torch.rand_like(inputs_256) * 0.1, -0.2, 0.2)
            ema_inputs = inputs_256 + noise

            
            outputs = model(inputs_256)
            
            with torch.no_grad():
                ema_outputs = ema_model(ema_inputs)
            
            loss_seg = F.cross_entropy(outputs[:labeled_bs], targets[:labeled_bs])
            loss_seg_dice = dice_loss(outputs[:labeled_bs], targets[:labeled_bs].view(-1,1,256,256))

            consistency_dist = losses.softmax_mse_loss(outputs, ema_outputs)
            consistency_dist = torch.sum(consistency_dist)/(2*consistency_dist.numel())
            consistency_weight = get_current_consistency_weight(args.consistency, iter_num//500, args.consistency_rampup)
            consistency_loss = consistency_dist * consistency_weight

            loss = (loss_seg + loss_seg_dice) + consistency_loss
            loss = loss_seg + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, 0.99, iter_num)
            iter_num = iter_num + 1
            if local_rank == 0:
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
                writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
                writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
                logger.info('iteration %d : loss : %f loss_seg: %f loss_seg_dice: %f  consistency_loss:%f consistency_weight:%f' % 
                         (iter_num, loss.item(), loss_seg, loss_seg_dice, consistency_loss, consistency_weight))
                # logger.info('iteration %d : loss : %f loss_seg: %f consistency_loss:%f consistency_weight:%f' % 
                #          (iter_num, loss.item(), loss_seg, consistency_loss, consistency_weight))
                
            ## change lr
            # if iter_num % 2500 == 0:
            #     lr_ = args.lr * 0.1 ** (iter_num // 2500)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_
            # if iter_num % 1000 == 0 and local_rank == 0:
            #     save_mode_path = os.path.join(args.work_dir, 'iter_' + str(iter_num) + '_' + str(alpha) +'.pth')
            #     torch.save(model.state_dict(), save_mode_path)
            #     print("save model to {}".format(save_mode_path))

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break
        if local_rank == 0 and iter_num % 1000 == 0:

            # save model parameters
            save_mode_path = os.path.join(args.work_dir, '_latest_' + str(alpha) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logger.info("save model to {}".format(save_mode_path))

             # save ema model parameters
            save_mode_path = os.path.join(args.work_dir, '_latest'+ '_ema_' + str(alpha)+ '.pth')
            torch.save(ema_model.state_dict(), save_mode_path)
            logger.info("save ema model to {}".format(save_mode_path))
    if local_rank == 0:

        # save model parameters
        save_mode_path = os.path.join(args.work_dir, '_latest_' + str(alpha) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logger.info("save model to {}".format(save_mode_path))

        # save ema model parameters
        save_mode_path = os.path.join(args.work_dir, '_latest'+ '_ema_' + str(alpha)+ '.pth')
        torch.save(ema_model.state_dict(), save_mode_path)
        logger.info("save ema model to {}".format(save_mode_path))

        # close writer
        writer.close()
            

    destroy_process_group()

def log_listener(log_queue, log_path):
    """
    main log listener, used to listen to the log queue and print the logs
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    while True:
        # Get the next record from the queue, use .get() instead of .get_nowait()
        record = log_queue.get()
        if record is None:  
            break
        logger.handle(record)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-task_name", type=str, default="1case_labeled_large_trian")
    parser.add_argument(
        "-model_cfg", type=str, default="sam2_hiera_l.yaml", help="model config file"
    )
    parser.add_argument("-pretrain_model_path",
                        type=str,
                        default="checkpoints/sam2_hiera_large.pt",
    )
    parser.add_argument("-work_dir", type=str, default="./work_dir_without_sam2")
    # train
    parser.add_argument("-batch_size", type=int, default=24)
    parser.add_argument("-bbox_shift", type=int, default=5)
    parser.add_argument("-num_workers", type=int, default=0)
    # Optimizer parameters
    parser.add_argument(
        "-weight_decay", type=float, default=0.001, help="weight decay (default: 0.001)"
    )
    parser.add_argument(
        "-lr", type=float,
        default=1e-3,
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
    parser.add_argument("-val_npy_path", type=str, help="path to validation data")
    parser.add_argument(
        "-max_iterations",
        type=int,
        default=20000,
        help="maximum epoch number to train",
    )
    parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
    parser.add_argument(
        "-tr_npy_path",
        type=str,
        default="./ISIC-2017/train",
        help="path to training npy files; two subfolders: gts and imgs",
    )
    parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
    parser.add_argument('--labeled_ratio', type=float, default=0.1, help='ratio of labeled data')
    parser.add_argument('--alpha', type=float, default=0.01, help='alpha')
    parser.add_argument('--model', type=str, default='UNet', help="the model need to train")
    
    parser.add_argument('-dataset', type=str, choices=['spine', 'skin'], default='spine')
    parser.add_argument('-labelled_num', type=int, default=8, help='number of labelled data')

    args, unknown = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "6"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "6"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    args.work_dir = os.path.join(args.work_dir, args.model, args.task_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.work_dir, exist_ok=True)
    # set the logger
    mp.set_start_method("spawn")
    log_queue = mp.Queue()

    listener = mp.Process(target=log_listener, args=(log_queue, os.path.join(args.work_dir, "log.txt")))
    listener.start()

    # start the sub process
    mp.spawn(main, args=(args, log_queue), nprocs=2)

    # stop the listener
    log_queue.put(None)
    listener.join()
