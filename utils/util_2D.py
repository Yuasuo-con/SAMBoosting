import numpy as np
from scipy.spatial.distance import cdist
from skimage.segmentation import find_boundaries

def compute_dice_scores(mask_pred, mask_gt):
    intersection = np.sum(mask_pred * mask_gt)
    return 2 * intersection / (np.sum(mask_pred) + np.sum(mask_gt))

def compute_iou_scores(mask_pred, mask_gt):
    intersection = np.sum(mask_pred * mask_gt)
    union = np.sum(mask_pred) + np.sum(mask_gt) - intersection
    return intersection / union if union != 0 else 0

def average_symmetric_surface_distance(mask_pred, mask_gt):
    # 提取边界点
    pred_boundary = np.argwhere(find_boundaries(mask_pred, mode="inner"))
    gt_boundary   = np.argwhere(find_boundaries(mask_gt, mode="inner"))

    # 如果两者都没有前景
    if len(pred_boundary) == 0 and len(gt_boundary) == 0:
        return 0.0
    # 如果一边为空，另一边有边界
    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return 0.0

    # 计算距离
    dist_pred_to_gt = cdist(pred_boundary, gt_boundary)
    min_dist_pred_to_gt = np.min(dist_pred_to_gt, axis=1)

    dist_gt_to_pred = cdist(gt_boundary, pred_boundary)
    min_dist_gt_to_pred = np.min(dist_gt_to_pred, axis=1)

    # 平均对称表面距离
    asd_pred_to_gt = np.mean(min_dist_pred_to_gt)
    asd_gt_to_pred = np.mean(min_dist_gt_to_pred)

    return (asd_pred_to_gt + asd_gt_to_pred) / 2

def hausdorff_distance(mask_pred, mask_gt, percentile=95):
    # 提取边界点
    pred_boundary = np.argwhere(find_boundaries(mask_pred, mode="inner"))
    gt_boundary   = np.argwhere(find_boundaries(mask_gt, mode="inner"))

    # 都没有前景
    if len(pred_boundary) == 0 and len(gt_boundary) == 0:
        return 0.0
    # 一边没有
    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return 0.0

    # 最近距离集合
    dist_pred_to_gt = cdist(pred_boundary, gt_boundary)
    min_dist_pred_to_gt = np.min(dist_pred_to_gt, axis=1)

    dist_gt_to_pred = cdist(gt_boundary, pred_boundary)
    min_dist_gt_to_pred = np.min(dist_gt_to_pred, axis=1)

    # 合并后取 95th percentile
    all_distances = np.concatenate([min_dist_pred_to_gt, min_dist_gt_to_pred])
    return np.percentile(all_distances, percentile)

if __name__ == "__main__":
    # 示例数据
    mask_pred = np.array([[0, 0, 0, 1, 1, 1],])
    # 示例：使用模型输出和真实标签
    mask_pred = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])  # 假设的模型预测掩码
    mask_gt = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])     # 假设的地面真实掩码

    # 计算Dice, IoU, ASD, 和 HD95
    dice = compute_dice_scores(mask_pred, mask_gt)
    iou = compute_iou_scores(mask_pred, mask_gt)
    asd = average_symmetric_surface_distance(mask_pred, mask_gt)
    hd95 = hausdorff_distance(mask_pred, mask_gt)

    print(f"Dice: {dice}")
    print(f"IoU: {iou}")
    print(f"ASD: {asd}")
    print(f"HD95: {hd95}")
