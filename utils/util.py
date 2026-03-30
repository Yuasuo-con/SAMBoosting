import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
import scipy.ndimage

def remove_small_components(mask, min_size=100, connectivity=1):
    # mask: binary mask (3D)
    # min_size: minimum number of voxels to keep
    labeled, num = scipy.ndimage.label(mask, structure=scipy.ndimage.generate_binary_structure(3, connectivity))
    sizes = np.bincount(labeled.ravel())
    # Create mask of components to keep
    keep_mask = np.zeros_like(mask, dtype=bool)
    for i in range(1, len(sizes)):
        if sizes[i] >= min_size:
            keep_mask[labeled == i] = True
    return keep_mask.astype(np.uint8)

def remove_all_small_regions(
    pred_mask: np.ndarray, min_size: int = 100
) -> np.ndarray:
    
    num_classes = pred_mask.max() + 1
    cleaned_pred = pred_mask.copy()

    for cls in range(1, num_classes): 
        cleaned_pred[(pred_mask == cls)] = 0  
        cleaned_cls = remove_small_components(pred_mask == cls, min_size=min_size)
        cleaned_pred[cleaned_cls == 1] = cls
    return cleaned_pred

def compute_iou_scores(mask_pred, mask_gt):
    assert mask_pred.shape == mask_gt.shape, "Shape mismatch"
    classes = np.unique(mask_gt)[1:]
    iou_scores = []
    for cls in classes:
        mask_pred_cls = (mask_pred == cls)
        mask_gt_cls = (mask_gt == cls)
        intersection = np.logical_and(mask_pred_cls, mask_gt_cls).sum()
        union = np.logical_or(mask_pred_cls, mask_gt_cls).sum()
        iou_score = intersection / union
        iou_scores.append(iou_score)
    return iou_scores

def compute_dice_scores(mask_pred, mask_gt):
    assert  mask_pred.shape == mask_gt.shape, "mask_pred and mask_gt must have the same shape"
    classes = np.unique(mask_gt)[1:]
    dice_scores = []
    for cls in classes:
        mask_pred_cls = (mask_pred == cls)
        mask_gt_cls = (mask_gt == cls)
        intersection = np.logical_and(mask_pred_cls, mask_gt_cls).sum()
        union = mask_pred_cls.sum() + mask_gt_cls.sum()
        dice_score = 2 * intersection / union
        dice_scores.append(dice_score)
    return dice_scores

def get_surface_voxels(mask):
    # mask: binary 3D array
    eroded = binary_erosion(mask)
    surface = mask ^ eroded
    return np.stack(np.nonzero(surface), axis=-1) 

def compute_asd_for_class(pred, gt, spacing=(1.0, 1.0, 1.0)):

    pred_surface = get_surface_voxels(pred)
    gt_surface = get_surface_voxels(gt)
    
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return np.nan 

    pred_surface = pred_surface * spacing
    gt_surface = gt_surface * spacing


    tree_pred = cKDTree(pred_surface)
    tree_gt = cKDTree(gt_surface)

    d1, _ = tree_gt.query(pred_surface)
    d2, _ = tree_pred.query(gt_surface)

    asd = (d1.sum() + d2.sum()) / (len(d1) + len(d2))
    return asd

def compute_asd_all_classes(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0), num_classes=4):
    results = []
    for c in range(1, num_classes):  
        pred_c = (pred_mask == c)
        gt_c = (gt_mask == c)
        asd = compute_asd_for_class(pred_c, gt_c, spacing)
        results.append(asd)
    return results

def compute_hd95(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)):
    pred_surface = get_surface_voxels(pred_mask)
    gt_surface = get_surface_voxels(gt_mask)

    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return np.nan

    pred_surface = pred_surface * spacing
    gt_surface = gt_surface * spacing

    tree_pred = cKDTree(pred_surface)
    tree_gt = cKDTree(gt_surface)

    d_pred_to_gt, _ = tree_gt.query(pred_surface)
    d_gt_to_pred, _ = tree_pred.query(gt_surface)

    hd95 = max(
        np.percentile(d_pred_to_gt, 95),
        np.percentile(d_gt_to_pred, 95)
    )
    return hd95



def compute_hd95_all_classes(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0), num_classes=4):
    results = []

    for c in range(1, num_classes):
        pred_c = (pred_mask == c)
        gt_c = (gt_mask == c)
        hd95 = compute_hd95(pred_c, gt_c, spacing)
        results.append(hd95)
        # print(f"Class {c}: HD95 = {hd95:.2f}")
    return results