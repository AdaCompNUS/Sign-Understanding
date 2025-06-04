import numpy as np
from scipy.ndimage import label, center_of_mass
import cv2, os, pdb
from utils.mobilesam import GroundedSAM
from utils import file_utils

def crop_buffer_bbox(img_path, bbox_cords, buffer = 10):
    '''
    crops bbox and a buffer area around the image -- this would be used to be fed in again to Grounded-DINO and SAM
    '''
    if isinstance(img_path, str):
        img = cv2.imread(f'{img_path}')
    elif isinstance(img_path, np.ndarray):
        img = img_path
    x_min , y_min, x_max , y_max = bbox_cords
    return img[ max(0,int(y_min) - buffer): min(img.shape[0], int(y_max) + buffer+1), max(0,int(x_min) - buffer) : min(img.shape[1],int(x_max) + buffer+1)]
     
def get_image_crops(img_path, crop_model):
    crop_model.execute_model(img_path, type='box')
    return_bbox_list = crop_model.detections.xyxy
    conf_list = crop_model.detections.confidence
    crop_img_list = []
    for det in crop_model.detections.xyxy:
        crop_img = crop_buffer_bbox(img_path, det)
        crop_img_list.append(crop_img)
    return crop_img_list , return_bbox_list, conf_list

def compute_iou(boxA, boxB):
    # box format: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return interArea / float(boxAArea + boxBArea - interArea)

def compute_iou_matrix(preds, gts):
    iou_matrix = np.zeros((len(preds), len(gts)))
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            iou_matrix[i, j] = compute_iou(pred, gt)
    return iou_matrix

def greedy_match(preds, gts, iou_threshold=0.75):
    iou_matrix = compute_iou_matrix(preds, gts)
    matched_pred_indices = set()
    matched_gt_indices = set()
    matches = []

    while True:
        max_iou = -1
        max_pair = None
        for i in range(len(preds)):
            if i in matched_pred_indices:
                continue
            for j in range(len(gts)):
                if j in matched_gt_indices:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_pair = (i, j)

        if max_iou < iou_threshold or max_pair is None:
            break

        i, j = max_pair
        matches.append((i, j, max_iou))
        matched_pred_indices.add(i)
        matched_gt_indices.add(j)

    return matches