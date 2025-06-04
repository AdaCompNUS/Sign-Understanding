import os,pdb
import datetime
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils import file_utils

def convert_bbox_xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox[:4]
    return [int((x1 + x2)/2), int((y1+y2)/2), int(x2 - x1), int(y2 - y1)]

def build_coco_format(gt_dict, pred_dict):
    annotations = []
    images = []
    predictions = []
    annotation_id = 1

    for image_id, gt_boxes in gt_dict.items():
        images.append({
            "id": image_id,
            "width": 1920,  
            "height": 1080
        })
        for box in gt_boxes:
            x, y, w, h = box
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1
        pred= pred_dict.get(image_id, [])[0]
        sc = pred_dict.get(image_id, [])[1]
        for idx in range(len(pred)): 
            x, y, w, h = convert_bbox_xyxy_to_xywh(pred[idx])
            predictions.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "score": sc[idx]
            })

    coco_gt = {
        "info": {"description": "Eval", "date_created": str(datetime.datetime.now())},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "navigational sign board with arrows"}]
    }

    return coco_gt, predictions

def evaluate_coco(gt_dict, pred_dict, iou_thresholds=[0.5, 0.75]):
    gt_json, pred_json = build_coco_format(gt_dict, pred_dict) # now both gt and pred in xywh format
    res_dir = '/home/ayush/arxiv'
    ann_path = os.path.join(res_dir, "gt.json")
    pred_path = os.path.join(res_dir, "pred.json")

    file_utils.save_file_json(ann_path, gt_json)
    file_utils.save_file_json(pred_path, pred_json)
    
    # Define size categories
    size_categories = {
        'small': (0, 32**2),
        'medium': (32**2, 96**2),
        'large': (96**2, float('inf'))
    }

    coco_gt = COCO(ann_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.scoreThrs = np.linspace(0.0, 1.0, 101)
    coco_eval.params.iouThrs = np.array(iou_thresholds)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()