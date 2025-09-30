import numpy as np
from scipy.ndimage import label, center_of_mass
import cv2

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


def convert_to_binary(img, bbox=None, mode = "bbox"):
    if mode == "bbox":
        assert len(bbox), 'provide bbox if you choose bbox type for binary conversion' 
        img_bin = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        x_min , y_min, x_max , y_max = bbox
        img_bin[int(y_min):int(y_max) + 1 , int(x_min):int(x_max)+1] = 255
        return img_bin
    
    elif mode == 'irregular':
        raise NotImplementedError
        
def calculate_orientation(binary_image):
    """
    Calculate the orientation vector of a 2D shape in a binary image.
    Returns angle in radians and the unit vector of orientation.
    """
    # Calculate moments
    y_coords, x_coords = np.nonzero(binary_image)
    x_bar, y_bar  = np.mean(x_coords), np.mean(y_coords)
    
    # Calculate central moments
    u20 = np.sum((x_coords - x_bar) ** 2)
    u02 = np.sum((y_coords - y_bar) ** 2)
    u11 = np.sum((x_coords - x_bar) * (y_coords - y_bar))
    
    # Calculate orientation angle
    theta = 0.5 * np.arctan2(2 * u11, u20 - u02)
    
    # Calculate unit vector
    direction_vector = np.array([np.cos(theta), np.sin(theta)])
    
    return theta, direction_vector

def get_shape_properties(binary_image):
    """
    Get basic properties of the shape including centroid and orientation.
    """
    # Find centroid
    labeled_array, num_features = label(binary_image)
    cy , cx  = center_of_mass(binary_image)
    
    # Get orientation
    theta, direction = calculate_orientation(binary_image)
    
    return {
        'centroid': (cx, cy),
        'angle_rad': theta,
        'angle_deg': np.degrees(theta),
        'direction_vector': direction
    }

def rotate_sign_to_align_bbox(crop_img, bbox_cords1 , irregular_binary_mask, ablation = False):
    x_min , y_min, x_max , y_max = bbox_cords1 # regular shape
    bbox_binary = convert_to_binary(crop_img, bbox_cords1)
    center = get_shape_properties(irregular_binary_mask)['centroid'] # center of irregular
    angle_bbox = np.degrees(calculate_orientation(bbox_binary)[0])
    angle_irr = np.degrees(calculate_orientation(irregular_binary_mask)[0])
    # print(f"BBOX ANGLE: {angle_bbox}")
    # print(f"IRR ANAGLE: {angle_irr}")
    if angle_irr > 0 : 
        angle = min(10,angle_irr)
    else:
        angle = max(-10,angle_irr)
    
    scale = 1
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(crop_img, rotation_matrix, (crop_img.shape[1], crop_img.shape[0]))
    if ablation:
        return angle_irr, crop_img
    return angle_irr, rotated_image

def get_rotated_image_crops(img_path, crop_model, ocr_map_queue=None, ablation=True):
    #ablation True means we are not canonicalizing the crop
    ctd, area = crop_model.execute_model(img_path, ocr_map_queue,  type='box')
    crop_img = crop_buffer_bbox(img_path, crop_model.detections.xyxy[0])
    # if not ablation:
    #     crop_model.execute_model(crop_img,ocr_map_queue, type='mask')
    irregular_binary_mask = crop_model.detections.mask[0].astype(np.uint8)*255
    ang , rot_img = rotate_sign_to_align_bbox(crop_img, crop_model.detections.xyxy[0], irregular_binary_mask, ablation)
    return ctd, area, ang, rot_img