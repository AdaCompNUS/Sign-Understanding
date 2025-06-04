import torch, os
import numpy as np 
import cv2, pdb
import supervision as sv
import torchvision
from segment_anything import SamPredictor
from MobileSAM.setup_mobile_sam import setup_model
from groundingdino.util.inference import Model

class GroundedSAM:

    def __init__(self, config, model ='mobile_sam'):
        self.config = config
        self.model_name = model
        self.video_name = None
        self.setup_model()
        print('You are initialising GroundingDINO. Please ensure you have changed the paths in the config file.')
        self.GROUNDING_DINO_CONFIG_PATH = self.config['checkpoints']['grounding_dino_config_path']
        self.GROUNDING_DINO_CHECKPOINT_PATH = self.config['checkpoints']['grounding_dino_checkpoint_path']
        self.grounding_dino_model = Model(model_config_path=self.config['checkpoints']['grounding_dino_config_path'], model_checkpoint_path=self.config['checkpoints']['grounding_dino_checkpoint_path'])
    
    def setup_model(self):
        if self.model_name == 'mobile_sam':
            checkpoint = torch.load(self.config['checkpoints']['mobile_sam_checkpoint'])
            mobile_sam = setup_model()
            mobile_sam.load_state_dict(checkpoint, strict=True)
            mobile_sam.to(device=self.config['sam']['device'])
            self.sam_predictor = SamPredictor(mobile_sam)
    
    def execute_model(self, image_path, type = 'box'):
        if isinstance(image_path, str):
            self.temp_image_path = image_path
            image = cv2.imread(f'{image_path}')
        else:
            image = image_path
            
        self.get_detections(image, caption=self.config['sam']['caption'])
        
        if self.config['sam']['nms_process']:
            self.nms_process()
        
        if self.config['sam']['save_max_conf_box']:
            self.max_conf_process()
        
        if self.config['sam']['remove_inner_box']:
            idxs = self.remove_inner_boxes(self.detections.xyxy)
            self.detections.xyxy = self.detections.xyxy[idxs]
            self.detections.confidence = self.detections.confidence[idxs]
            self.detections.class_id = self.detections.class_id[idxs]
            
        if type == 'box' and self.config['sam']['save_box']:
            self.save_rgb_annotations(image)
        
        if type == 'mask' and self.config['sam']['save_binary_masks']:
            self.get_masks(image)
            self.max_conf_process()
            self.save_binary_annotations()

    def get_detections(self, image, caption= None):
        self.detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=[caption],
            box_threshold=self.config['sam']['box_thresh'],
            text_threshold=self.config['sam']['text_thresh']
        )      

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
      self.sam_predictor.set_image(image)
      result_masks = []
      for box in xyxy:
          masks, scores, logits = self.sam_predictor.predict(
              box=box,
              multimask_output=True
          )
          index = np.argmax(scores)
          result_masks.append(masks[index])
      return np.array(result_masks)

    def get_masks(self, image):
        self.detections.mask = self.segment(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=self.detections.xyxy
        )
        
    def save_binary_annotations(self):
        self.binary_mask = self.detections.mask[0].astype(np.uint8)*255
        if not os.path.exists(f"{self.config['root']}/{self.config['sam']['output_binary_mask_folder']}/{self.video_name}/"):
            os.makedirs(f"{self.config['root']}/{self.config['sam']['output_binary_mask_folder']}/{self.video_name}/")
        
        cv2.imwrite(f"{self.config['root']}/{self.config['sam']['output_binary_mask_folder']}/{self.video_name}/{os.path.basename(self.temp_image_path)}", self.binary_mask)

    def save_rgb_annotations(self, image):
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_frame = image.copy()
        labels = [
            f"{self.config['sam']['caption']} {confidence:0.2f}" 
            for _, _, confidence, _, _, _ 
            in self.detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=self.detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=self.detections, labels=labels)
        if not os.path.exists(f"{self.config['root']}/{self.config['sam']['output_ann_box_folder']}/{self.video_name}/"):
            os.makedirs(f"{self.config['root']}/{self.config['sam']['output_ann_box_folder']}/{self.video_name}/")
        
        cv2.imwrite(f"{self.config['root']}/{self.config['sam']['output_ann_box_folder']}/{self.video_name}/{os.path.basename(self.temp_image_path)}", annotated_frame)

    def max_conf_process(self):
        conf = np.array([c for c in self.detections.confidence])
        idx = np.argsort(conf)[-1]
        self.detections.confidence = np.array([self.detections.confidence[idx]])
        self.detections.class_id = np.array([self.detections.class_id[idx]])
        self.detections.xyxy = np.reshape(self.detections.xyxy[idx], (1,4))
    
    def nms_process(self):
        
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(self.detections.xyxy), 
            torch.from_numpy(self.detections.confidence), 
            self.config['sam']['nms_thresh']
        ).numpy().tolist()
        self.detections.xyxy = self.detections.xyxy[nms_idx]
        self.detections.confidence = self.detections.confidence[nms_idx]
        self.detections.class_id = self.detections.class_id[nms_idx]
        
    def remove_inner_boxes(self, boxes, margin=2):
        """Remove boxes that are fully inside other boxes with tolerance."""
        keep = []
        keep_i = []
        for i, box in enumerate(boxes):
            is_inside = False
            for j, other in enumerate(boxes):
                if i != j and self.is_box_inside(box, other, self.config['sam']['remove_inner_box_margin']):
                    is_inside = True
                    break
            if not is_inside:
                keep.append(box)
                keep_i.append(i)
        return keep_i

    def is_box_inside(self, inner, outer, margin=20):
        """Check if inner box is inside outer box with a margin (in pixels)."""
        return (
            inner[0] >= outer[0] + margin and
            inner[1] >= outer[1] + margin and
            inner[2] <= outer[2] - margin and
            inner[3] <= outer[3] - margin
        )

    def largest_box_process(self):
        areas = []
        for box in self.detections.xyxy:
            width = box[2] - box[0]
            height = box[3] - box[1]
            areas.append(width * height)

        areas = np.array(areas)
        idx = np.argsort(areas)[-1]   
        self.largest_area = areas[idx]
        self.detections.confidence = np.array([self.detections.confidence[idx]])
        self.detections.class_id = np.array([self.detections.class_id[idx]])
        self.detections.xyxy = np.reshape(self.detections.xyxy[idx], (1,4))