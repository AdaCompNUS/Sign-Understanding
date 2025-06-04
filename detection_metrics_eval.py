import cv2 , pdb
import numpy as np
import os, ast
import matplotlib.pyplot as plt
from utils import img_proc_utils, mobilesam, file_utils, coco_script
from google import genai
from google.genai import types
import time
from PIL import Image
from io import BytesIO
from utils import file_utils

def plotLabels(im,frameName,  bb, text, symbols, pred_text, pred_symbols):

	h,w,c = im.shape
	left, bottom, width, height = bb[0] - 0.5*bb[2], bb[1] - 0.5*bb[3], bb[2], bb[3]
	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
	axs[0].imshow(im)
	rect = plt.Rectangle((left, bottom), width, height,fc='none', edgecolor="red", alpha=0.5, lw=5)
	axs[0].add_patch(rect)

	canvas = 255 * np.ones((h, w, c), dtype=np.uint8)
	axs[1].imshow(canvas)
	fontsize=8
	pad = 40
	axs[1].text(20, 50, "Ground Truth Text:", fontsize=fontsize, weight='bold')
	idx = 1
	for t, d in text.items():
		axs[1].text(20, 50 + idx * pad, t + " - " + d, fontsize=fontsize)
		idx += 1

	axs[1].text(20, 50 + idx * pad, "Ground Truth Symbols:", fontsize=fontsize, weight='bold')
	idx += 1
	for s, d in symbols.items():
		axs[1].text(20, 50 + idx * pad, str(s) + " - " + str(d), fontsize=fontsize)
		idx += 1

	idx += 5
	axs[1].text(20, 50 + idx * pad, "Frame Path:", fontsize=fontsize, weight='bold')
	idx += 1
	axs[1].text(20, 50 + idx * pad, frameName, fontsize=fontsize)
	
	axs[1].text(1000, 50, "Predicted Text:", fontsize=fontsize, weight='bold')
	idx = 1
	for t, d in pred_text.items():
		axs[1].text(1000, 50 + idx * pad, str(t) + " - " + str(d), fontsize=fontsize)
		idx += 1

	axs[1].text(1000, 50 + idx * pad, "Predicted Symbols:", fontsize=fontsize, weight='bold')
	idx += 1
	for s, d in pred_symbols.items():
		axs[1].text(1000, 50 + idx * pad, str(s) + " - " + str(d), fontsize=fontsize)
		idx += 1
	plt.tight_layout()
	plt.show()
	fig.clf()

def save_bbox_gt(img_path, bb, i, cropped_gt_dir):
	new_bb = [bb[0] - 0.5*bb[2], bb[1] - 0.5*bb[3], bb[0] + 0.5*bb[2], bb[1] + 0.5*bb[3]]
	crop_img = img_proc_utils.crop_buffer_bbox(img_path, new_bb, buffer=0)
	n = os.path.basename(img_path)[:-4] + f'_{i}' + os.path.basename(img_path)[-4:]
	cv2.imwrite(f"{cropped_gt_dir}/{n}", crop_img)
	return n

def post_process_gemini_response(text, width, height):
	lines = text.splitlines()
	for i, line in enumerate(lines):
		if line == "```json":
			json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
			json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
			break  # Exit the loop once "```json" is found
	temp_list = [ast.literal_eval(json_output)[i]['box_2d'] for i in range(len(ast.literal_eval(json_output)))]
	temp_conf_list = [ast.literal_eval(json_output)[i]['confidence'] for i in range(len(ast.literal_eval(json_output)))] 
	temp_list_2 = [[temp_list[i][1]*width/1000, temp_list[i][0]*height/1000, temp_list[i][3]*width/1000, temp_list[i][2]*height/1000]  for i in range(len(temp_list))]
	print(temp_list_2)
	return temp_list_2, temp_conf_list

if __name__ == '__main__':
	'''
	you can create the recogntion_dataset from this script
	'''
	root = '/home/ayush/arxiv' #UPDATE PATH HERE
	print('Please ensure you have changed the root path in this script ...\n\n')

	config = file_utils.load_yaml(os.path.join(root, 'config/detection_eval_config.yaml'))
	
	detcn_images = os.path.join(root, config['exp']['dataset_dir'])
	gtJsonPath = os.path.join(root, config['exp']['detcn_gt']) 
	predJsonPath = os.path.join(root, config['exp']['recg_pred']) 
	recognition_cropped_gt_dir = os.path.join(root, 'gt/recognition_dataset') 
	file_utils.makeCheck(recognition_cropped_gt_dir)

	mode = config['mode']
	api_key = file_utils.load_yaml(os.path.join(root,config['exp']['gemini_api_key_path']))['api_key']
	
	client = genai.Client(api_key = api_key)
	bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with keys as box_2d and confidence. Never return masks or code fencing. Limit to 25 objects. Add a confidence score for each of your detection.
    """

	prompt = "Detect ALL 2d bounding boxes for all navigational sign boards present in this image. Also give confidence for each bounding box."

	safety_settings = [
		types.SafetySetting(
			category="HARM_CATEGORY_DANGEROUS_CONTENT",
			threshold="BLOCK_ONLY_HIGH",
		),
	]
     
	gtJson = file_utils.read_json(gtJsonPath)
	predJson = file_utils.read_json(predJsonPath) #used for viusalising your recognition results 
	
	cnt = 0
	if mode == 'detection':
		g_sam = mobilesam.GroundedSAM(config)
		g_sam.video_name = 'detection_0000'
		
	gt_dict = {}
	pred_dict = {}
	for idx, item in enumerate(gtJson):
		if idx == 10: break
		gt_boxes = []
		frame_name = item['imagePath']
		if os.path.exists(os.path.join(detcn_images,item['imagePath'])):
			im = cv2.imread(os.path.join(detcn_images,item['imagePath']))
			width, height = im.shape[1], im.shape[0]
		else:
			pdb.set_trace()
			
		annotations = item['annotation']
		if mode == 'detection':
			if config['exp']['model_name'] == 'g-dino':
				g_sam.execute_model(os.path.join(detcn_images,item['imagePath']))
			elif config['exp']['model_name'] == 'gemini':
				im = Image.open(BytesIO(open(os.path.join(detcn_images,item['imagePath']), "rb").read()))
				im.thumbnail([1024,1024], Image.Resampling.LANCZOS)
				while True:
					try:
						response = client.models.generate_content(
							model=config['exp']['model_version'],
							contents=[prompt, im],
							config = types.GenerateContentConfig(
								system_instruction=bounding_box_system_instructions,
								temperature=0.5,
								safety_settings=safety_settings,
							)
						)
						print(response.text)
						gemini_bbox_list, gemini_conf_list = post_process_gemini_response(response.text, width, height)
						break
					except Exception as e:
						print(e)
						print('re-prompting....')

		for i,annotation in enumerate(annotations):

			bb = annotation['boundingBox']
			gt_boxes.append(bb)
			mixed = annotation['mixed'] if "mixed" in annotation else {}
			if annotation['text labels'] == {} and annotation['symbol labels'] == {} and mixed == {}:
				continue
			else:
				#only saving the readbale signs in recg_dataset from fullpipeline_dataset
				new_name = save_bbox_gt(os.path.join(detcn_images,item['imagePath']), bb, i, recognition_cropped_gt_dir)
				if mode == 'debug':
					_pred = [pr['voted response'] for pr in predJson if pr['frame_path'] == os.path.join(recognition_cropped_gt_dir, new_name)]
					pred = ast.literal_eval(_pred[0])
					gt_text = annotation['text labels']
					pred_text = pred['T']
					gt_symbols = annotation['symbol labels']
					pred_symbols = pred['S']

					if len(gt_text) or len(gt_symbols):
						cnt += 1
						plotLabels(im,os.path.join(recognition_cropped_gt_dir, new_name), bb, gt_text, gt_symbols, pred_text, pred_symbols)
		
		if mode == 'detection':
			gt_dict[idx] = gt_boxes #xywh-center
			if config['exp']['model_name'] == 'g-dino':
				pred_dict[idx] = [g_sam.detections.xyxy.tolist(), g_sam.detections.confidence.tolist()] #xyxy
			elif config['exp']['model_name'] == 'gemini':
				pred_dict[idx] = [gemini_bbox_list, gemini_conf_list]

	if mode == 'detection':
		coco_script.evaluate_coco(gt_dict, pred_dict, iou_thresholds=[0.25, 0.5, 0.75])

