import os
import json
import cv2
import time
from collections import deque
from tqdm import tqdm
import math
import base64, json, os
import openai, cv2
from utils import img_proc_utils, mobilesam, file_utils
from utils import process_utils
from ast import literal_eval
from collections import deque
from tqdm import tqdm
import ast, argparse
from openai import OpenAI
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

from utils.ros_vlm import VLM

def frame_paths_from_folder(folder_path):
    frames= []
    for f in os.listdir(folder_path):
        frame = os.path.join(folder_path,f)
        frames.append(frame)
    return frames

class Baseline_VLM:
    def __init__(self, config, args) -> None:
        self.root = args.root
        self.config = file_utils.load_yaml(config)
        self.config['root'] = self.root

        self.img_queue = deque(maxlen=self.config['exp']['prompt_img_len'])
        self.video_name = None
        self.model_name = self.config['exp']['model_name']
        self.rot_crops = self.config['exp']['rot_crops'] #if you want to prompt the crops
        self.prompt_img_quality = self.config['exp']['prompt_img_quality']
        self.retry_count = self.config['exp']['retry_count']
        self.last_message = None
        self.symbol_list = file_utils.read_gtlabels(os.path.join(self.root, self.config['exp']['symbols']))[:10]
        self.setup_model()
        if self.config['name'] == 'full-pipeline':
            self.setup_crop_model()

    def setup_crop_model(self):
        if self.config['exp']['crop_gen_model'] == 'gemini-2.0-flash':
            self.crop_model = genai.Client(api_key = file_utils.load_yaml(os.path.join(self.root,self.config['exp']['gemini_detection_api_key']))['api_key'])
            self.crop_safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
            ]
        elif self.config['exp']['crop_gen_model'] == 'g-dino':
            groundedsam = mobilesam.GroundedSAM(self.config) 
            self.crop_model = groundedsam
        
    def setup_model(self):
        if self.model_name == 'openai':
            api_key = file_utils.load_yaml(os.path.join(self.root,self.config['exp']['openai_api_key_path']))['api_key']
            self.client = openai
        
        if self.model_name == 'gemini':
            api_key = file_utils.load_yaml(os.path.join(self.root,self.config['exp']['gemini_api_key_path']))['api_key']
            self.client = OpenAI(api_key = api_key, base_url = "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.client.api_key = api_key
        
    def create_prompt(self):
        filedata =  file_utils.read_prompt(os.path.join(self.root,self.config['exp']['prompt_file']))
        filedata = filedata.replace('REPLACE_DIRECTION_LIST', f"{self.config['exp']['directions']}")
        filedata = filedata.replace('GT_SYMBOL_LIST', f"{self.symbol_list}")
        self.prompt = filedata
          
    def create_crops(self, img_path=None):
        '''uses your crop model to create crops of navigational sign boards that are fed to VLM'''
        if self.config['exp']['crop_gen_model'] == 'gemini-2.0-flash':
            if isinstance(img_path, deque) and len(img_path) == 1:
                file_utils.makeCheck( f"{self.root}/{self.config['exp']['gem_output_rotated_crop_folder']}/{self.video_name}")
                rot_img_path = f"{self.root}/{self.config['exp']['gem_output_rotated_crop_folder']}/{self.video_name}/{os.path.basename(img_path[0])}"
                
                im = cv2.imread(img_path[0])
                width, height = im.shape[1], im.shape[0]
                im = Image.open(BytesIO(open(img_path[0], "rb").read()))
                im.thumbnail([1024,1024], Image.Resampling.LANCZOS)
                while True:
                    try:
                        time.sleep(3)
                        response = self.crop_model.models.generate_content(
                            model=self.config['exp']['crop_gen_model'],
                            contents=[self.config['exp']['crop_prompt'], im],
                            config = types.GenerateContentConfig(
                                system_instruction=self.config['exp']['crop_bbox_instr'],
                                temperature=self.config['exp']['crop_temp'],
                                safety_settings=self.crop_safety_settings,
                            )
                        )
                        print(response.text)
                        gemini_bbox_list, gemini_conf_list = vlm.post_process_gemini_response(response.text, width, height)
                        break
                    except Exception as e:
                        print(e)
                        print('re-prompting....')
                for i in range(len(gemini_bbox_list)): 
                    try:
                        cv2.imwrite(f'{rot_img_path[:-4]}_{i}.jpg',img_proc_utils.crop_buffer_bbox(img_path[0], gemini_bbox_list[i], buffer=10))
                    except Exception as e:
                        print(e)
                        continue
                self.img_dict[0]['rot_crops'] = [f'{rot_img_path[:-4]}_{idx}.jpg' for idx in range(len(gemini_bbox_list))]
                self.img_dict[0]['bbox'] = gemini_bbox_list
                self.img_dict[0]['conf'] = gemini_conf_list
                    
        elif self.config['exp']['crop_gen_model'] == 'g-dino':    
            if isinstance(img_path, str):
                rot_img_path = f"{self.root}/{self.config['sam']['output_rotated_crop_folder']}/{self.video_name}/{os.path.basename(img_path)}"
                if not os.path.exists(f"{self.root}/{self.config['sam']['output_rotated_crop_folder']}/{self.video_name}"):
                    os.makedirs(f"{self.root}/{self.config['sam']['output_rotated_crop_folder']}/{self.video_name}")
                
                temp_crop_name_list = list()
                rot_im_lst , bbox_lst, conf_lst = img_proc_utils.get_rotated_image_crops(img_path, self.crop_model)
                for idx, rot_im in enumerate(rot_im_lst):
                    cv2.imwrite(f"{rot_img_path[:-4]}_{idx}.jpg", rot_im)
                    self.img_dict['full'] = f"{img_path}"
                    temp_crop_name_list.append(f"{rot_img_path[:-4]}_{idx}.jpg")  
                self.img_dict['rot_crops'] = temp_crop_name_list
                self.img_dict['bbox'] = bbox_lst
                self.img_dict['conf'] = conf_lst
                    
            elif isinstance(img_path, deque):
                for x, img_p in enumerate(img_path):
                    rot_img_path = f"{self.root}/{self.config['sam']['output_rotated_crop_folder']}/{self.video_name}/{os.path.basename(img_p)}"
                    if not os.path.exists(f"{self.root}/{self.config['sam']['output_rotated_crop_folder']}/{self.video_name}"):
                        os.makedirs(f"{self.root}/{self.config['sam']['output_rotated_crop_folder']}/{self.video_name}")
                
                    temp_crop_name_list = list()
                    rot_im_lst , bbox_lst , conf_lst = img_proc_utils.get_image_crops(img_p, self.crop_model)
                    for idx, rot_im in enumerate(rot_im_lst):
                        cv2.imwrite(f"{rot_img_path[:-4]}_{idx}.jpg", rot_im)
                        self.img_dict[x]['full'] = f"{img_p}"
                        temp_crop_name_list.append(f"{rot_img_path[:-4]}_{idx}.jpg")
                    self.img_dict[x]['rot_crops'] = temp_crop_name_list
                    self.img_dict[x]['bbox'] = bbox_lst
                    self.img_dict[x]['conf'] = conf_lst
                    

    def create_message(self, image_path):
        
        self.create_prompt()
        if not self.config['exp']['rot_crops']:
           # used for recognition evaluation (directly feeding in the crop (already saved) as a whole image)
            if not self.img_dict[0]['full']:
                self.img_dict[0]["full"] = image_path[0]
            
            all_messages = []
            messages=[
                    {"role": "system", "content": "You are a helpful assistant capable of understanding navigational signs."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{self.prompt}"
                            },
                            {
                                "type": "image_url",
                                "image_url": { "url": self.get_image_string(self.img_dict[0]['full']), "detail": self.prompt_img_quality }
                            }
                        ]
                    }
                ]
            all_messages.append(messages)
        
        elif self.config['exp']['rot_crops']:
            #used for full pipeline evaluation (feeding a full image and then creating all crops and feeding in sequentially)
            if not self.img_dict[0]['rot_crops']:
                self.create_crops(image_path)
            all_messages = []
            for rot_img in self.img_dict[0]['rot_crops']:
                messages=[
                        {"role": "system", "content": "You are a helpful assistant capable of understanding navigational signs."},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{self.prompt}"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": { "url": self.get_image_string(rot_img), "detail": self.prompt_img_quality }
                                }
                            ]
                        }
                    ]
                all_messages.append(messages)
        return all_messages

    def post_process_gemini_response(self, text, width, height):
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
    
    def process_gpt_output(self, resp):
        temp = [choice.message.content.replace('\n', '') for choice in resp]
        temp_1 = [te.replace('json','') for te in temp]
        temp_2 = [te.replace('```','') for te in temp_1]
        temp_3 = [te.replace('python','') for te in temp_2]
        temp_4 = [te.replace('\t','') for te in temp_3]
        temp_final = [json.loads(te) for te in temp_4]
        # print(">>>")
        # print(temp_final)
        # print("<<<")
        return temp_final

    def prompt_model(self, image_path):
        if self.config['exp']['prompt_img_len'] == 1:
            self.list_message = self.create_message(image_path)
            mega_resp = []
            for mes in self.list_message:
                retry_count = 0
                while retry_count < self.retry_count:
                    try:
                        # if self.model_name == 'gemini': time.sleep(2)
                        completion = self.client.chat.completions.create(
                        model=self.config['exp']['model_version'],
                        messages = mes,
                        n=self.config['exp']['voting_iter_count']
                        )
                        prc_resp = self.process_gpt_output(completion.choices)
                        print(prc_resp)
                        print(type(prc_resp), type(prc_resp[0]))
                        mega_resp.append(prc_resp)
                        break
                    except Exception as e:
                        print(e)
                        print('redoing...prompting error for this crop...')
                        retry_count += 1

                if retry_count >= self.retry_count:
                    print(f'gave up ... prompting failure')
                    mega_resp.append(-1)        
                    
        return mega_resp
    
    def encode_image(self, image_path):
        with open(f'{image_path}', "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_image_string(self, image_path):
        return f"data:image/jpg;base64,{self.encode_image(image_path)}"
              
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="baseline")
    parser.add_argument('--root', type=str, help='/path/to/Sign-Understanding')
    args = parser.parse_args()

    root = args.root

    while True:
        r = input("Recognition or Full-Pipeline Evaluation? R/F\n")
        if r.upper() == 'R':
            config = os.path.join(root, 'config/recognition_eval_config.yaml')  
            break
        elif r.upper() == 'F':
            config =  os.path.join(root,'config/full_pipeline_eval_config.yaml') 
            break
        print('Please enter valid response....')

    vlm = VLM(config, args)
    # vlm = VLM(config)
    print(f"You are using this config: {config}")
    print(f"You are using this model: {vlm.config['exp']['model_name']} and version {vlm.config['exp']['model_version']}")
    print(f"You are using this prompt: {vlm.config['exp']['prompt_file']} and symbol_list {vlm.config['exp']['symbols']}")
    confidence_tries = vlm.config['exp']['voting_iter_count']
    
    if vlm.config['exp']['source'] == 'selected-frames' and vlm.config['name'] == 'recognition':
        names = [f"{vlm.config['exp']['model_name']}"]
    elif vlm.config['exp']['source'] == 'selected-frames' and vlm.config['name'] == 'full-pipeline':
        names = [f"{vlm.config['exp']['crop_gen_model']}-{vlm.config['exp']['model_name']}"]
        
    print("Starting eval...")
    for nm in names:
        vlm.video_name = nm
        if vlm.config['name'] == 'full-pipeline':
            vlm.crop_model.video_name = nm
        
        if vlm.config['exp']['source'] == 'selected-frames' and vlm.config['name'] == 'recognition':
            all_frame_folder = os.path.join(root, vlm.config['exp']['dataset_dir']) 
        elif vlm.config['exp']['source'] == 'selected-frames' and vlm.config['name'] == 'full-pipeline':
            all_frame_folder = os.path.join(root, vlm.config['exp']['dataset_dir']) 
            bbox_gt_path = os.path.join(root,vlm.config['exp']['recg_groundtruth'])
            bbox_gt_dict = dict()
            gt_resp_dict = dict()
            for idx, item in enumerate(file_utils.read_json(bbox_gt_path)):
                gt_boxes = []
                recg_ann = []
                anns = item['annotation']
                for ix, ann in enumerate(anns): 
                    bb = ann['boundingBox']
                    gt_boxes.append([int(bb[0] - 0.5*bb[2]), int(bb[1] - 0.5*bb[3]), int(bb[0] + 0.5*bb[2]), int(bb[1] + 0.5*bb[3])])
                    if 'mixed' in ann:
                        recg_ann.append({'text labels' : ann['text labels'], 
                                        'symbol labels': ann['symbol labels'],
                                        'mixed': ann['mixed']})
                    else:
                        recg_ann.append({'text labels': ann['text labels'], 
                                        'symbol labels' : ann['symbol labels']})
                bbox_gt_dict[item['imagePath']] = gt_boxes #xyxy list
                gt_resp_dict[item['imagePath']] = recg_ann
        frame_paths = frame_paths_from_folder(all_frame_folder)
        
        base = 0
        correct = 0
        history = list()
        match_history = list()
        vlm.img_queue = deque(maxlen=vlm.config['exp']['prompt_img_len'])
        
        print(f"Evaluating {len(frame_paths)} frames...")
        bbox_preds = dict()
        for cnt, frame_path in tqdm(enumerate(frame_paths)):
            result = dict()
            vlm_decider_flag = True 
            vlm.img_queue.append(frame_path)
            vlm.img_dict = {i: {'full': None, 'rot_crops': None, 'bbox': None} for i in range(len(vlm.img_queue))}
            vlm.last_message = None
            # resp = vlm.prompt_model(vlm.img_queue)

            if vlm.config['exp']['rot_crops']:
                if len(vlm.img_queue) != 1:
                    raise NotImplementedError
                # Should only take a single image path
                tmp_deq = deque(maxlen=1)
                tmp_deq.append(frame_path)
                vlm.create_crops(tmp_deq)

            resp = []
            for rot_im in vlm.img_dict[0]['rot_crops']:
                individual_resp = vlm.prompt_model([rot_im], return_json=False)
                resp.append(individual_resp)
            
            outputs = []
            for r in resp:
                if r != -1:
                    dict_final_keys = process_utils.most_common_keys(r, thresh=math.ceil(vlm.config['exp']['voting_iter_count']/2))
                    result = process_utils.get_most_common_directions(dict_final_keys,r)
                else:
                    result = {}
                outputs.append(result)
            if vlm.config['name'] == 'full-pipeline':
                bbox_preds = [[int(x) for x in vlm.img_dict[0]['bbox'][i]] for i in range(len(outputs))]
                frame_gt_boxes = bbox_gt_dict[os.path.basename(frame_path)]
                frame_gt_anns = gt_resp_dict[os.path.basename(frame_path)]
            
                matches = img_proc_utils.greedy_match(bbox_preds, frame_gt_boxes, vlm.config['exp']['bbox_match_iou'])
                for (p_i, g_i, iu) in matches:
                    match_history.append({
                        'frame_path': frame_path,
                        'crop_bbox': bbox_preds[p_i],
                        'confidence': float(vlm.img_dict[0]['conf'][p_i]),
                        'match_gt_bbox': frame_gt_boxes[g_i],
                        'iou': float(iu),
                        'voted response': outputs[p_i],
                        'gt response': frame_gt_anns[g_i]})

                pred_gt_match = []
                for id, rs in enumerate(outputs):
                    history.append({
                        'frame_path': frame_path,
                        'prompt_imgs': list(vlm.img_queue),
                        'crop_path': vlm.img_dict[0]['rot_crops'][id],
                        'crop_bbox': tuple(int(x) for x in vlm.img_dict[0]['bbox'][id]),
                        'match_gt_bbox': [] ,
                        'match_ann_gt_bbox' : [], 
                        'voted response': f'{rs}',
                        'confidence': float(vlm.img_dict[0]['conf'][id]),
                    })
            else:
                history.append({
                        'frame_path': frame_path,
                        'voted response': f'{outputs[0]}',
                    })
            vlm.img_queue.popleft()
    
        file_utils.makeCheck(os.path.join(root,vlm.config['exp']['result_dir'], nm))
        res_dir = os.path.join(root,vlm.config['exp']['result_dir'], nm)
        if vlm.config['name'] == 'recognition':
            file_utils.save_file_json(f"{res_dir}/recognition_results.json", history)
        
        else:
            #we need this unprocessedGTResponse for full-pipeline evaluations
            file_utils.save_file_json(f"{res_dir}/all-full-pipeline.json", history)
            file_utils.save_file_json(f"{res_dir}/unprocessedGTResponse.json", match_history)
