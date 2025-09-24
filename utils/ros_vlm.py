from collections import deque, Counter
import openai
from openai import OpenAI
import base64, time
import os, cv2, json
# from cv_bridge import CvBridge
from utils import file_utils, img_proc_utils, mobilesam

class VLM:
    def __init__(self, config, args=None) -> None:
        self.config = file_utils.load_yaml(config)
        if 'root' not in self.config.keys():
            self.config['root'] = args.root
        if 'pkg_path' not in self.config.keys():
            self.config['pkg_path'] = ''
        self.root = self.config['root']
        self.model_name = self.config['exp']['model_name']
        self.max_retry_count = self.config['exp']['retry_count']
        self.last_message = None
        self.setup_gen_model()
        # self.bridge = CvBridge()
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

    def setup_gen_model(self):   
        if self.model_name == 'gemini':
            api_key = file_utils.load_yaml(os.path.join(self.config['root'],self.config['exp']['gemini_api_key_path']))['api_key']
            self.client = OpenAI(api_key = api_key, base_url = "https://generativelanguage.googleapis.com/v1beta/openai/")
        elif self.model_name == 'openai':
            api_key = file_utils.load_yaml(os.path.join(self.config['root'],self.config['exp']['openai_api_key_path']))['api_key']
            self.client = openai
        self.client.api_key = api_key

    def encode_rosmsg_image(self, rosmsg_image):
        cv_image = self.bridge.imgmsg_to_cv2(rosmsg_image,desired_encoding='passthrough')
        _, buffer = cv2.imencode('.jpg', cv_image)
        return base64.b64encode(buffer).decode("utf-8")
    
    def encode_image_path(self, image_path):
        with open(f'{image_path}', "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_image_string(self, image):
        if isinstance(image, str):
            # An image path
            encoded = self.encode_image_path(image)
        else:
            # ROS Image message
            encoded = self.encode_rosmsg_image(image)
        return f"data:image/jpg;base64,{encoded}"

    def create_prompt(self):
        filedata =  file_utils.read_prompt(os.path.join(self.config['root'],self.config['pkg_path'], self.config['exp']['prompt_file']))
        filedata = filedata.replace('REPLACE_DIRECTION_LIST', f"{self.config['exp']['directions']}")
        self.prompt = filedata

    # def create_temporal_message(self, arm_image_queue):
    #         self.create_prompt() 
    #         content_msg = [{
    #                         "type": "text",
    #                         "text": f"{self.prompt}"
    #                         }]
    #         for i in range(len(arm_image_queue)):
    #             content_msg.append({
    #                                 "type": "image_url",
    #                                 "image_url": { "url": self.get_image_string(arm_image_queue[i]), "detail": "high"}
    #                                })
                
    #         if self.config['exp']['rot_crops']:
                
    #             messages=[
    #                     {"role": "system", "content": "You are a helpful assistant capable of understanding navigational signs."},
    #                     {
    #                         "role": "user",
    #                         "content": content_msg  
    #                     }
    #                 ]    
                
    #         else:
    #             raise NotImplementedError
            
    #         return messages
    
    def create_temporal_message(self, arm_image_queue):
            self.create_prompt() 
            content_msg = [{
                            "type": "text",
                            "text": f"{self.prompt}"
                            }]
            for i in range(len(arm_image_queue)):
                content_msg.append({
                                    "type": "image_url",
                                    "image_url": { "url": self.get_image_string(arm_image_queue[i]), "detail": "high"}
                                   })
                
            if self.config['exp']['rot_crops']:
                
                messages=[
                        {"role": "system", "content": "You are a helpful assistant capable of understanding navigational signs."},
                        {
                            "role": "user",
                            "content": content_msg  
                        }
                    ]    
                
            else:
                raise NotImplementedError
            
            return messages

    def generative_prompting(self, arm_image_queue, retry_count):
        if not retry_count:
            self.last_message = self.create_temporal_message(arm_image_queue)
        completion = self.client.chat.completions.create(
            model=self.config['exp']['model_version'],
            messages = self.last_message, 
            n=self.config['exp']['voting_iter_count']
        )
                        
        temp_final = self.process_gpt_output(completion.choices, type='gen')
        return temp_final
           
    def process_gpt_output(self, resp, type='discr'):
        temp = [choice.message.content.replace('\n', '').replace('#','') for choice in resp]
        temp_1 = [te.replace('json','') for te in temp]
        temp_2 = [te.replace('```','') for te in temp_1]
        temp_3 = [te.replace('python','') for te in temp_2]
        temp_4 = [te.replace('\t','') for te in temp_3]
        if type == 'discr':
            temp_final = [json.loads(te) for te in temp_4]
            dr_lst = []
            for dic in temp_final:
                assert dic['direction'].upper() in self.config['exp']['directions'], print('Got unknown direction ... reprompting')
                dr_lst.append(dic['direction'].upper())
            #if the response was a tuple list
            # dr_lst = []
            # for re in temp_4:
            #     re = ast.literal_eval(re)
            #     assert re[1].upper() in self.config['exp']['directions'], pdb.set_trace()
            #     dr_lst.append(re[1].upper())
            temp_final_0 = Counter(dr_lst).most_common()
            temp_final = [(t[0],t[1]/self.config['exp']['discr_iter_count']) for t in temp_final_0]
            
        elif type =='gen':
            temp_final = [json.loads(te) for te in temp_4]
            # print(">>>")
            # print(temp_final)
            # print("<<<")
            # for dic in temp_final:
            #     for _,v in dic.items():
            #         if isinstance(v,str): 
            #             assert v.upper() in self.config['exp']['directions'], print('Wrong Direction Output... reprompting....')
            #         elif isinstance(v,list) or isinstance(v,tuple): 
            #             raise ValueError('Got a tuple.. not expected ... we reprompt')
            #             # for vi in v:
            #             #     # print(vi)
            #             #     assert vi.upper() in self.config['exp']['directions'], print('Wrong Direction Output... reprompting....') 
            #         else:
            #             raise ValueError('Got a dict.. not expected ... we reprompt')
        return temp_final
        
    def prompt_model(self, crop_image_queue, return_json=True):
        retry_count = 0
        while retry_count < self.max_retry_count:
            try:
                #step-1 generative prompting
                gen_temp_final = self.generative_prompting(crop_image_queue, retry_count)
                print(gen_temp_final)
                break
            
            except Exception as e:
                print(e)
                print('failed to get a py dict : re-prompting....')
                retry_count += 1

        if retry_count >= self.max_retry_count:
            print(f'prompting failure')
            gen_temp_final = -1
            
        if return_json:
            return json.dumps(gen_temp_final)
        else:
            return gen_temp_final
                
    def create_crops(self, img_path=None):
        '''uses your crop model to create crops of navigational sign boards that are fed to VLM'''
        if self.config['exp']['crop_gen_model'] == 'gemini-2.0-flash':
            raise NotImplementedError
                    
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
