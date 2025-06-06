import numpy as np
import os, pdb,ast
import argparse
from utils import metrics, file_utils
import pandas as pd
import itertools

def reformat_full_pipeline_results(fullpipelineGtPath, recg_gt):
    recg_gt = file_utils.read_json(recg_gt)

    gt = file_utils.read_json(fullpipelineGtPath)
    nameList = []
    new_gt = []
    for e1 in gt:
        if e1['frame_path'] in nameList:
            continue
        else:
            nameList.append(e1['frame_path'])
            temp = {}
            temp['imagePath'] = e1['frame_path']
            temp['annotation'] = []
            new_gt.append(temp)
    for ei in new_gt:
        name = ei['imagePath']
        for ej in gt:
            if ej['frame_path'] == name:
                tem = {}
                tem['crop_bbox'] = ej['crop_bbox']
                tem['match_gt_bbox'] = ej['match_gt_bbox']
                tem['confidence'] = ej['confidence']
                tem['iou'] = ej['iou']
                tem['voted response'] = ej['voted response']
                tem['gt response'] = ej['gt response']
                ei['annotation'].append(tem)
    new_namelist = [os.path.basename(n) for n in nameList]

    #adding images that were completely mmissed by the pipeline (no bbox in them crossed the 0.5 iou threshold or no detection in that image)
    if args.recall:
        for e in recg_gt:
            if e['imagePath'] not in new_namelist:
                temp = {}
                temp['imagePath'] = e['imagePath']
                temp['annotation'] = []
                new_gt.append(temp)
                new_namelist.append(e['imagePath'])
    i = 0
    
    for e_check in recg_gt:
        for e_2 in new_gt:
            if os.path.basename(e_2['imagePath']) == e_check['imagePath']:
                for ann in e_check['annotation']:
                    mx = ann['mixed'] if "mixed" in ann else {}
                    if ann['text labels'] == {} and ann['symbol labels'] == {} and mx == {}:
                        continue
                    else:
                        bboxlist = [e_2['annotation'][i]['match_gt_bbox'] for i in range(len(e_2['annotation']))]
                        check_bb = [int(ann['boundingBox'][0] - 0.5*ann['boundingBox'][2]), int(ann['boundingBox'][1] - 0.5*ann['boundingBox'][3]), int(ann['boundingBox'][0] + 0.5*ann['boundingBox'][2]), int(ann['boundingBox'][1] + 0.5*ann['boundingBox'][3])]
                        #adding signs where there was no pred matched to the readable gt as either non-detection or non crossing 0.5 threshold
                        if check_bb not in bboxlist:
                            tempDict = {}
                            tempDict['crop_bbox'] = []
                            tempDict['match_gt_bbox'] = check_bb
                            tempDict['confidence'] = 0
                            tempDict['iou'] = 0
                            tempDict['voted response'] = {}
                            tempDict['gt response'] = {"text labels": ann['text labels'] , "symbol labels": ann['symbol labels'], "mixed": mx}
                            if args.recall:
                                e_2['annotation'].append(tempDict)
                                i += 1
                                
    file_utils.makeCheck(os.path.join(os.path.dirname(fullpipelineGtPath),'processed'))
    newPath = os.path.join(os.path.dirname(fullpipelineGtPath),'processed/processedGTResponse.json')
    file_utils.save_file_json(newPath, new_gt)
    return newPath 

def getMaxIndex(sim_mat):
    max_val = np.max(sim_mat)
    if max_val != 0.0:
        return np.unravel_index(np.argmax(sim_mat), sim_mat.shape)
    else:
        return -1

def hungarian_local_implementation(func, gt_copy, pred_copy, loc_mapping):
    rows = []
    if len(gt_copy) == 0 or len(pred_copy) == 0:
        return loc_mapping
    
    for loc in gt_copy:
        rows.append(func(loc, pred_copy))
    sim_mat = np.array(rows)
    og_len = min(len(gt_copy), len(pred_copy))
    for _ in range(og_len):
        max_idx = getMaxIndex(sim_mat)
        if max_idx != -1:    
            loc_mapping[gt_copy[max_idx[0]]] = pred_copy[max_idx[1]] 
            pred_copy.remove(pred_copy[max_idx[1]])
            gt_copy.remove(gt_copy[max_idx[0]])   
            if np.shape(sim_mat)[0] == 1 or np.shape(sim_mat)[1] == 1:
                break
            sim_mat = np.delete(sim_mat, max_idx[0], axis=0)
            sim_mat = np.delete(sim_mat, max_idx[1], axis=1)
            
        else:
            break
    return loc_mapping

def process_raw_dict(dct, mixed_dict=None):
    return_dict = {}
    for k,v in dct.items():
        if isinstance(v,str):
            if 'DOWN' not in v.upper().replace(' ','') and 'AND' not in v.upper().replace(' ',''):
                return_dict[k.upper().replace(' ','')] = v.upper().replace(' ','')
        elif isinstance(v,list) or isinstance(v,tuple):
                return_dict[k.upper().replace(' ','')] = tuple(sorted([vi.upper().replace(' ','') for vi in v if 'DOWN' not in vi.upper().replace(' ','') and 'AND' not in vi.upper().replace(' ','')]))
    if mixed_dict != {} and mixed_dict is not None:
        for k,v in mixed_dict.items():
            if isinstance(v,str):
                if 'DOWN' not in v.upper().replace(' ','') and 'AND' not in v.upper().replace(' ',''):
                    return_dict[k.upper().replace(' ','')] = v.upper().replace(' ','')
            elif isinstance(v,list) or isinstance(v,tuple):
                    return_dict[k.upper().replace(' ','')] = tuple(sorted([vi.upper().replace(' ','') for vi in v if 'DOWN' not in vi.upper().replace(' ','') and 'AND' not in vi.upper().replace(' ','')]))
    return return_dict
    
def give_match_score(txt_loc_mapping, sym_loc_mapping, txtpred, txtgt, sympred, symgt):
    txt_match_score = 0
    sym_match_score = 0
    for txt_loc_ky in txt_loc_mapping:
        if isinstance(txtpred[txt_loc_mapping[txt_loc_ky]], tuple) and isinstance(txtgt[txt_loc_ky], tuple):
            txt_match_score += len(set(txtpred[txt_loc_mapping[txt_loc_ky]]) & set(txtgt[txt_loc_ky])) / len(set(txtgt[txt_loc_ky])) # set intersection over gt length
        elif isinstance(txtpred[txt_loc_mapping[txt_loc_ky]], tuple) and isinstance(txtgt[txt_loc_ky], str):
            txt_match_score += 0 # if gt was 1 but you predict a list -- no points given
        elif isinstance(txtpred[txt_loc_mapping[txt_loc_ky]], str) and isinstance(txtgt[txt_loc_ky], str):
            txt_match_score += txtpred[txt_loc_mapping[txt_loc_ky]] == txtgt[txt_loc_ky]
        elif isinstance(txtpred[txt_loc_mapping[txt_loc_ky]], str) and isinstance(txtgt[txt_loc_ky], tuple):
            txt_match_score += (txtpred[txt_loc_mapping[txt_loc_ky]] in txtgt[txt_loc_ky]) / len(set(txtgt[txt_loc_ky]))
    
    for sym_loc_ky in sym_loc_mapping:
        if isinstance(sympred[sym_loc_mapping[sym_loc_ky]], tuple) and isinstance(symgt[sym_loc_ky], tuple):
            sym_match_score += len(set(sympred[sym_loc_mapping[sym_loc_ky]]) & set(symgt[sym_loc_ky])) / len(set(symgt[sym_loc_ky])) # set intersection over gt length
        elif isinstance(sympred[sym_loc_mapping[sym_loc_ky]], tuple) and isinstance(symgt[sym_loc_ky], str):
            sym_match_score += 0 # if gt was 1 but you predict a list -- no points given
        elif isinstance(sympred[sym_loc_mapping[sym_loc_ky]], str) and isinstance(symgt[sym_loc_ky], str):
            sym_match_score += sympred[sym_loc_mapping[sym_loc_ky]] == symgt[sym_loc_ky]
        elif isinstance(sympred[sym_loc_mapping[sym_loc_ky]], str) and isinstance(symgt[sym_loc_ky], tuple):
            sym_match_score += (sympred[sym_loc_mapping[sym_loc_ky]] in symgt[sym_loc_ky]) / len(set(symgt[sym_loc_ky]))

    return txt_match_score, sym_match_score

def process_jsons_for_predictions(args):
    print(args)
    txt_indiv_precision = 0
    txt_indiv_recall = 0
    sym_indiv_precision = 0
    sym_indiv_recall = 0

    total_sign_count = 0
    global_perf_sign = 0
    txt_global_perf_sign = 0
    sym_global_perf_sign = 0
    txt_global_match_score = 0
    sym_global_match_score = 0
    overall_global_match_score = 0

    txt_global_P_score = 0
    txt_global_G_score = 0
    overall_global_P_score = 0
    overall_global_G_score = 0
    
    sym_global_P_score = 0
    sym_global_G_score = 0

    mapping_history = []

    if args.f[-5:] == '.json':
        predJson = file_utils.read_json(os.path.join(args.op_dir, args.f))
        gtJson = file_utils.read_json(args.gt_path)
        for entry in gtJson:

            img_path = os.path.join(args.dataset_dir, entry['imagePath'])
            for i,ann in enumerate(entry['annotation']):
                if args.exp_name == 'full-pipeline':                
                    ann_check = ann['gt response']
                elif args.exp_name == 'recognition':
                    ann_check = ann
                mx = ann_check['mixed'] if "mixed" in ann_check else None
                
                if ann_check['text labels'] == {} and ann_check['symbol labels'] == {} and mx is None: # removing signs that weren't readable by humans
                    continue
                else:
                    if args.exp_name == 'recognition':
                        framePath = img_path[:-4] + f'_{i}' + img_path[-4:]
                        if os.path.basename(framePath) in file_utils.read_gtlabels(args.excludePath):
                            #removing bad view and complex signs
                            # print(os.path.basename(framePath))
                            continue
                        else:
                            _pred = [pr['voted response'] for pr in predJson if os.path.basename(pr['frame_path']) == os.path.basename(framePath)]
                            pred = ast.literal_eval(_pred[0])
                            mixed = ann['mixed'] if "mixed" in ann else {}
                            txtgt = process_raw_dict(ann['text labels'], mixed)
                            symgt = process_raw_dict(ann['symbol labels'], mixed)
                            txtpred = process_raw_dict(pred['T'])
                            sympred = process_raw_dict(pred['S'])

                    elif args.exp_name == 'full-pipeline':
                        #removing complex signs
                        if os.path.basename(entry['imagePath']) in file_utils.read_gtlabels(args.excludePath):
                            # print(os.path.basename(entry['imagePath']))
                            continue
                        else:
                            pred = ann['voted response']
                            mixed = ann['gt response']['mixed'] if "mixed" in ann['gt response'] else {}
                            txtgt = process_raw_dict(ann['gt response']['text labels'], mixed)
                            symgt = process_raw_dict(ann['gt response']['symbol labels'], mixed)
                            txtpred = process_raw_dict(pred.get('T',{}))
                            sympred = process_raw_dict(pred.get('S',{}))

                    txt_loc_mapping = {}
                    sym_loc_mapping = {}
                    total_sign_count += 1

                    #LOCATION NAME MATCHING     
                    if args.loc in ["hard","soft"] and args.sym in ["hard","soft-glove", "soft-clip", "soft-word2vec", "soft-bert"]:
                    #match_exact_text_tags -- symbols soft assignment
                        predtxt_copy = list(txtpred.keys())
                        predsym_copy = list(sympred.keys())
                        gttxt_copy = list(txtgt.keys())
                        gtsym_copy = list(symgt.keys())

                        #mandatory hard matching step
                        for loc in txtgt.keys():
                            if loc in txtpred.keys():
                                txt_loc_mapping[loc] = loc
                                predtxt_copy.remove(loc)
                                gttxt_copy.remove(loc)
                
                        if args.loc == 'soft':
                            #create a similariy matrix of remaining keys
                            txt_loc_mapping = hungarian_local_implementation(metrics.lexical_similarity, gttxt_copy, predtxt_copy, txt_loc_mapping)
                            

                        #mandatory hard matching step
                        for loc in symgt.keys():
                            if loc in sympred.keys():
                                sym_loc_mapping[loc] = loc
                                predsym_copy.remove(loc)
                                gtsym_copy.remove(loc)
                         
                        if args.sym == "soft-clip":
                            sym_loc_mapping = hungarian_local_implementation(metrics.clip_similarity, gtsym_copy, predsym_copy, sym_loc_mapping)
                        
                        txt_match_score, sym_match_score = give_match_score(txt_loc_mapping, sym_loc_mapping, txtpred, txtgt, sympred, symgt)
                        
                        common_match_score = 0
                        pressScore = 0
                        for com_key in mixed:
                            comScoreTup = []
                            presScoreTup = []
                            if com_key in sym_loc_mapping:
                                presScoreTup.append(1)
                                if type(symgt[com_key]) == type(sympred[sym_loc_mapping[com_key]]) and symgt[com_key] == sympred[sym_loc_mapping[com_key]]:
                                    comScoreTup.append(1)
                                else:
                                    comScoreTup.append(0)
                            
                            if com_key in txt_loc_mapping:
                                presScoreTup.append(1)
                                if type(txtgt[com_key]) == type(txtpred[txt_loc_mapping[com_key]]) and txtgt[com_key] == txtpred[txt_loc_mapping[com_key]]:
                                    comScoreTup.append(1)
                                else:
                                    comScoreTup.append(0)

                            if sum(comScoreTup) in [0,1]: common_match_score += 0
                            if sum(comScoreTup) == 2: common_match_score += 1
                            if sum(presScoreTup) == 2: pressScore += 1

                        if len(txtpred.keys()):
                            txt_indiv_precision +=  (txt_match_score/len(txtpred.keys()))
                        
                        if len(txtgt.keys()):
                            txt_indiv_recall += (txt_match_score/len(txtgt.keys()))

                        if len(sympred.keys()):
                            sym_indiv_precision +=  (sym_match_score/len(sympred.keys()))

                        if len(symgt.keys()):
                            sym_indiv_recall += (sym_match_score/len(symgt.keys()))
                        
                        if txt_match_score == len(txtgt.keys()): txt_global_perf_sign += 1
                        if sym_match_score == len(symgt.keys()): sym_global_perf_sign += 1
                        if txt_match_score == len(txtgt.keys()) and sym_match_score == len(symgt.keys()): 
                            global_perf_sign += 1
                        elif sym_match_score + txt_match_score - common_match_score == len(symgt.keys()) + len(txtgt.keys()) - len(mixed):
                            global_perf_sign += 1 

                        txt_global_match_score += txt_match_score #hard match
                        sym_global_match_score += sym_match_score #hard match
                        overall_global_match_score += txt_match_score + sym_match_score - common_match_score

                    txt_global_P_score += len(txtpred.keys())
                    txt_global_G_score += len(txtgt.keys())
                    
                    sym_global_P_score += len(sympred.keys())
                    sym_global_G_score += len(symgt.keys())
                    tempList = []
                    tempList.extend(list(txtpred.keys()))
                    tempList.extend(list(sympred.keys()))
                    overall_global_P_score += len(txtpred.keys()) + len(sympred.keys()) - pressScore
                    overall_global_G_score += len(symgt.keys()) + len(txtgt.keys()) - len(mixed)

                    mapping_history.append(txt_loc_mapping)
                    mapping_history.append(sym_loc_mapping)

        txt_recall = txt_global_match_score/txt_global_G_score
        sym_recall = sym_global_match_score/sym_global_G_score
        txt_precision = txt_global_match_score/txt_global_P_score
        sym_precision = sym_global_match_score/sym_global_P_score
        global_precision = overall_global_match_score/ overall_global_P_score
        global_recall = overall_global_match_score/overall_global_G_score
        
        txt_sign_accuracy = txt_global_perf_sign/total_sign_count
        sym_sign_accuracy = sym_global_perf_sign/total_sign_count
        overall_accuracy = global_perf_sign/total_sign_count
        
        # print(f'individual metrics for the dataset: \n txt_precision: {txt_indiv_precision/total_sign_count} \n sym_precision: {sym_indiv_precision/total_sign_count} \n txt_recall: {txt_indiv_recall/total_sign_count} \n sym_recall: {sym_indiv_recall/total_sign_count} \n txt_sign_accuracy: {txt_sign_accuracy} \n sym_sign_accuracy: {sym_sign_accuracy} \n overall accuracy: {overall_accuracy}')
        return txt_precision, txt_recall, txt_sign_accuracy, sym_precision, sym_recall, sym_sign_accuracy, overall_accuracy, global_precision, global_recall
                      
def parse_arguments():
    parser = argparse.ArgumentParser(description="arguments for evaluations")
    parser.add_argument('--root', type=str, help='/path/to/Sign-Understanding')
    return parser.parse_args()

if __name__ == "__main__":
    np.random.seed(42)
    args = parse_arguments()
    while True:
        try:
            r = input("Recognition or Full-Pipeline Evaluation? R/F")
            if r.upper() == 'R':
                args.config_file = os.path.join(args.root, 'config/recognition_eval_config.yaml') 
                config = file_utils.load_yaml(args.config_file)
                args.exp_name = config['name'] #'recognition'
                args.op_dir = os.path.join(args.root,config['exp']['result_dir'], 'gemini')
                args.dataset_dir = os.path.join(args.root,config['exp']['dataset_dir']) #you need to run the detection script to generate this dataset
                args.gt_path = os.path.join(args.root,config['exp']['groundtruth']) 
                args.excludePath = os.path.join(args.root,config['exp']['excludePath'])

                file_utils.makeCheck(args.op_dir)
                assert os.path.exists(args.dataset_dir), "please load the dataset in the correct directory"
                assert os.path.exists(args.gt_path), "please load the gt_annotation in the correct directory"
                
            elif r.upper() == 'F':
                while True:
                    try:
                        rec = input("You wish to generate per sign precision or per sign recall? P/R")
                        if rec.upper() == 'P':
                            args.recall = False
                        elif rec.upper() == 'R':
                            args.recall = True
                        break
                    except:
                        print('Please enter valid response... \n')

                args.config_file = os.path.join(args.root, 'config/full_pipeline_eval_config.yaml') 
                config = file_utils.load_yaml(args.config_file)
                args.exp_name = config['name'] #'full-pipeline'
                args.op_dir = os.path.join(args.root,config['exp']['result_dir'], 'g-dino-gemini', 'processed') 
                args.dataset_dir = os.path.join(args.root,config['exp']['dataset_dir']) 
                args.matchresp_path = os.path.join(args.root,config['exp']['fpmatchResponse']) # it should be the path of the matched-response of full-pipeline
                args.recg_gt = os.path.join(args.root,config['exp']['recg_groundtruth'])
                args.excludePath = os.path.join(args.root,config['exp']['excludePath'])

                file_utils.makeCheck(args.op_dir)
                assert os.path.exists(args.dataset_dir), "please load the dataset in the correct directory"
                assert os.path.exists(args.recg_gt), "please load the recg gt in the correct directory"
                assert os.path.exists(args.matchresp_path), "please load the fullpipeline response in the correct directory"
                
            break
        except Exception as e:
            print('Please enter valid response....')
                   
    loc_args = ['hard','soft']
    symb_args = ['soft-clip']
    results = []
    if args.exp_name == 'full-pipeline':
        #combines the response to a format which consists of all readable signs and the prrediction by baseline (None added if no prediction by baseline)
        args.gt_path = reformat_full_pipeline_results(args.matchresp_path, args.recg_gt)
    
    for f in os.listdir(args.op_dir):
        if f[-5:] == '.json':
            for a1, a2 in itertools.product(loc_args, symb_args):
                args.loc = a1
                args.sym = a2
                args.f = f
                tp, tr, tacc, sp, sr, sacc, acc, _,_ = process_jsons_for_predictions(args)
                if args.exp_name == 'recognition':
                    results.append([a1,a2,tp,tr,tacc, sp, sr, sacc, acc])
                elif args.exp_name == 'full-pipeline' and args.loc == 'soft':
                    results.append([a1,a2,acc])
                
                print('--------------------------------------------------------')
            if args.exp_name == 'recognition':
                df = pd.DataFrame(results, columns=["loc", "symb", "txt_precision", "txt_recall","txt_sign-accuracy", "sym_precision", "sym_recall","sym_sign_accuracy", "total_acc"])
            elif args.exp_name == 'full-pipeline' and args.loc == 'soft':
                df = pd.DataFrame(results, columns=["loc", "symb", "per_sign"])
            print(df)
            
