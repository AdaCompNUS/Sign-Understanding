import yaml, json, os

def load_yaml(filepath):
    with open(f"{filepath}", "r") as file:
        data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues            
    return data

def save_file_json(file_path, data):
    with open(f'{file_path}', 'w') as g:
        json.dump(data, g, indent=4)

def read_gtlabels(filepath):
    '''only for decider method -- when to prompt vlm -- not actually sent in vlm prompt'''
    labels = []
    try:
        with open(f"{filepath}", "r") as file:
            for line in file:
                labels.append(line.strip()) 
    except Exception as e:
        print(e)
    return labels

def read_prompt(filepath):
    with open(f"{filepath}", 'r') as file:
        filedata = file.read()
    return filedata

def read_json(filepath):
    try:
        with open(filepath) as json_file:
            return json.load(json_file) 
    except Exception as e:
        print(e)

def makeCheck(fol_path):
    if not os.path.exists(fol_path):
        os.makedirs(fol_path)
