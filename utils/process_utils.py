from collections import Counter

def most_common_keys(dict_list, thresh = 3, n=None):
    """
    Find the most common keys across multiple dictionaries
    
    Parameters:
    dict_list (list): List of dictionaries to analyze
    n (int, optional): Number of most common keys to return. If None, returns all keys in order.
    
    Returns:
    list: List of the most common keys in order of frequency
    """
    # Flatten all keys from all dictionaries
    all_keys_t = []
    all_keys_s = []
    
    for d in dict_list:
        if 's' not in d:
            d['s'] = {}
        if 't' not in d:
            d['t'] = {}
            
        all_keys_t.extend([k for k in d['t'].keys() if not isinstance(d['t'][k], dict)]) 
        all_keys_s.extend([k for k in d['s'].keys() if not isinstance(d['s'][k], dict)])
    all_keys_t_2 = [ak.replace(' ','').upper() for ak in all_keys_t]
    all_keys_s_2 = [ak.replace(' ','').upper() for ak in all_keys_s]
    
    # Count occurrences of each key
    key_counts_t = Counter(all_keys_t_2)
    key_counts_s = Counter(all_keys_s_2)
    
    # Get most common keys
    most_common_t = key_counts_t.most_common(n)
    most_common_s = key_counts_s.most_common(n)
    
    # Return just the keys (without counts)
    temp_keys_s = [key for key, count in most_common_s if count >= thresh]
    temp_keys_t = [key for key, count in most_common_t if count >= thresh]
    return {'S': temp_keys_s , 'T': temp_keys_t}

def get_most_common_directions(dict_final_keys, ans):
    #final keys should be a list of the keys you have selected through majority voting
    result = {'S': {}, 'T': {}}    
    ans_2 = [{k.replace(' ','').upper(): {vk.upper().replace(' ',''): vv.upper().replace(' ','') if isinstance(vv, str) else [vi.upper().replace(' ','') for vi in vv] for vk,vv in v.items()}} for ans_i in ans for k,v in ans_i.items()]
    
    for tag in ['S','T']:
        for key in dict_final_keys[tag]:  
            try:
                values_dir = []
                for d in ans_2:
                    try:
                        if isinstance(d[tag][key], str):
                            values_dir.append(d[tag][key].upper().replace(' ',''))
                        elif isinstance(d[tag][key], list):
                            values_dir.append(tuple(sorted([d[tag][key][p].upper().replace(' ','') for p in range(len(d[tag][key]))])))
                    except:
                        continue
                most_common_value_dir = Counter(values_dir).most_common(1)[0][0]
                result[tag][key] = most_common_value_dir
                
            except Exception as e:
                print(f'During calculating most popular directions. Encountered : {e}')
                continue
    return result

