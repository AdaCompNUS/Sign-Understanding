import torch
import open_clip

clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

def lexical_similarity(s1, s2_list):
    """sub-string check"""
    res = []
    all_jacc_score = jaccard_similarity(s1, s2_list)
    s1_norm = s1.lower()
    for s2 in s2_list:
        s2_norm = s2.lower()
        if s1_norm in s2_norm:
            res.append(1)
        else:
            res.append(0)
    # final_score = [max(all_jacc_score[i], res[i]) for i in range(len(s2_list))]
    final_score = [all_jacc_score[i]*res[i] for i in range(len(s2_list))]
    return final_score

def clip_similarity(word, word_list):
    f_word = f"symbol of {word}"
    f_word_list = [f"symbol of {w}" for w in word_list]
    tokens = clip_tokenizer([f_word] + f_word_list)
    with torch.no_grad():
        embeddings = clip_model.encode_text(tokens).float()
    sims = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1:], dim=-1)
    return sims.tolist()
    
def jaccard_similarity(s1, s2_list):
    """Jaccard similarity between two token sets"""
    s1_norm = s1.lower()
    all_jacc_score = []
    for s2 in s2_list:
        s2_norm = s2.lower()    
        set1, set2 = set(s1_norm), set(s2_norm)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jacc_score = intersection / union if union > 0 else 0.0
        all_jacc_score.append(jacc_score)
    return all_jacc_score
    