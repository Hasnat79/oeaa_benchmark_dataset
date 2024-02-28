

import json
import os
from sklearn.metrics import precision_recall_fscore_support
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from result_text_generator import write_metrics
# openai.api_key = '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'
def auc_roc_curve(filtered_res):
    # saves a auc_roc_curve plot in the current working folder
    # each data dict has a key called 'pred_failure'
    ground_truth = [value['failure'] for value in filtered_res.values() ]
    predictions = [value['pred_failure'] for value in filtered_res.values() ]
    
    print(f"total keys: {len(ground_truth)}")
    tp = sum(1 for entry_id, entry in filtered_res.items() if entry['pred_failure'] == 1 and entry['failure'] == 1 )
    tn = sum(1 for entry_id, entry in filtered_res.items () if entry['pred_failure'] == 0 and entry['failure'] == 0 )
    fp = sum(1 for entry_id, entry in filtered_res.items () if entry['pred_failure'] == 1 and entry['failure'] == 0 )
    fn = sum(1 for entry_id, entry in filtered_res.items () if entry['pred_failure'] == 0 and entry['failure'] == 1 )
    
    global metrics
    metrics['tp'].append(tp)
    metrics['tn'].append(tn)
    metrics['fp'].append(fp)
    metrics['fn'].append(fn)
    # print(f"tp: {tp}")
    # print(f"tn: {tn}")
    # print(f"fp: {fp}")
    # print(f"fn: {fn}")
    # calculate accuracy
    acc = (tp+tn)/(tp+tn+fp+fn)
    metrics['acc'].append(acc)
    # print(f"acc: {round(acc,2)}")
    # calculate precision, recall and f1 score
    precision, recall, f1_score, support = precision_recall_fscore_support(ground_truth, predictions, average='binary')
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1_score)

    auc_roc_curve = roc_auc_score(ground_truth, predictions)
    metrics['auc_roc_curve'].append(auc_roc_curve)
    # print(f"auc_roc_curve: {auc_roc_curve}")
    saves_auc_roc_curve_plot(ground_truth,predictions,auc_roc_curve, "./p0_auc_roc_")

def saves_auc_roc_curve_plot(ground_truth,predictions,auc, file_name):
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions, pos_label=1)
    # saves a auc_roc_curve plot in the current working folder
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(file_name + ".png")
    plt.close()
def files_merger(dir_name):
    # takes a directory name and merger all json files in that directory according to the number_ of file names
    files = os.listdir(dir_name)
    files = [file for file in files if file.endswith(".json")]
    files.sort(key=lambda x: int(x.split(".")[0].split("_")[0])) #needs to be changed according to the file name
    # print(files)
    merged_results = {}
    for file in files:
        with open(dir_name + "/" + file, "r") as f:
            data = json.load(f)
        # with open(dir_name + "/" + file.split(".")[0] + ".json", "w") as f:
        #     json.dump(data, f, indent=4)
        # os.remove(dir_name + "/" + file)

        #updating the merged_results dictionary with data
        for key, value in data.items():
            merged_results[key] = value

    return merged_results
     
def write_json(output_file_name,data): # "./merged_results.json"
    with open(output_file_name,'w') as f:
        json.dump(data,f,indent =4)
def load_json(file):
    with open(file,'r') as f: 
        return json.load(f)
def filter_results(merged_res_dict):
    # each data dict has a key called 'pred_failure'
    # this function filters out the results dictionary that has a key called 'pred_failure': 'na'
    filtered_results = {}
    for key, value in merged_res_dict.items():
        if value['pred_failure']!= 'na':
            filtered_results[key] = value
    print(len(filtered_results))
    return filtered_results

def calculate_precision_recall_f1(filtered_res):
    # extract ground truth and predictions
    ground_truth = [value['failure'] for value in filtered_res.values()]
    predictions = [value['pred_failure'] for value in filtered_res.values()]
    # calculate precision, recall and f1 score
    precision, recall, f1_score, support = precision_recall_fscore_support(ground_truth, predictions, average='binary')
    # print(f"precision: {precision}, recall: {recall}, f1_score: {f1_score}, support: {support}")
    print(f"precision: {round(precision,2)}, recall: {round(recall,2)}, f1_score: {round(f1_score,2)}, support: {support}")
    return precision, recall, f1_score, support


def cal_cider_score(reference_tokens, candidate_tokens):

    # print('reference_tokens', reference_tokens)
    # print('candidate_tokens', candidate_tokens)

    # Convert tokens to string for TF-IDF vectorization
    generated_text = ' '.join(candidate_tokens)
    reference_texts = [' '.join(reference_tokens)]

    # print('generated_text', generated_text)
    # print('reference_texts', reference_texts)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([generated_text] + reference_texts)

    # print('type(tfidf_matrix)', type(tfidf_matrix))
    # print('tfidf_matrix:', tfidf_matrix)
    # print('tfidf_matrix[0]:', tfidf_matrix[0])
    # print('tfidf_matrix[1:]:', tfidf_matrix[1:])

    # Calculate cosine similarity
    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    # print('similarities', similarities)

    # Compute consensus score
    consensus_score = similarities.mean()
    # print('consensus_score', consensus_score)

    # Compute sentence length penalty
    generated_length = len(candidate_tokens)
    # reference_lengths = [len(ref) for ref in reference_tokens] # this applies when there are multiple references
    reference_lengths = len(reference_tokens) # this applies when there is only one reference
    # print('generated_length', generated_length)
    # print('reference_lengths', reference_lengths)

    length_penalty = max(0, 1 - abs(generated_length - np.mean(reference_lengths)) / np.mean(reference_lengths))
    # print('length_penalty', length_penalty)

    # Compute CIDEr score
    cider_score = consensus_score * length_penalty
    # print('cider_score', cider_score)
    # stop

    return cider_score

def cal_bert_score(reference, candidate, tokenizer, model):
    
    #encoding the reference and candidate
    reference_encoding = tokenizer(reference, return_tensors='pt', padding=True, truncation=True)
    candidate_encoding = tokenizer(candidate, return_tensors='pt', padding=True, truncation=True)

    #passing the reference and candidate through the model
    with torch.no_grad():
        reference_output = model(**reference_encoding)
        candidate_output = model(**candidate_encoding)
    
    # getting the hidden states from the model
    candidate_hidden_states = candidate_output.last_hidden_state.mean(dim=1).detach().numpy()
    reference_hidden_states = reference_output.last_hidden_state.mean(dim=1).detach().numpy()

    #similarity score
    similarity = np.dot(reference_hidden_states, candidate_hidden_states.T)/(np.linalg.norm(reference_hidden_states)*np.linalg.norm(candidate_hidden_states))

    return similarity[0][0]

def get_prediction(v, i):
    if " Yes, the" in v[f'videollava_gen{i}_llama2_generation'] :
        return 1
    elif "No, the" in v[f'videollava_gen{i}_llama2_generation'] :
        return 0
    else:
        return None
def generate_cider_scores (merged_res, i):
    cider_scores = []
    for k, v in tqdm(merged_res.items()):
        if 'goal_1' in v:
            reference = (v['goal_1']+v['wentwrong_1']).split()
            candidate_tokens = v[f'videollava_generation_{i}'].split("Answer with Explanation: ")[0].split()
            cider_scores.append(cal_cider_score(reference, candidate_tokens))
    return np.mean(cider_scores)


def calculate_bert_scores(merged_res,i,tokenizer,model):
    similarity_scores =[]
    for k,v in tqdm(merged_res.items()):
        if 'goal_1' in v:
            reference = [(v['goal_1']+v['wentwrong_1'])]
            candidate = [v[f'videollava_generation_{i}'].split("Answer with Explanation: ")[0]]
            similarity_scores.append(cal_bert_score(reference, candidate,tokenizer, model))
    return np.mean(similarity_scores)

if __name__ == "__main__":
    res_doc = ""
    
    metrics = {
        'missing_prediction': [],
        'total_failure': [],
        'total_normal': [],
        'tp': [],
        'tn': [],
        'fp': [],
        'fn': [],
        'acc': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'auc_roc_curve': [],
        'cider_score': [],
        'bert_score': [],
        'prompt':[]
    }


    for i in range(3,4): #three prompts
        file_name = f"./p{i}_videollava_x_oops_baseline_res.json"
        merged_res = load_json(file_name)
        metrics['prompt'].append(merged_res[list(merged_res.keys())[0]][f'videollava_prompt_{i}'])
        missing_predictions = {}
    
        for k,v in merged_res.items():
            prediction = get_prediction(v, i)
            if prediction is None:
                missing_predictions[k] = v[f'videollava_gen{i}_llama2_generation']
            else:
                v['pred_failure'] = prediction

        # write_json("./need_manual_labeling/need_manual_labeling_files.json",kk)
        for k in missing_predictions:
            del merged_res[k]

        metrics['missing_prediction'].append(len(missing_predictions))
        total_failure = sum(1 for entry_id, entry in merged_res.items() if entry['failure'] == 1 )
        metrics['total_failure'].append(total_failure)
        metrics['total_normal'].append(len(merged_res)-total_failure)

        auc_roc_curve(merged_res)
        calculate_precision_recall_f1(merged_res)

        #cider
        cider_scores= []
        # aa = sum([1 for k,v in merged_res.items() if 'goal_1' in v])
        # print(aa) #2892 faulure instances with caption
        for k,v in tqdm(merged_res.items()):
            if 'goal_1' in v:
                reference = (v['goal_1']+v['wentwrong_1']).split()
                # candidate_tokens = v['videollava_generation_1'].split()
                
                # only narration is used to calculate    
                # candidate_tokens = v[f'videollava_generation_{i}'].split("Answer with Explanation: ")[0].split()
                # print(v[f'videollava_gen{i}_llama2_generation'].split(".\n"))

                candidate_tokens =v[f'videollava_gen{i}_llama2_generation'].split()
                cider_scores.append(cal_cider_score(reference, candidate_tokens))


        # print("average cider score:", np.mean(cider_scores))
        metrics['cider_score'].append(generate_cider_scores(merged_res, i))

        ##bert_score
        # load pre-trained bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        similarity_scores =[]

        # print("Average bert score similarity:", np.mean(similarity_scores))
        metrics['bert_score'].append(calculate_bert_scores(merged_res,i,tokenizer,model))

    write_metrics(metrics)
    print(metrics)

