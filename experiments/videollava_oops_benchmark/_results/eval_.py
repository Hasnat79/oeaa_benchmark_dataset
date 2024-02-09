

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
    
     
    print(f"tp: {tp}")
    print(f"tn: {tn}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")
    # calculate accuracy
    acc = (tp+tn)/(tp+tn+fp+fn)
    print(f"acc: {round(acc,2)}")
    # calculate precision, recall and f1 score
    precision, recall, f1_score, support = precision_recall_fscore_support(ground_truth, predictions, average='binary')
    auc_roc_curve = roc_auc_score(ground_truth, predictions)
    print(f"auc_roc_curve: {auc_roc_curve}")
    saves_auc_roc_curve_plot(ground_truth,predictions,auc_roc_curve, "./auc_roc_")

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


merged_res = files_merger("./")
# merged_res = load_json("../p2_merged_results.json")
print(len(merged_res)) #4170
kk ={}
c=0
for k,v in merged_res.items():
    if "Yes. The" in v['videollava_generation_1'] or "Yes.</s>" in v['videollava_generation_1']:
        v['pred_failure'] = 1
    elif "No. The" in v['videollava_generation_1'] or "No.</s>" in v['videollava_generation_1']:
        v['pred_failure'] = 0
    else: 
        # kk.append(k)
        kk[k]= v['videollava_generation_1']

        # print(v['videollva_generation_1'])
        c+=1

write_json("./need_manual_labeling/need_manual_labeling_files.json",kk)
for k in kk:
    del merged_res[k]

print(f"yes/no missing instances count: {c}") #400
total_failure = sum(1 for entry_id, entry in merged_res.items() if entry['failure'] == 1 )
print(f"failure: {total_failure}/{len(merged_res)}")
print(f"normal: {len(merged_res)-total_failure}")
# filtered_res = filter_results(merged_res) # total videos after filtering: 4317
auc_roc_curve(merged_res)
calculate_precision_recall_f1(merged_res)

#cider
cider_avg = 0
aa = sum([1 for k,v in merged_res.items() if 'goal_1' in v])
print(aa) #2892 faulure instances with caption
for k,v in merged_res.items():
    if 'goal_1' in v:
        reference = (v['goal_1']+v['wentwrong_1']).split()
        # candidate_tokens = v['videollava_generation_1'].split()
        
        # only narration is used to calculate    
        candidate_tokens = v['videollava_generation_1'].split("Answer with Explanation: ")[0].split()
        
        cider_avg +=cal_cider_score(reference, candidate_tokens)


cider_avg = cider_avg/aa
print(f"avg cider score: {cider_avg}")


# 4170
# yes/no missing instances count: 400
# total keys: 3770
# tp: 2798
# tn: 14
# fp: 936
# fn: 22
# acc: 0.75
# auc_roc_curve: 0.5034677118327734
# precision: 0.75, recall: 0.99, f1_score: 0.85, support: None