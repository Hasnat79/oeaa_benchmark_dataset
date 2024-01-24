import json 
import os
from pathlib import Path

def load(filename):
    return json.load(open(filename, 'r'))

def save(filename, data):
    return json.dump(data, open(filename, 'w'), indent =4)

def txt_to_lst(txt):
    import os
    from pathlib import Path
    lst = []
    with Path(txt).open() as f: 
        for line in f: 
            lst.append(line.strip('\n'))
    return lst
#oops val-set list
val_filtered_paths = "annotations/val_filtered.txt"
val_videos = txt_to_lst(val_filtered_paths)
#oops val-set annotation file
heldout_transition_time_fixed = load("annotations/heldout_transition_times_fixed.json")

# oops val set description
val_desc_data = load("cap/val_format_fixed.json")
print(f"val desc data file count: {len(val_desc_data.keys())}")
#tracks how many files of original oops-val set
# matches with val-set description file
failure_count =0
normal_count = 0
match_count =0
count =0
missing = 0

#checking missing val videos: val-caps vs val-set

for video in val_videos: 
    if video not in val_desc_data.keys(): 
        missing += 1

print(f"Val videos missing caps: {missing}")




missing_cap_failure_count = 0
missing_cap_normal_count = 0
# iterate through validatin set oops file list
with Path(val_filtered_paths).open() as f: 
    for line in f: 
        # print(line.rstrip("\n"))

        #checks if the file name exists in val-set description file
        if line.rstrip("\n") in list(val_desc_data.keys()):
            match_count+=1

            if heldout_transition_time_fixed[line.rstrip('\n')]["n_notfound"] ==0:
                failure_count +=1
                # print("ok")
                
            else: 
                normal_count +=1
                # print(line.rstrip('\n'))
        else:

            
            if heldout_transition_time_fixed[line.rstrip('\n')]["n_notfound"] ==0:
                missing_cap_failure_count +=1
                # print("ok")
                print(line.rstrip('\n'))
                break
                
            elif heldout_transition_time_fixed[line.rstrip('\n')]["n_notfound"] == 1:
                missing_cap_normal_count +=1
                # break
                
            

        count += 1

print(f"val-set vs val-set-desc file matches: {match_count}")
print(f"failure count: {failure_count}")
print(f"normal count: {normal_count}")
print(f"""failure percentage : %.2f%%
normal percentage : %.2f%%""" % ((failure_count*100)/match_count, (normal_count*100)/match_count))
print(f"total val-set file_count: {count}")
print(f"Val-videos missing captions: {count-match_count}")
print(f"Val-videos missing captions failure count: {missing_cap_failure_count}")
print(f"Val-videos missing captions normal count: {missing_cap_normal_count}")

# val desc data file count: 3593
# Val videos missing caps: 1157
# val-set vs val-set-desc file matches: 3554
# failure count: 3268
# normal count: 286
# failure percentage : 91.95%
# normal percentage : 8.05%
# total val-set file_count: 4711
# Val-videos missing captions: 1157
# Val-videos missing captions failure count: 289
# Val-videos missing captions normal count: 868