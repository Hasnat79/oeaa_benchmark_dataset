import json 
import os
from pathlib import Path
import random

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





if __name__ == "__main__":
    #oops val-set list
    val_filtered_paths = "../../data/oops/annotations/val_filtered.txt"
    val_videos = txt_to_lst(val_filtered_paths)
    #oops val-set annotation file
    heldout_transition_time_fixed = load("../../data/oops/annotations/heldout_transition_times_fixed.json")
    ablate_data = {}

    failure = 0
    normal = 0
    val_videos = random.sample(val_videos, len(val_videos))
    for val_data in val_videos:

        if failure <5 and heldout_transition_time_fixed[val_data]["n_notfound"]== 0:
            failure+=1
            ablate_data[val_data] = {"failure" : 1}
        
        if normal < 5 and heldout_transition_time_fixed[val_data]["n_notfound"] == 1: 
            normal += 1
            ablate_data[val_data] ={"failure" : 0} 
        # print(val_data)

    save("ablate_data.json",ablate_data)
