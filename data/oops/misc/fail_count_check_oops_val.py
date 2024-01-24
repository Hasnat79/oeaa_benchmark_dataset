import json 
import os
from pathlib import Path

def load(filename):
    return json.load(open(filename, 'r'))

def save(filename, data):
    return json.dump(data, open(filename, 'w'), indent =4)


val_filtered_paths = "annotations/val_filtered.txt"
transition_time_fixed = load("annotations/transition_times_fixed.json")
heldout_transition_time_fixed  = load("annotations/heldout_transition_times_fixed.json")

count = 0
failure_count = 0
normal_count = 0


with Path(val_filtered_paths).open() as f: 
    for line in f: 
        # print(line.rstrip("\n"))
        if heldout_transition_time_fixed[line.rstrip('\n')]["n_notfound"]==0:
            failure_count +=1
        elif heldout_transition_time_fixed[line.rstrip('\n')]["n_notfound"]==1:
            normal_count +=1

        count += 1

print(f"val_filtered_paths count: {count}")
print(f"failure_count: {failure_count}")
print(f"normal_count: {normal_count}")

print(f"""failure percentage : %.2f%%
normal percentage : %.2f%%""" % ((failure_count*100)/count, (normal_count*100)/count))

print(f"len(heldout_transition_times_fixed): {len(heldout_transition_time_fixed.keys())}")
# val_filtered_paths count: 4711
    # failure_count: 3557
    # normal_count: 1154
    # failure percentage : 75.50%
    # normal percentage : 24.50%
# len(transition_times_fixed): 10961