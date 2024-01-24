import json

with open ("/scratch/user/hasnat.md.abdullah/AnomalyWatchdog/data/oops_dataset/annotations/heldout_transition_times.json","r") as f: 
    x = json.load(f)

with open("./heldout_transition_times_fixed.json", "w") as f : 
    json.dump(x,f,indent =4)