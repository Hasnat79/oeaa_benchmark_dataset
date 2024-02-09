import json

def load_json(filename):
    return json.load(open(filename))


need_man_label = load_json("need_manual_labeling_files.json")
print(len(need_man_label))
c=0
d=0
e=0
for k,v in need_man_label.items():
    if 'Yes.</s>' in v :
        c+=1
    elif 'No.</s>' in v:
        d+=1
    else: 
        e+=1

print(f"Only 'yes' and no explanation {c}")
print(f"Only 'no' and no explanation {d}")
print(f"yes/no missing instances: {e}")