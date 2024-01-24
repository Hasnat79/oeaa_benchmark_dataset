import json 

def load(filename):
    return json.load(open(filename, 'r'))

def save(filename, data):
    return json.dump(data, open(filename, 'w'), indent =4)


val_cap = load("oops_val_cap.json")
val = load("val_format_fixed.json")

assert len(val_cap) == len(val), "both files must be same sizel"

print(len(val_cap))
print(len(val))