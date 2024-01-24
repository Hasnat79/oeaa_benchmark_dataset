import json 

def load(filename):
    return json.load(open(filename, 'r'))

def save(filename, data):
    return json.dump(data, open(filename, 'w'), indent =4)

val = load("val.json")

save("val_format_fixed.json", val)