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












if __name__ == "__main__":
    #oops val-set list
    val_filtered_paths = "../annotations/val_filtered.txt"
    val_videos = txt_to_lst(val_filtered_paths)

    #oops val-set annotation file
    heldout_transition_time_fixed = load("../annotations/heldout_transition_times_fixed.json")
    transition_time_fixed = load("../annotations/transition_times_fixed.json")
    # /scratch/user/hasnat.md.abdullah/oeaa_benchmark_dataset/data/oops/cap/svos/val_format_fixed.json
    val_desc = load("../cap/svos/val_format_fixed.json")

    #--------------------------------------------
    failure_count = 0
    normal_count = 0
    oops_val = {}
    key_err =0
    for video in val_videos: 
        data = {}
        data['t'] = heldout_transition_time_fixed[video]['t']
        data['duration'] = transition_time_fixed[video]["len"]
        data['path'] =f"{video}.mp4"

        try: 
            if heldout_transition_time_fixed[video]["n_notfound"] == 0:
                #failure video
                failure_count+=1
                data['failure'] = 1
                

                data["wentwrong_1"] = val_desc[video][0]["wentwrong"]
                data["goal_1"] = val_desc[video][0]["goal"]
                data["wentwrong_2"] = val_desc[video][1]["wentwrong"]
                data["goal_2"] = val_desc[video][1]["goal"]

            else: 
                data['failure'] = 0
                normal_count +=1
        except Exception as e: 
            print("video is not available in description set")
            key_err+=1
        oops_val[video] = data

    save("../oops_benchmark_val.json",oops_val)
    print(len(oops_val))
    print(f"valset: fail_count {failure_count}")
    print(f"valset: normal_count {normal_count}")
    print(f"Fail videos n/a in description set: {key_err}")

    # #-of val-set failed vids that has captions -> 3268
    count_keys_without_goal_1 = sum(1 for key, value in oops_val.items() if "goal_1" not in value)#3268

    print(f"#-of val-set failed vids that has captions: {count_keys_without_goal_1}")




