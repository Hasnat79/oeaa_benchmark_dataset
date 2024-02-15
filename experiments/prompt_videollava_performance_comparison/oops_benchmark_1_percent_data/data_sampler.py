from constants import OOPS_BENCHMARK
import json
import random


def load_json(filename):
    return json.load(open(filename))
def write_json(filename, data):
    return json.dump(data, open(filename,'w'),indent=4)

if __name__ == '__main__':
    oops_benchmark = load_json(OOPS_BENCHMARK)

    # randomly sample 40 videos, 30 will be failure = 1, 10 will be failure = 0
    random.seed(0)
    video_ids = list(oops_benchmark.keys())
    random.shuffle(video_ids)
    failure_count = 0
    normal_count = 0

    sampled_data = {}
    
    for video_id in video_ids:
        if oops_benchmark[video_id]['failure'] == 1 and failure_count < 30:
            sampled_data[video_id] = oops_benchmark[video_id]
            failure_count += 1
        elif oops_benchmark[video_id]['failure'] == 0 and normal_count < 10:
            sampled_data[video_id] = oops_benchmark[video_id]
            normal_count += 1
        if failure_count == 30 and normal_count == 10:
            break
    write_json("sampled_oops_benchmark.json", sampled_data)

    #check the distribution of failure and normal videos in sampled data
    print("Sampled data distribution: ", {k: len([v for v in sampled_data.values() if v['failure'] == k]) for k in [0, 1]})
    # Sampled data distribution:  {0: 10, 1: 30}
