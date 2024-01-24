
- oops dataset dir path: ../AnomalyWatchdog/data/oops_dataset
- oops val video dir path: ../AnomalyWatchdog/data/oops_dataset/oops_video/val


- oops validation caption data size: 3593(videos: all failure videos) 
- oops val - set (validation_filtered.txt) size: 4711 
    - val-set missing captions: 1157/4711
    - val-set has captions: 3554/4711
        - failure count: 3268 (92%)
        - normal count: 286 (8%) -- according to human-eval they are normal !!!


There are 3 annotators in the oops-val set
Definition 1: 
    - If all of the 3 annotators finds the video to be normal --> normal
    - If any of the 3 annotators finds the video to be unusual --> unusual
Distribution: 
    - failure percentage : 85.33%
    - normal percentage : 14.67%


Definition 2:
    - if all of the 3 annotators finds the video to be unusual --> unusual
    - if any of the 3 annotators find the video to be normal --> normal
Distribution:
    - failure percentage : 38.31%
    - normal percentage : 61.69%


Human Evaluation: According to paper (heldout_transition_times_fixed.json)
    - A fourth person evaluates the 3 annotators 
Distribution: 
    - failure percentage : 75.50%
    - normal percentage : 24.50%

