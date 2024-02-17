# oeaa_benchmark_dataset
This repo comprises the development codes for a benchmark dataset for open ended activity analytics.

## capture key frame technique development
- go to `videollava/videollava/model/multimodal_encoder/languagebind/video/processing_video.py`
- from line `89-96`, this portion of code works form sampling the frames from a video
    - currently sampling uniformly
        - ```bash 
            frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)```
    - Environment setup: 
        - ```bash 
            conda env create -f videollava_env.yml``` 
    - Objective
    - We have to change the sampling technique 
    - Sampling idea: 
        - reference paper [ZeroCap](https://arxiv.org/abs/2111.14447)
    - Rough idea but not limited to
        - sample 8 frames
        - take each frame, calculate cosine sim with the next frame
            - if they are similar (apx. 90%+), skip that frame
            - Otherwise, sample that frame id/index
            - if 8 frames (currently default frame size) sampled, end the sampling process
        

