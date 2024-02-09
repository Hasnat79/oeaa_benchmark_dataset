import os
import json
from pprint import pprint
# from tqdm import tqdm
from pytube import YouTube





url_list = ['http://www.youtube.com/watch?v=I7fdv1q9-m8',
'https://www.youtube.com/watch?v=hKf-IwHM6TI',
'https://www.youtube.com/watch?v=pzPhXbGEpSo',
'https://www.youtube.com/watch?v=sAgAvYT3D8s',
'https://www.youtube.com/watch?v=JM8BHjJTSFM'
]		


try:
    for url in url_list:
        # Create a YouTube object with the provided URL
        yt = YouTube(url)
        
        # Filter the available video streams to get the highest resolution
        video = yt.streams.get_highest_resolution()
        
        # Download the video to the specified output path, rename the video as the key
        video.download(url.split('=')[-1] + '.mp4')
        
        print("Video downloaded successfully!")
    
    

except Exception as e:
    print("error:",e)

