import os
import pdb
import json
import glob
from collections import Counter
import time
import datetime
import numpy as np
import copy
import ffmpeg
import random
from dataset_reader import DatasetReader
import sys
import av
import math
from concurrent.futures import ProcessPoolExecutor

if sys.platform == 'darwin':
    sys_prefix = '/Users/yhuang/video_trim/data/'
    num_cpus = 8
else:
    sys_prefix = 'data/'
    num_cpus = 24

def print_and_run(cmd):
    print(cmd)
    os.system(cmd)

executor = ProcessPoolExecutor(max_workers=num_cpus)

#generate end clips:
def generate_visualization(source, r):
    names = r.get_ids()
    #pdb.set_trace()
    prefix = r.vis_prefix
    os.makedirs(prefix, exist_ok=True)
    for name in names[:50]:
        config = r.read_id(name)
        video_path = config['video']
        frames = config['frames']
        start_times = config['start_times']
        duration = config['duration']

        if len(frames) == 0:
            print(f'{name} contains 0 cuts')
            continue

        #pdb.set_trace()
        # tuple of time, frame
        last_time = 0
        start_times.append(duration)

        end_times = [0] + start_times
        neg_times = []
        for idx, start_time in enumerate(end_times[:-1]):
            true_start_time = start_time + 0.1
            true_end_time = end_times[idx+1] - 0.1

            duration = true_end_time - true_start_time
            # clip duration: 2. Random range: 1-3
            if duration > 5:
                neg_start_time = start_time + random.uniform(1, 3)
                neg_end_time = end_times[idx+1] - random.uniform(1, 3)

                #true_times.append(true_start_time, true_end_time)
                neg_times.append([true_start_time, true_end_time, neg_start_time, neg_end_time])

        random.shuffle(neg_times)
        #
        #for idx, pair in enumerate(neg_times[:10]):
        if True and len(neg_times) > 0:
            idx = 0
            pair = neg_times[0]
            true_start_time, true_end_time, neg_start_time, neg_end_time

            cmd = f'ffmpeg -y -threads 1 -ss {true_start_time} -t 2 -i {video_path} -threads 1 -vcodec libx264 -an clips_start/{source}_{name}_{idx}_A.mp4'
            executor.submit(print_and_run, cmd)

            cmd = f'ffmpeg -y -threads 1 -ss {neg_start_time} -t 2 -i {video_path} -threads 1 -vcodec libx264 -an clips_start/{source}_{name}_{idx}_B.mp4'
            executor.submit(print_and_run, cmd)

            cmd = f'ffmpeg -y -threads 1 -ss {true_end_time - 2} -t 2 -i {video_path} -threads 1 -vcodec libx264 -an clips_end/{source}_{name}_{idx}_A.mp4'
            executor.submit(print_and_run, cmd)

            cmd = f'ffmpeg -y -threads 1 -ss {neg_end_time - 2} -t 2 -i {video_path} -threads 1 -vcodec libx264 -an clips_end/{source}_{name}_{idx}_B.mp4'
            executor.submit(print_and_run, cmd)

if __name__ == "__main__":
    vimeo90k1 = DatasetReader.create('vimeo90k', sys_prefix + 'vimeo90k', shotcut_path='/shotcut/', vis_prefix='cache_exa')
    generate_visualization('vimeo90k', vimeo90k1)

    #youtube1 = DatasetReader.create('youtube', sys_prefix + 'youtube', shotcut_path='/shotcut/', vis_prefix='cache_exa2')
    #generate_visualization('youtube', youtube1)

    #roughcut1 = DatasetReader.create('roughcut', sys_prefix + 'roughcut', shotcut_path='/shotcut/', vis_prefix='cache_exa')
    #generate_visualization('roughcut', roughcut1)

    print('All task has generated')
    executor.shutdown()
    print('All done')
