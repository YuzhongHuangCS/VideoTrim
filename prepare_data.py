import os
import pdb
import json
import glob
from jinja2 import Template
from collections import Counter
import time
import datetime
import numpy as np
import copy
from concurrent.futures import ProcessPoolExecutor
import torch
import torchvision
import ffmpeg

from dataset_reader import DatasetReader
import sys

if sys.platform == 'darwin':
    sys_prefix = '/Users/yhuang/video_trim/data/'
    num_cpus = 8
else:
    sys_prefix = '/home/code-base/data_space/'
    num_cpus = 4

def print_and_run(cmd):
    print(cmd)
    os.system(cmd)

executor = ProcessPoolExecutor(max_workers=num_cpus)

#generate end clips:
def generate_visualization(source, r):
    names = r.get_ids()
    pos_counter = 0
    neg_counter = 0
    pos_per_clip = 10
    neg_per_clip = 10

    pos_map = {}
    neg_map = {}
    for name in names:
        prefix = r.vis_prefix
        os.makedirs(prefix, exist_ok=True)

        config = r.read_id(name)
        video_path = config['video']
        start_times = config['start_times']
        duration = config['duration']

        if len(start_times) == 0:
            print(f'{name} contains 0 cuts')
            continue

        pos_times = []
        last_time = 0
        start_times.append(duration)

        for end_time in start_times:
            if end_time - last_time > 2:
                pos_times.append(end_time)
                last_time = end_time

        pos_times_shuffle = copy.deepcopy(pos_times)
        random.shuffle(pos_times_shuffle)
        for idx, end_time in enumerate(pos_times_shuffle):
            cmd = f'ffmpeg -y -threads 1 -ss {end_time-2.08} -t 2 -i {video_path} -threads 1 -vf "fps=fps=10" -vcodec libx264 -an {prefix}/pos_{pos_counter}.mp4'
            executor.submit(print_and_run, cmd)

            pos_map[pos_counter] = [source, name, end_time]
            pos_counter += 1
            if idx > pos_per_clip:
                break

        pos_times = [0] + pos_times
        neg_times = []
        for idx, start_time in enumerate(pos_times[:-1]):
            neg_start_time = start_time
            neg_end_time = pos_times[idx+1]-2
            if neg_end_time - neg_start_time > 2:
                neg_times.append([neg_start_time, neg_end_time])

        random.shuffle(neg_times)
        for idx, pair in enumerate(neg_times):
            start, end = pair
            end_time = random.uniform(start+2, end)

            cmd = f'ffmpeg -y -threads 1 -ss {end_time-2.08} -t 2 -i {video_path} -threads 1 -vf "fps=fps=10" -vcodec libx264 -an {prefix}/neg_{neg_counter}.mp4'
            executor.submit(print_and_run, cmd)

            neg_map[neg_counter] = [source, name, end_time]
            neg_counter += 1
            if idx > neg_per_clip:
                break

    with open('pos_map.json', 'w') as pos_fout:
        json.dump(pos_map, pos_fout)

    with open('neg_map.json', 'w') as neg_fout:
        json.dump(neg_map, neg_fout)

        config = r.read_id(name)
        video_path = config['video']
        start_times = config['start_times']
        duration = config['duration']

        if len(start_times) == 2:
            print(f'{name} contains 0 cuts')
            continue

        # anything smaller than -2 works
        pos_times = []
        last_time = -10
        for start_time in start_times:
            if start_time - last_time > 2:
                pos_times.append(start_time)
                last_time = start_time
        if duration - pos_times[-1] < 2:
            pos_times.pop()

        for idx, start_time in enumerate(pos_times):
            cmd = f'ffmpeg -y -threads 1 -ss {start_time} -t 2 -i {video_path} -threads 1 -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an {prefix}_{r.model_prefix}{idx}_pos.mp4'
            print(cmd)
            executor.submit(os.system, cmd)

        pos_times.append(duration)
        neg_times = []
        for idx, start_time in enumerate(pos_times[:-1]):
            neg_start_time = start_time + 2
            neg_end_time = pos_times[idx+1]
            if neg_end_time - neg_start_time > 2:
                neg_times.append([neg_start_time, neg_end_time])

        for idx, pair in enumerate(neg_times):
            start, end = pair
            cmd = f'ffmpeg -y -threads 1 -ss {start} -t {end-start} -i {video_path} -threads 1 -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an {prefix}_{r.model_prefix}{idx}_neg.mp4'
            print(cmd)
            executor.submit(os.system, cmd)

            info = {
                'duration': end-start
            }

            info_filename = f'{prefix}_{r.model_prefix}{idx}_neg.json'
            with open(info_filename, 'w') as fout:
                json.dump(info, fout)

if __name__ == "__main__":
    youtube1 = DatasetReader.create('youtube', '/home/code-base/data_space/youtube', shotcut_path='/shotcut/', model_prefix='', model_name='Model 1', vis_prefix='cache')
    generate_visualization('youtube', youtube1)
    print('All task has generated')
    executor.shutdown()
    print('All done')