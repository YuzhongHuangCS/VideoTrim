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
import random
from dataset_reader import DatasetReader
import sys

if sys.platform == 'darwin':
    sys_prefix = '/Users/yhuang/video_trim/data/'
    num_cpus = 8
else:
    sys_prefix = '/home/code-base/data_space/'
    num_cpus = 29

def print_and_run(cmd):
    print(cmd)
    os.system(cmd)

executor = ProcessPoolExecutor(max_workers=num_cpus)

#generate end clips:
def generate_visualization(source, r):
    names = r.get_ids()
    pos_counter = 0
    neg_counter = 0
    pos_per_clip = sys.maxsize
    neg_per_clip = sys.maxsize
    pos_map = {}
    neg_map = {}

    prefix = r.vis_prefix
    os.makedirs(prefix, exist_ok=True)
    for name in names:
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
            if end_time - last_time > 2.5:
                pos_times.append(end_time)
            last_time = end_time

        #random.shuffle(pos_times)
        for idx, end_time in enumerate(pos_times):
            cmd = f'ffmpeg -y -threads 1 -ss {end_time-1} -i {video_path} -threads 1 -vframes 1 -vf "scale=iw/2:ih/2" {prefix}/pos_{pos_counter}.jpg'
            executor.submit(print_and_run, cmd)

            pos_map[pos_counter] = [source, name, end_time]
            pos_counter += 1
            if idx > pos_per_clip:
                break

        end_times = [0] + start_times
        neg_times = []
        for idx, start_time in enumerate(end_times[:-1]):
            neg_start_time = start_time
            neg_end_time = end_times[idx+1]-2
            if neg_end_time - neg_start_time > 2.5:
                neg_times.append([neg_start_time, neg_end_time])

        #random.shuffle(neg_times)
        for idx, pair in enumerate(neg_times):
            start, end = pair
            duration = end - start

            cmd = f'ffmpeg -y -threads 1 -ss {start + 1} -i {video_path} -threads 1 -vframes 1 -vf "scale=iw/2:ih/2" {prefix}/neg_{neg_counter}.jpg'
            
            executor.submit(print_and_run, cmd)

            neg_map[neg_counter] = [source, name, start, end]
            neg_counter += 1
            if idx > neg_per_clip:
                break

    with open('map_data/pos_map_jpg.json', 'w') as pos_fout:
        json.dump(pos_map, pos_fout)

    with open('map_data/neg_map_jpg.json', 'w') as neg_fout:
        json.dump(neg_map, neg_fout)

if __name__ == "__main__":
    youtube1 = DatasetReader.create('youtube', sys_prefix + 'youtube', shotcut_path='/shotcut/', vis_prefix='cache_jpg2')
    generate_visualization('youtube', youtube1)
    print('All task has generated')
    executor.shutdown()
    print('All done')