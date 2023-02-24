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

        remove_dict = {}
        for end_time in start_times:
            if end_time - last_time > 2:
                pos_times.append(end_time)
                if end_time - last_time <= 2.5:
                    remove_dict[end_time] = end_time - last_time
            last_time = end_time

        #random.shuffle(pos_times)
        #print(remove_dict)
        for idx, end_time in enumerate(pos_times):
            if end_time in remove_dict:
                if os.path.exists(f'{prefix}/pos_{pos_counter}.mp4'):
                    print(f'{prefix}/pos_{pos_counter}.mp4', remove_dict[end_time])
                    os.remove(f'{prefix}/pos_{pos_counter}.mp4')

            pos_counter += 1


if __name__ == "__main__":
    youtube1 = DatasetReader.create('youtube', sys_prefix + 'youtube', shotcut_path='/shotcut/', vis_prefix='cache')
    generate_visualization('youtube', youtube1)
    print('All task has generated')
    executor.shutdown()
    print('All done')