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
import ffmpeg
import collections

if sys.platform == 'darwin':
    sys_prefix = '/Users/yhuang/video_trim/data/'
    num_cpus = 8
else:
    sys_prefix = '/home/code-base/data_space/'
    num_cpus = 29

def print_and_run(cmd):
    print(cmd)
    os.system(cmd)

#generate end clips:
def generate_visualization(source, r):
    names = sorted(r.get_ids())

    prefix = r.vis_prefix
    os.makedirs(prefix, exist_ok=True)
    
    db = collections.defaultdict(dict)
    for name in names:
        config = r.read_id(name)
        video_path = config['video']
        start_times = config['start_times']
        duration = config['duration']

        if len(start_times) == 0:
            print(f'{name} contains 0 cuts')
            continue

        in_fname = f'cache_whole/{source}_{name}.mp4'
        if not os.path.exists(in_fname):
            continue
        print(in_fname)

        pos_times = []
        last_time = 0
        start_times.append(duration)

        for end_time in start_times:
            if end_time - last_time > 2.5:
                pos_times.append(end_time)
            last_time = end_time

        #random.shuffle(pos_times)
        db[name]['start'] = []
        db[name]['end'] = []
        db[name]['neg'] = []
        for idx, end_time in enumerate(pos_times):
            db[name]['end'].append(end_time-2-1.0/8)
            data = torchvision.io.read_video(in_fname, start_pts=(end_time-2), end_pts=end_time+1/8, pts_unit='sec')[0]
            print(data.shape)
            out_fname = f'cache_debug2/end_{idx}.mp4'
            torchvision.io.write_video(out_fname, data, fps=8)

        end_times = [0] + start_times
        neg_times = []
        for idx, start_time in enumerate(end_times[:-1]):
            neg_start_time = start_time+2
            neg_end_time = end_times[idx+1]-2
            if neg_end_time - neg_start_time > 2.5:
                neg_times.append([neg_start_time, neg_end_time])

        #random.shuffle(neg_times)
        for idx, pair in enumerate(neg_times):
            start, end = pair
            db[name]['neg'].append([start, end])
            data = torchvision.io.read_video(in_fname, start_pts=start, end_pts=end, pts_unit='sec')[0]
            out_fname = f'cache_debug2/neg_{idx}.mp4'
            torchvision.io.write_video(out_fname, data, fps=8)

        #start
        pos_times.insert(0, 0)
        pos_times.pop()            
        #random.shuffle(pos_times)
        for idx, start_time in enumerate(pos_times):
            db[name]['start'].append(start_time)
            data = torchvision.io.read_video(in_fname, start_pts=start_time+1/32, end_pts=(start_time+2), pts_unit='sec')[0]
            #print(data.shape)
            out_fname = f'cache_debug2/start_{idx}.mp4'
            torchvision.io.write_video(out_fname, data, fps=8)

        return db
    return db


if __name__ == "__main__":
    #vimeo90k1 = DatasetReader.create('vimeo90k', sys_prefix + 'vimeo90k', shotcut_path='/shotcut/', vis_prefix='cache_exa')
    #generate_visualization('vimeo90k', vimeo90k1)

    youtube1 = DatasetReader.create('youtube', sys_prefix + 'youtube', shotcut_path='/shotcut/', vis_prefix='cache_exa2')
    db = generate_visualization('youtube', youtube1)
    
    pdb.set_trace()
    #roughcut1 = DatasetReader.create('roughcut', sys_prefix + 'roughcut', shotcut_path='/shotcut/', vis_prefix='cache_exa')
    #generate_visualization('roughcut', roughcut1)


    print('All done')