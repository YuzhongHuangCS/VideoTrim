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

sys_prefix = '/nas/home/yuzhongh/video_trim/data/'
num_cpus = 29

def print_and_run(cmd):
    print(cmd)
    os.system(cmd)

executor = ProcessPoolExecutor(max_workers=num_cpus)

#generate end clips:
def generate_visualization(source, r):
    names = r.get_ids()

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
            out_path = f'{prefix}/end_{source}_{name}_{idx}.mp4'
            if os.path.exists(out_path):
                probe = ffmpeg.probe(out_path)
                duration_real = float(probe['format']['duration'])
                if duration_real < 2.5:
                    continue  
                else:
                    print(out_path, duration_real)

            cmd = f'ffmpeg -y -threads 1 -ss {end_time-2.05} -t 2 -i {video_path} -threads 1 -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an {prefix}/end_{source}_{name}_{idx}.mp4'
            executor.submit(print_and_run, cmd)

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
            duration = end - start
            
            out_path = f'{prefix}/neg_{source}_{name}_{idx}.mp4'

            if os.path.exists(out_path):
                probe = ffmpeg.probe(out_path)
                duration_real = float(probe['format']['duration'])
                diff = abs(duration_real - duration)
                if diff < 0.3:
                    continue
                else:
                    print(out_path, duration_real, duration)

            cmd = f'ffmpeg -y -threads 1 -ss {start} -t {duration} -i {video_path} -threads 1 -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an {prefix}/neg_{source}_{name}_{idx}.mp4'
            executor.submit(print_and_run, cmd)

            info = {
                'duration': duration
            }

            info_filename = f'{prefix}/neg_{source}_{name}_{idx}.json'
            with open(info_filename, 'w') as fout:
                json.dump(info, fout)
                
        #start
        pos_times.insert(0, 0)
        pos_times.pop()            
        #random.shuffle(pos_times)
        for idx, start_time in enumerate(pos_times):
            out_path = f'{prefix}/start_{source}_{name}_{idx}.mp4'
            if os.path.exists(out_path):
                probe = ffmpeg.probe(out_path)
                duration_real = float(probe['format']['duration'])
                if duration_real < 2.5:
                    continue
                else:
                    print(out_path, duration_real)
            
            cmd = f'ffmpeg -y -threads 1 -ss {start_time} -t 2 -i {video_path} -threads 1 -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an {prefix}/start_{source}_{name}_{idx}.mp4'
            executor.submit(print_and_run, cmd)


if __name__ == "__main__":
    #vimeo90k1 = DatasetReader.create('vimeo90k', sys_prefix + 'vimeo90k', shotcut_path='/shotcut/', vis_prefix='cache_exa')
    #generate_visualization('vimeo90k', vimeo90k1)

    youtube1 = DatasetReader.create('youtube', sys_prefix + 'youtube', shotcut_path='/shotcut/', vis_prefix='cache_exa')
    generate_visualization('youtube', youtube1)
    
    #roughcut1 = DatasetReader.create('roughcut', sys_prefix + 'roughcut', shotcut_path='/shotcut/', vis_prefix='cache_exa')
    #generate_visualization('roughcut', roughcut1)

    print('All task has generated')
    executor.shutdown()
    print('All done')
