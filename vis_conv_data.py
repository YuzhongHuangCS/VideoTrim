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

    prefix = r.vis_prefix
    os.makedirs(prefix, exist_ok=True)
    for name in names:
        config = r.read_id(name)
        video_path = config['video']

        probe = ffmpeg.probe(video_path)
        print(video_path, probe['streams'][0]['avg_frame_rate'])
    
        #new_path = f'cache_whole/{source}_{name}.mp4'
        cmd = f'ffmpeg -y -threads 1 -i {video_path} -threads 1 -vf "fps=fps=8" -s 112x112 -aspect 1  -an {new_path}'
        #executor.submit(print_and_run, cmd)


if __name__ == "__main__":
    #vimeo90k1 = DatasetReader.create('vimeo90k', sys_prefix + 'vimeo90k', shotcut_path='/shotcut/', vis_prefix='cache_exa')
    #generate_visualization('vimeo90k', vimeo90k1)

    youtube1 = DatasetReader.create('youtube', sys_prefix + 'youtube', shotcut_path='/shotcut/', vis_prefix='cache_exa2')
    generate_visualization('youtube', youtube1)
    
    #roughcut1 = DatasetReader.create('roughcut', sys_prefix + 'roughcut', shotcut_path='/shotcut/', vis_prefix='cache_exa')
    #generate_visualization('roughcut', roughcut1)

    print('All task has generated')
    executor.shutdown()
    print('All done')