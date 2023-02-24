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
videos = glob.glob('cache/*.mp4')

for v in videos:
    output_name = v.replace('.mp4', '.jpg').replace('cache', 'cache_jpg')
    cmd = f'ffmpeg -y -threads 1 -ss 1 -i {v} -threads 1 -vframes 1 {output_name}'
    executor.submit(print_and_run, cmd)

print('All task has generated')
executor.shutdown()
print('All done')