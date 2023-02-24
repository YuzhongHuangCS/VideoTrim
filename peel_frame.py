import os

import collections
import gc
import glob
import json
import os
import pdb
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

def print_and_run(cmd):
    print(cmd)
    os.system(cmd)

executor = ProcessPoolExecutor(max_workers=36)

test_video_prefix = 'editstock/footage/'
cache_video_prefix = 'cache_test/'

test_filenames = os.listdir(test_video_prefix)
for test_video_file in test_filenames:
    test_name = test_video_file.split('.')[0]

    output_file = cache_video_prefix + test_video_file.split('.')[0] + '.mp4'
    cmd = f'ffmpeg -y -threads 1 -i {test_video_prefix}{test_video_file} -threads 1 -vf select="gte(n\, 3)" -an editstock/footage_peel/{test_video_file}'
    executor.submit(print_and_run, cmd)
    #os.system(cmd)

print('All task has generated')
executor.shutdown()
print('All done')
