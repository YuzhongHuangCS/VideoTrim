import collections
import gc
import glob
import json
import os
import pdb
import pickle
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import sklearn.metrics


#model.eval()


#files = os.listdir('/home/code-base/data_space/')
test_video_prefix = 'data/'

names_test = os.listdir('data/vimeo90k/videos')
#names_test = ['3e827a46-7d79-4ac9-a6ec-93321d6fcbc8.mp4']

for test_video_file in names_test:
    test_name_full = 'vimeo90k_' + test_video_file.split('.')[0]
    #source, test_name = test_name_full.split('_')
    source = 'vimeo90k'
    #test_video_file = test_name+'.mp4'
    cache_video_prefix = 'cache_test/'

    #input_file = f'test/{test_video_file}'
    input_file = f'{test_video_prefix}{source}/videos/{test_video_file}'
    #if not os.path.exists(input_file):
    #    continue
        #input_file = input_file.replace('.mp4', '.mov')

    output_file = f'{cache_video_prefix}vimeo90k_{test_video_file}'
    cmd = f'ffmpeg -y -i {input_file} -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an {output_file}'
    os.system(cmd)

    vis_file = f'cache_vis/vimeo90k_{test_video_file}'
    cmd = f'ffmpeg -y -i {input_file} -vf "fps=fps=8" -s 1080x720 -vcodec libx264 -an {vis_file}'
    os.system(cmd)

print('All task has generated')
#executor.shutdown()
print('All done')
