import json
import sys
import os
import ffmpeg
import pdb
import collections
import random

from concurrent.futures import ProcessPoolExecutor
from dataset_reader import DatasetReader

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

'''
pos = json.loads(open('map_data/pos_map.json').read())

duration_map = collections.defaultdict(int)
total_counter = 0
n_total = len(pos)
pos_ary = list(pos.items())
random.shuffle(pos_ary)

for pos_counter, item in pos_ary:
    source, name, end_time = item
    video_path = sys_prefix + source + '/videos/' + name + '.mp4'
    out_path = f'cache/pos_{pos_counter}.mp4'
    if not os.path.exists(out_path):
        pdb.set_trace()
    else:
        probe = ffmpeg.probe(out_path)
        duration = float(probe['format']['duration'])
        duration_map[duration] += 1
        #print(duration)
    if total_counter % 1000 == 0:
        print(total_counter, n_total)
        print(duration_map)
    
    total_counter += 1
print(duration_map)
exit()
'''
neg = json.loads(open('map_data/pos_map.json').read())
from collections import defaultdict
duration_map = defaultdict(int)

for neg_counter, item in neg.items():
    source, name, end = item
    #source, name, start, end = item
    video_path = sys_prefix + source + '/videos/' + name + '.mp4'
    out_path = f'cache/pos_{neg_counter}.mp4'
    if not os.path.exists(out_path):
        pass
        #pdb.set_trace()
    else:
        probe = ffmpeg.probe(out_path)
        duration = float(probe['format']['duration'])
        duration_map[duration] += 1
        '''
        duration_should = 2
        
        #duration_should = end - start
        diff = abs(duration - duration_should)
        if diff < 0.2:
            pass
            #print(diff)
        else:
            print('Not', out_path, duration, duration_should, diff)
        '''
        #cmd = f'ffmpeg -y -threads 1 -ss {end_time-2} -t 1.95 -i {video_path} -threads 1 -vf "fps=fps=8" -vcodec libx264 -an {out_path}'
        #executor.submit(print_and_run, cmd)
print(duration_map)
print('All task has generated')
#executor.shutdown()
print('All done')