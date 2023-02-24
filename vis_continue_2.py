import json
import sys
import os
import ffmpeg
import pdb

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

for pos_counter, item in pos.items():
    source, name, end_time = item
    video_path = sys_prefix + source + '/videos/' + name + '.mp4'
    out_path = f'abtest/pos_{pos_counter}.mp4'
    if not os.path.exists(out_path):
        pdb.set_trace()
    else:
        
        cmd = f'ffmpeg -y -threads 1 -ss {end_time-2} -t 1.95 -i {video_path} -threads 1 -vf "fps=fps=8" -vcodec libx264 -an {out_path}'
        executor.submit(print_and_run, cmd)

        out_path = f'abtest/pos_{pos_counter}.gif'

        cmd = f'ffmpeg -y -threads 1 -ss {end_time-2} -t 1.95 -i {video_path} -threads 1 -vf "fps=fps=8" -f gif -an {out_path}'
        executor.submit(print_and_run, cmd)
'''
neg = json.loads(open('map_data/neg_map.json').read())
for neg_counter, item in neg.items():
    if int(neg_counter) < 15000:
        continue
    source, name, start, end = item
    video_path = sys_prefix + source + '/videos/' + name + '.mp4'
    out_path = f'cache/neg_{neg_counter}.mp4'
    if not os.path.exists(out_path):
        pass
    else:
        probe = ffmpeg.probe(out_path)
        duration = float(probe['format']['duration']) 
        duration_should = end - start
        diff = abs(duration - duration_should)
        if diff < 0.2:
            pass
        else:
            os.remove(f'cache/neg_{neg_counter}.mp4')
            #cmd = f'ffmpeg -y -threads 1 -ss {start} -t {end-start} -i {video_path} -threads 1 -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an cache/neg_{neg_counter}.mp4'
            print(neg_counter, duration, duration_should)
            #print(start, end)
            #print(cmd)
            #os.system(cmd)
            #pdb.set_trace()
            #executor.submit(print_and_run, cmd)

print('All task has generated')
#executor.shutdown()
print('All done')