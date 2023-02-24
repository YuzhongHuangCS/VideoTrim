import os
import json
import pdb
import sys
import numpy as np
import random
import collections
from concurrent.futures import ProcessPoolExecutor

test_video_prefix = 'BCD/'
test_filenames = sorted(os.listdir(test_video_prefix))
cuts_pos = collections.defaultdict(list)
cuts_neg = collections.defaultdict(list)
#cuts_rnd = collections.defaultdict(list)

def print_and_run(cmd):
    print(cmd)
    os.system(cmd)

def find_peaks(start, end, cuts, vis_file):
    start_peak = []
    end_peak = []
    assert len(start) == len(end)
    for i in range(1, len(start) - 1):
        if start[i] > 0.5:
            if start[i-1] < start[i] and start[i] > start[i+1]:
                start_peak.append(i)
        if end[i] > 0.5:
            if end[i-1] < end[i] and end[i] > end[i+1]:
                end_peak.append(i)

    for i in range(0, len(start_peak)):
        frame_start = start_peak[i]
        score = start[frame_start]
        time_start = frame_start / 8
        cuts[vis_file].append([time_start, time_start + 2, score])

for test_video_file in test_filenames:
    test_name = test_video_file.split('.')[0]
    vis_file = 'BCD/' + test_name + '.mp4'
    json_filename = f'scores/start_{test_name}.json'
    if not os.path.exists(json_filename):
        continue

    data = json.loads(open(json_filename).read())

    start = data['start']
    end = data['end']
    find_peaks(start, end, cuts_pos, vis_file)

    start_flip = 1 - np.asarray(start)
    end_flip = 1 - np.asarray(end)
    find_peaks(start_flip, end_flip, cuts_neg, vis_file)

    '''
    total_time = len(start) / 8
    for _ in range(5):
        time_start = random.uniform(5, total_time - 20)
        time_end = time_start + random.uniform(3, 15)
        cuts_rnd[vis_file].append([time_start, time_end, 0])
    '''
for k in cuts_pos:
    cuts_pos[k].sort(key=lambda x: -x[2])

for k in cuts_neg:
    cuts_neg[k].sort(key=lambda x: -x[2])

#for k in cuts_rnd:
#    random.shuffle(cuts_rnd[k])

config = {
    'pos': cuts_pos,
    'neg': cuts_neg,
    #'rnd': cuts_rnd
}

name_map = {
    'pos': 'pos',
    'neg': 'neg',
}

filename_list = sorted(list(cuts_pos.keys() & cuts_neg.keys()))

os.makedirs('clips', exist_ok=True)
cmds = []
for name, ary in config.items():
    for i, vis_file in enumerate(filename_list):
        cut = ary[vis_file][0]

        time_start = cut[0]
        time_end = cut[1]

        duration = time_end - time_start
        cmd = f'ffmpeg -ss {time_start} -t {duration} -i {vis_file} -vcodec libx264 -an clips_start/{i}_{name_map[name]}.mp4'
        cmds.append(cmd)
        os.system(cmd)

#pdb.set_trace()
with open('cmds.txt', 'w') as fout:
    for cmd in cmds:
        fout.write(cmd + '\n')

print('All done')
