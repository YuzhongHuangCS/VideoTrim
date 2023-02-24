import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torchvision
import torchvision.io
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import glob
import pdb
import json
import random
import torch.optim as optim
import resnet
import numpy as np
import time
import os
import gc
import sklearn.metrics
import collections
import pickle
import ffmpeg

import numpy as np
from tqdm import tqdm
import pdb
from PIL import Image
import torchvision.io
import pickle
#import cv2
import json
from concurrent.futures import ProcessPoolExecutor
import sklearn.preprocessing


executor = ProcessPoolExecutor(max_workers=2)



def float2int_nan(ary):
    ret = []
    for x in ary:
        if np.isnan(x):
            ret.append(None)
        else:
            ret.append(int(x))
    return ret


#files = os.listdir('/home/code-base/data_space/')
#test_video_prefix = '/home/code-base/data_space/'
test_video_prefix = 'data/'

#names_test = json.loads(open('split_config.json').read())['test']
scores_recording = []
names_test = os.listdir('data/roughcut/videos')
for test_video_file in names_test:
    test_name =  test_video_file.split('.')[0]
    test_name_full = 'roughcut_' + test_name
    source = 'roughcut'
    cache_video_prefix = 'cache_test/'

   # input_file = f'test/{test_video_file}'
    #input_file = f'{test_video_prefix}{source}/videos/{test_video_file}'
    #if not os.path.exists(input_file):
    #    continue
        #input_file = input_file.replace('.mp4', '.mov')

    input_file = f'{test_video_prefix}{source}/videos/{test_video_file}'
    output_file = f'{cache_video_prefix}roughcut_{test_video_file}'

    probe = ffmpeg.probe(input_file)
    duration = float(probe['format']['duration'])

    '''
    if not os.path.exists(output_file):
        cmd = f'ffmpeg -y -i {input_file} -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an {output_file}'
        os.system(cmd)

    if not os.path.exists(f'{cache_video_prefix}{test_name_full}'):
        os.makedirs(f'{cache_video_prefix}{test_name_full}', exist_ok=True)
        cmd = f'ffmpeg -y -i {output_file} {cache_video_prefix}{test_name_full}/%d.jpg'
        os.system(cmd)
    continue
    '''

    #vis_file = f'cache_vis/{test_name_full}.mp4'
    #cmd = f'ffmpeg -y -i {input_file} -vf "fps=fps=8" -s 1080x720 -vcodec libx264 -an {vis_file}'
    #os.system(cmd)

    json_filename = f'scores/start_{test_name_full}.json'
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as fin:
            try:
                data = json.load(fin)
            except Exception as e:
                os.remove(json_filename)
                continue

        #pdb.set_trace()
        try:
            shotcut_filename = f'data/{source}/shotcut/{test_name}.json'
            shotcut = json.loads(open(shotcut_filename).read())
        except Exception as e:
            shotcut_filename = f'data/{source}/shotcut/{test_name}_shotcut.json'
            if not os.path.exists(shotcut_filename):
                continue
            shotcut = json.loads(open(shotcut_filename).read())

        start_times = [x['startTime']/1000 for x in shotcut['atoms']]
        pos_times = []
        last_time = 0
        start_times.append(duration)

        for end_time in start_times:
            if end_time - last_time > 2.5:
                pos_times.append(end_time)
            last_time = end_time

        end_frames = []
        for idx, end_time in enumerate(pos_times):
            end_frames.append(int(round(end_time * 8)-1))

        pos_times.insert(0, 0)
        pos_times.pop()
        start_frames = []
        for idx, start_time in enumerate(pos_times):
            start_frames.append(int(round(start_time * 8)))

        start_frames = np.asarray(start_frames)
        end_frames = np.asarray(end_frames)
        #pdb.set_trace()

        #truth_idx = np.rint(np.asarray([v['startTime'] / 1000.0 for v in shotcut['atoms']]) * 8).astype(int)
        n_entry = len(data['end'])
        if n_entry == 15:
            continue
        truth_start = np.asarray([0] * n_entry)
        truth_start[np.minimum(start_frames, len(truth_start)-1)] = 1
        truth_end = np.asarray([0] * n_entry)
        truth_end[np.minimum(end_frames, len(truth_end)-1)] = 1
        data['truth_start'] = truth_start.tolist()
        data['truth_end'] = truth_end.tolist()

        truth_start = truth_start.astype(np.float64)
        truth_end = truth_end.astype(np.float64)
        #data['truth_start'] = truth.tolist()
        #data['truth_end'] = truth.tolist()[1:] + [0]

        start_zero_mark = np.asarray([None] * n_entry)
        end_zero_mark = np.asarray([None] * n_entry)
        start = np.asarray(data['start'])
        end = np.asarray(data['end'])
        #pdb.set_trace()

        for i in range(1, 16):
            #start[np.maximum(start_frames-i, 0)] = None
            #end[np.minimum(end_frames+i, n_entry-1)] = None


            #if i != 1:
            #truth_start[np.maximum(start_frames-i, 0)] = None
            #truth_end[np.minimum(end_frames+i, n_entry-1)] = None

            start_zero_mark[np.maximum(start_frames-i, 0)] = 1
            end_zero_mark[np.minimum(end_frames+i, n_entry-1)] = 1

        #pdb.set_trace()
        #data['start'] = start.tolist()
        #data['end'] = end.tolist()
        #data['truth_start'] = float2int_nan(truth_start)
        #data['truth_end'] = float2int_nan(truth_end)
        data['start_zero_mark'] = start_zero_mark.tolist()
        data['end_zero_mark'] = end_zero_mark.tolist()

        start_none_idx = np.isnan(start)
        end_none_idx = np.isnan(end)

        start_bin = sklearn.preprocessing.binarize(start[~start_none_idx].reshape(-1, 1), threshold=0.5)
        end_bin = sklearn.preprocessing.binarize(end[~end_none_idx].reshape(-1, 1), threshold=0.5)

        truth_start_valid = truth_start[~start_none_idx].astype(int)
        truth_end_valid = truth_end[~end_none_idx].astype(int)

        data['start_acc'] = sklearn.metrics.accuracy_score(truth_start_valid, start_bin)
        data['start_precision'] = sklearn.metrics.precision_score(truth_start_valid, start_bin)
        data['start_recall'] = sklearn.metrics.recall_score(truth_start_valid, start_bin)
        data['start_f1'] = sklearn.metrics.f1_score(truth_start_valid, start_bin)
        data['start_conf'] = sklearn.metrics.confusion_matrix(truth_start_valid, start_bin).tolist()

        data['end_acc'] = sklearn.metrics.accuracy_score(truth_end_valid, end_bin)
        data['end_precision'] = sklearn.metrics.precision_score(truth_end_valid, end_bin)
        data['end_recall'] = sklearn.metrics.recall_score(truth_end_valid, end_bin)
        data['end_f1'] = sklearn.metrics.f1_score(truth_end_valid, end_bin)
        data['end_conf'] = sklearn.metrics.confusion_matrix(truth_end_valid, end_bin).tolist()

        json_filename = f'scores_new/start_{test_name_full}.json'
        #print(json_filename)
        with open(json_filename, 'w') as fout:
            json_str = json.dumps(data)
            fout.write(json_str.replace('NaN', 'null'))

        scores_recording.append([(data['start_f1'] + data['end_f1'])/2.0, test_name_full])
        #if os.path.exists(vis_file):
        #    os.rename(vis_file, f'scores_new/{test_name_full}.mp4')

    #score_filename = f'scores/{test_name}.pk'
    #with open(score_filename, 'wb') as fout:
    #    pickle.dump(Y_ary, fout)

    #
    #video_filename = vis_file
    #output_filename = f'scores_vis/{test_name}.mp4'
    #executor.submit(visualize, score_filename, shotcut_filename, video_filename, output_filename)

scores_recording = sorted(scores_recording, reverse=True)
with open('scores_f1_both.json', 'w') as fout:
    json.dump(scores_recording, fout)

print('All task has generated')
executor.shutdown()
print('All done')
