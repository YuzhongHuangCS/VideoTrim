import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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

from matplotlib import gridspec
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.animation as animation
from tqdm import tqdm
import pdb
from PIL import Image
import torchvision.io
import pickle
import cv2
import json
from concurrent.futures import ProcessPoolExecutor
import sklearn.preprocessing


executor = ProcessPoolExecutor(max_workers=2)

class SubplotAnimation():
    def __init__(self, fig, files, preds, cap, truth):
        self.files = files
        self.preds = preds

        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        self.ax1 = fig.add_subplot(gs[0])
        self.ax2 = fig.add_subplot(gs[1])

        self.ax2.get_xaxis().set_animated(True)

        self.ax2.set_title('Model')

        self.ax1.axis('off')

        zeros = np.zeros((720, 1080), dtype=np.uint8)
        self.im = self.ax1.imshow(zeros, aspect="equal")

        self.plt1, = self.ax2.plot((0, 0), 'r')
        self.plt2, = self.ax2.plot((0, 0), 'bo')

        self.plt3, = self.ax2.plot((0, 0), 'g')
        self.plt4, = self.ax2.plot((0, 0), 'go')

        self.plotwidth = 25
        self.idx = 0

        self.t = tqdm(total=len(preds), desc='Animation')
        self.frames = np.arange(0, len(preds))
        self.cap = cap
        self.truth = truth

    def __del__(self):
        self.t.close()

    def updateData(self, *args):
        self.t.update(1)
        pred = self.preds[self.idx]
        truth = self.truth[self.idx]
        #file = self.files[self.idx]
        #img = loadImage(file)
        ret, img_cv = self.cap.read()
        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        self.im.set_data(img)

        self.plt1.set_data(self.frames, self.preds)
        self.plt2.set_data((self.idx, pred))
        self.plt3.set_data(self.frames, self.truth)
        self.plt4.set_data((self.idx, truth))

        # center the plot
        self.ax2.set_xlim((self.idx - self.plotwidth, self.idx + self.plotwidth))

        # set this to the total range of predictions
        self.ax2.set_ylim((-0.2, 1.2))

        self.idx = self.idx + 1
        return self.im, self.plt1, self.plt2, self.plt3, self.plt4


def visualize(score_filename, shotcut_filename, video_filename, output_filename):
    fin = open(score_filename, 'rb')
    preds = [0] * 16 + pickle.load(fin)
    frames = len(preds)

    shotcut = json.loads(open(shotcut_filename).read())
    truth_idx = np.rint(np.asarray([v['startTime'] / 1000.0 for v in shotcut['atoms']]) * 8).astype(int)
    truth = np.asarray([0] * frames)
    truth[truth_idx] =1

    cap = cv2.VideoCapture(video_filename)

    output_fn = output_filename

    # should be a list of the files for the video, for now I just
    # duplicate a single image
    
    files = None
    #files = np.asarray((['img.jpg']) * frames)

    # these are the predictions from the neural network
    #preds = np.random.random(size=(frames, ))

    fig = plt.figure(figsize=(8, 10), dpi=80)
    anim = SubplotAnimation(fig, files, preds, cap, truth)
    ani = animation.FuncAnimation(fig,
                                  anim.updateData,
                                  frames=frames - 1,
                                  blit=False,
                                  interval=125)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=8, bitrate=2000)
    ani.save(output_fn, writer=writer)


def float2int_nan(ary):
    ret = []
    for x in ary:
        if np.isnan(x):
            ret.append(None)
        else:
            ret.append(int(x))
    return ret

            
#files = os.listdir('/home/code-base/data_space/')
test_video_prefix = '/home/code-base/data_space/'

names_test = json.loads(open('split_config.json').read())['test']
scores_recording = []
names_test = ['roughcut_1ce975f5-17ce-4979-b325-b7f1732817d6']
for test_name_full in names_test:
    source, test_name = test_name_full.split('_')
    test_video_file = test_name+'.mp4'
    cache_video_prefix = 'cache_test/'
    
    input_file = f'{test_video_prefix}{source}/videos/{test_video_file}'
    output_file = f'{cache_video_prefix}{test_name_full}.mp4'
    
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

    vis_file = f'cache_vis/{test_name_full}.mp4'
    cmd = f'ffmpeg -y -i {input_file} -vf "fps=fps=8" -s 1080x720 -vcodec libx264 -an {vis_file}'
    #os.system(cmd)

    json_filename = f'scores/start_{test_name_full}.json'
    if os.path.exists(json_filename):

        with open(json_filename, 'r') as fin:
            try:
                data = json.load(fin)
            except Exception as e:
                os.remove(json_filename)
                continue
            data['end'] = data['end'][1:]
            data['start'] = data['start'][:-1]
            
        try:
            shotcut_filename = f'/home/code-base/data_space/{source}/shotcut/{test_name}.json'
            shotcut = json.loads(open(shotcut_filename).read())
        except Exception as e:
            shotcut_filename = f'/home/code-base/data_space/{source}/shotcut/{test_name}_shotcut.json'
            shotcut = json.loads(open(shotcut_filename).read())

        start_times = [x['startTime']/1000 for x in shotcut['atoms']]
        
        truth_idx = np.rint(np.asarray([v['startTime'] / 1000.0 for v in shotcut['atoms']]) * 8).astype(int)
        n_entry = len(data['end'])
        truth = np.asarray([0] * n_entry)
        if len(truth) == 15:
            continue
        truth[truth_idx] = 1
        
        truth_start = truth.astype(np.float64)
        truth_end = np.asarray(truth.tolist()[1:] + [0]).astype(np.float64)
        #data['truth_start'] = truth.tolist()
        #data['truth_end'] = truth.tolist()[1:] + [0]

        start_zero_mark = np.asarray([None] * n_entry)
        end_zero_mark = np.asarray([None] * n_entry)
        start = np.asarray(data['start'])
        end = np.asarray(data['end'])
        for i in range(1, 16):
            start[np.maximum(truth_idx-i, 0)] = None
            end[np.minimum(truth_idx+i-1, n_entry-1)] = None
            
            
            #if i != 1:    
            truth_start[np.maximum(truth_idx-i, 0)] = None
            truth_end[np.minimum(truth_idx+i-1, n_entry-1)] = None

            start_zero_mark[np.maximum(truth_idx-i, 0)] = 1
            end_zero_mark[np.minimum(truth_idx+i-1, n_entry-1)] = 1
            
        pdb.set_trace()
        data['start'] = start.tolist()
        data['end'] = end.tolist()
        data['truth_start'] = float2int_nan(truth_start)
        data['truth_end'] = float2int_nan(truth_end)
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
        with open(json_filename, 'w') as fout:
            json_str = json.dumps(data)
            fout.write(json_str.replace('NaN', 'null'))
            
        scores_recording.append([(data['start_f1'] + data['end_f1'])/2.0, test_name_full])
        if os.path.exists(vis_file):
            os.rename(vis_file, f'scores_new/{test_name_full}.mp4')

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