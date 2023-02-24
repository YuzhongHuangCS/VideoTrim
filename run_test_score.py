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
from imblearn.over_sampling import RandomOverSampler
    
files = os.listdir('/home/code-base/data_space/')
test_video_prefix = '/home/code-base/data_space/youtube/videos/'

names_test = json.loads(open('names_test_end.json').read())
for test_name in names_test:
    score_filename = f'scores/{test_name}.pk'
    with open(score_filename, 'rb') as fout:
        Y_ary = pickle.load(fout)

    Y_ary = [0] * 16 + Y_ary
    '''
    Y_ary = [float(x) for x in Y_ary]

    json_filename = f'scores/{test_name}.json'
    with open(json_filename, 'w') as fout:
        json.dump(Y_ary, fout)
    
    continue
    '''
    frames = len(Y_ary)
    shotcut_filename = f'/home/code-base/data_space/youtube/shotcut/{test_name}_shotcut.json'
    shotcut = json.loads(open(shotcut_filename).read())
    truth_idx = np.rint(np.asarray([v['startTime'] / 1000.0 for v in shotcut['atoms']]) * 8).astype(int)
    truth = np.asarray([0] * frames)
    truth[truth_idx] =1
    
    Y_ary = sklearn.preprocessing.binarize(np.asarray(Y_ary).reshape(-1, 1), threshold=0.5)
    
    ros = RandomOverSampler(random_state=1234)
    Y_resampled, truth_resampled = ros.fit_resample(Y_ary, truth)
    
    truth = truth_resampled
    Y_ary = Y_resampled
    acc = sklearn.metrics.accuracy_score(truth, Y_ary)
    precision = sklearn.metrics.precision_score(truth, Y_ary)
    recall = sklearn.metrics.recall_score(truth, Y_ary)
    f1 = sklearn.metrics.f1_score(truth, Y_ary)
    conf = sklearn.metrics.confusion_matrix(truth, Y_ary)
    
    output_filename = f'scores_vis/{test_name}_resample.txt'
    with open(output_filename, 'w') as fout:
        fout.write(f'Accuracy: {acc}\n')
        fout.write(f'Precision: {precision}\n')
        fout.write(f'Recall: {recall}\n')
        fout.write(f'F1: {f1}\n')
        fout.write(f'Confusion Matrix:\n{conf}\n')
    

print('All done')