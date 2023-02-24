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

#import cv2
#import ffmpeg
import numpy as np
#import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.io
#from matplotlib import gridspec
#from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import resnet

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cuda")
#device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

pretrain = torch.load('model/r3d18_K_200ep.pth', map_location='cuda')
model = resnet.generate_model(model_depth=18, n_classes=700)
model.load_state_dict(pretrain['state_dict'])
model.fc = nn.Linear(model.fc.in_features, 3)
#load 20
model_path = f'model/exa_resnet_18_xentropy_epoch_13.pth'
model.load_state_dict(torch.load(model_path, map_location='cuda'))

freeze = True
if freeze:
    model_freeze = resnet.generate_model(model_depth=18, n_classes=700)
    model_freeze.load_state_dict(pretrain['state_dict'])

    model.conv1.weight = model_freeze.conv1.weight
    model.bn1.weight = model_freeze.bn1.weight
    model.bn1.bias = model_freeze.bn1.bias

model.cuda()
torch.no_grad()
model.eval()


#files = os.listdir('/home/code-base/data_space/')
test_video_prefix = 'data/'

names_test = os.listdir('data/roughcut/videos')
#names_test = ['3e827a46-7d79-4ac9-a6ec-93321d6fcbc8.mp4']

for test_video_file in names_test:
    test_name_full = 'roughcut_' + test_video_file.split('.')[0]
    #source, test_name = test_name_full.split('_')
    source = 'roughcut'
    #test_video_file = test_name+'.mp4'
    cache_video_prefix = 'cache_test/'

    #input_file = f'test/{test_video_file}'
    input_file = f'{test_video_prefix}{source}/videos/{test_video_file}'
    #if not os.path.exists(input_file):
    #    continue
        #input_file = input_file.replace('.mp4', '.mov')

    output_file = f'{cache_video_prefix}roughcut_{test_video_file}'
    cmd = f'ffmpeg -y -i {input_file} -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an {output_file}'
    #os.system(cmd)

    vis_file = f'cache_vis/roughcut_{test_video_file}'
    cmd = f'ffmpeg -y -i {input_file} -vf "fps=fps=8" -s 1080x720 -vcodec libx264 -an {vis_file}'
    #os.system(cmd)

    #continue

    #if not os.path.exists(output_file):
    #    continue

    frames = torchvision.io.read_video(output_file, pts_unit='sec')[0].to(device)
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]

    end_ary = [0] * 15
    start_ary = []
    neg_ary = []
    for i in range(0, len(frames)-16):
        print(i)

        i_end = i+16
        X = frames[i:i_end]

        X = (torch.unsqueeze(X, 0) / 255.0)
        for z in range(3):
            X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

        X = X.permute(0, 4, 1, 2, 3)
        emb, output = model(X)

        end_prob_ary = torch.nn.functional.softmax(output[:, 0:2]).cpu().detach().numpy()[0]
        end_ary.append(end_prob_ary[1])

        start_prob_ary = torch.nn.functional.softmax(output[:, [0,2]]).cpu().detach().numpy()[0]
        start_ary.append(start_prob_ary[1])

    start_ary += [0] * 15
    start_ary = [float(x) for x in start_ary]
    end_ary = [float(x) for x in end_ary]

    json_filename = f'scores/start_{test_name_full}.json'
    with open(json_filename, 'w') as fout:
        data = {
            'start': start_ary,
            'end': end_ary
        }
        json.dump(data, fout)

    #score_filename = f'scores/{test_name}.pk'
    #with open(score_filename, 'wb') as fout:
    #    pickle.dump(Y_ary, fout)

    #shotcut_filename = f'/home/code-base/data_space/roughcut/shotcut/{test_name}_shotcut.json'
    #video_filename = vis_file
    #output_filename = f'scores_vis/{test_name}.mp4'
    #executor.submit(visualize, score_filename, shotcut_filename, video_filename, output_filename)

print('All task has generated')
executor.shutdown()
print('All done')
