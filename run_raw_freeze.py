import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.io
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import resnet

use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda")
#device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

pretrain = torch.load('model/r3d18_K_200ep.pth', map_location='cuda')
model = resnet.generate_model(model_depth=18, n_classes=700)
model.load_state_dict(pretrain['state_dict'])
model.fc = nn.Linear(model.fc.in_features, 3)

#load 20
model_path = f'model/exa_resnet_18_xentropy_epoch_13.pth'
model.load_state_dict(torch.load(model_path))

freeze = True
if freeze:
    model_freeze = resnet.generate_model(model_depth=18, n_classes=700)
    model_freeze.load_state_dict(pretrain['state_dict'])

    model.conv1.weight = model_freeze.conv1.weight
    model.bn1.weight = model_freeze.bn1.weight
    model.bn1.bias = model_freeze.bn1.bias

#pdb.set_trace()
model.cuda()
torch.no_grad()
model.eval()

executor = ProcessPoolExecutor(max_workers=24)
def print_and_run(cmd):
    print(cmd)
    os.system(cmd)


test_video_prefix = 'vimeo_raw2/'
cache_video_prefix = 'cache_test/'

test_filenames = os.listdir(test_video_prefix)
for test_video_file in test_filenames:
    test_name = test_video_file.split('.')[0]

    output_file = cache_video_prefix + test_video_file.split('.')[0] + '.mp4'
    cmd = f'ffmpeg -y -threads 1 -i {test_video_prefix}{test_video_file} -threads 1 -vf "fps=fps=8:round=up" -s 112x112 -aspect 1 -vcodec libx264 -an {output_file}'
    #executor.submit(print_and_run, cmd)
    #os.system(cmd)

    vis_file = 'cache_vis/' + test_video_file.split('.')[0] + '.mp4'
    cmd = f'ffmpeg -y -threads 1 -i {test_video_prefix}{test_video_file} -threads 1 -vf "fps=fps=8:round=up" -s 1080x720 -vcodec libx264 -an {vis_file}'
    #executor.submit(print_and_run, cmd)
    #os.system(cmd)
    #continue


    frames = torchvision.io.read_video(output_file, pts_unit='sec')[0].to(device)
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]

    end_ary = [0] * 15
    start_ary = []
    for i in range(0, len(frames)-16):
        if i % 100 == 0:
            print(i)
        i_end = i+15

        X = torch.unsqueeze(frames[i:i_end], 0)
        X = (X / 255.0)

        for z in range(3):
            X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

        X = X.permute(0, 4, 1, 2, 3)
        emb, output = model(X)

        prob = torch.nn.functional.softmax(output)
        prob_ary = prob.cpu().detach().numpy()[0]
        end_ary.append(prob_ary[1])
        start_ary.append(prob_ary[2])


    start_ary += [0] * 15
    start_ary = [float(x) for x in start_ary]
    end_ary = [float(x) for x in end_ary]

    json_filename = f'scores/start_{test_name}.json'
    with open(json_filename, 'w') as fout:
        data = {
            'start': start_ary,
            'end': end_ary
        }
        json.dump(data, fout)

    #continue
    #video_filename = vis_file
    #output_filename = f'scores_vis/start_{test_name}.mp4'
    #visualize(score_filename, video_filename, output_filename)
    #exit()
    #executor.submit(visualize, score_filename, video_filename, output_filename)


print('All task has generated')
executor.shutdown()
print('All done')
