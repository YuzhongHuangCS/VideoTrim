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
import shutil

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
import ffmpeg


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
model.cuda()
torch.no_grad()
model.eval()
    
    
#files = os.listdir('/home/code-base/data_space/')
test_video_prefix = '/home/code-base/data_space/'

names_test = json.loads(open('split_config.json').read())['test']
#names_test = ['youtube_c032f248-2b5e-4019-9c5f-672f3769dd82']

for test_name_full in names_test:
    source, test_name = test_name_full.split('_')
    test_video_file = test_name+'.mp4'
    cache_video_prefix = 'cache_test/'

    input_file = f'{test_video_prefix}{source}/videos/{test_video_file}'
    if not os.path.exists(input_file):
        continue
        #input_file = input_file.replace('.mp4', '.mov')

    probe = ffmpeg.probe(input_file)
    duration = float(probe['format']['duration'])
    
    output_file = f'{cache_video_prefix}{test_name_full}.mp4'
    cmd = f'ffmpeg -y -i {input_file} -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an {output_file}'
    #os.system(cmd)

    if not os.path.exists(output_file):
        continue

    vis_file = f'cache_vis/{test_name_full}.mp4'
    cmd = f'ffmpeg -y -i {input_file} -vf "fps=fps=8" -s 1080x720 -vcodec libx264 -an {vis_file}'
    #os.system(cmd)

    frames = torchvision.io.read_video(output_file, pts_unit='sec')[0].to(device)
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]

    try:
        shotcut_filename = f'/home/code-base/data_space/{source}/shotcut/{test_name}.json'
        shotcut = json.loads(open(shotcut_filename).read())
    except Exception as e:
        shotcut_filename = f'/home/code-base/data_space/{source}/shotcut/{test_name}_shotcut.json'
        shotcut = json.loads(open(shotcut_filename).read())
            
    start_times = [x['startTime']/1000 for x in shotcut['atoms']]
    pos_times = []
    last_time = 0
    start_times.append(duration)

    for end_time in start_times:
        if end_time - last_time > 2.5:
            pos_times.append(end_time)
        last_time = end_time

    end_frames = {}
    for idx, end_time in enumerate(pos_times):
        end_frames[int(round(end_time * 8))-1] = idx

    pos_times.insert(0, 0)
    pos_times.pop()  
    start_frames = {}
    for idx, start_time in enumerate(pos_times):
        start_frames[int(round(start_time * 8))] = idx

    end_ary = [0] * 15
    start_ary = []
    neg_ary = []
    for i in range(0, len(frames)-16):
        if i in start_frames:
            v_f = f'cache_exa/start_{test_name_full}_{start_frames[i]}.mp4'
            X = torchvision.io.read_video(v_f, pts_unit='sec')[0][0:16, :, :, :].to(device)
            #print('start preload', i)
        elif (i+15) in end_frames:
            v_f = f'cache_exa/end_{test_name_full}_{end_frames[i+15]}.mp4'
            X = torchvision.io.read_video(v_f, pts_unit='sec')[0][0:16, :, :, :].to(device)
            #print('end preload', i+15)
            #shutil.copyfile(v_f, f'cache_debug/{i}.mp4')
            
            #i_end = i+16
            #X2 = frames[i:i_end]
            #torchvision.io.write_video(f'cache_debug/{i}_write.mp4', X2, fps=8, video_codec='libx264')
        else:
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

    #shotcut_filename = f'/home/code-base/data_space/youtube/shotcut/{test_name}_shotcut.json'
    #video_filename = vis_file
    #output_filename = f'scores_vis/{test_name}.mp4'
    #executor.submit(visualize, score_filename, shotcut_filename, video_filename, output_filename)
    
print('All task has generated')
executor.shutdown()
print('All done')