import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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



executor = ProcessPoolExecutor(max_workers=2)

class SubplotAnimation():
    def __init__(self, fig, files, preds, cap):
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

        self.plotwidth = 25
        self.idx = 0

        self.t = tqdm(total=len(preds), desc='Animation')
        self.frames = np.arange(0, len(preds))
        self.cap = cap

    def __del__(self):
        self.t.close()

    def updateData(self, *args):
        self.t.update(1)
        pred = self.preds[self.idx]
        #file = self.files[self.idx]
        #img = loadImage(file)
        ret, img_cv = self.cap.read()
        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        self.im.set_data(img)

        self.plt1.set_data(self.frames, self.preds)
        self.plt2.set_data((self.idx, pred))

        # center the plot
        self.ax2.set_xlim((self.idx - self.plotwidth, self.idx + self.plotwidth))

        # set this to the total range of predictions
        self.ax2.set_ylim((-0.2, 1.2))

        self.idx = self.idx + 1
        return self.im, self.plt1, self.plt2


def visualize(score_filename, video_filename, output_filename):
    fin = open(score_filename, 'rb')
    preds = pickle.load(fin) + [0] * 16
    frames = len(preds)

    cap = cv2.VideoCapture(video_filename)

    output_fn = output_filename

    # should be a list of the files for the video, for now I just
    # duplicate a single image
    
    files = None
    #files = np.asarray((['img.jpg']) * frames)

    # these are the predictions from the neural network
    #preds = np.random.random(size=(frames, ))

    fig = plt.figure(figsize=(8, 10), dpi=80)
    anim = SubplotAnimation(fig, files, preds, cap)
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
model.fc = nn.Linear(model.fc.in_features, 2)
#load 20
model_path = f'model/start_resnet_18_xentropy_epoch_18.pth'
model.load_state_dict(torch.load(model_path))
model.cuda()
torch.no_grad()
model.eval()
    
    
test_video_prefix = 'raw/'
cache_video_prefix = 'cache_test_start/'

test_filenames = os.listdir(test_video_prefix)
for test_video_file in test_filenames:
    test_name = test_video_file.split('.')[0]

    output_file = f'{cache_video_prefix}{test_video_file}'
    cmd = f'ffmpeg -y -i {test_video_prefix}{test_video_file} -vf "fps=fps=8" -s 112x112 -aspect 1 -vcodec libx264 -an {output_file}'
    os.system(cmd)

    vis_file = f'cache_vis_start/{test_video_file}'
    cmd = f'ffmpeg -y -i {test_video_prefix}{test_video_file} -vf "fps=fps=8" -s 1080x720 -vcodec libx264 -an {vis_file}'
    os.system(cmd)


    frames = torchvision.io.read_video(output_file, pts_unit='sec')[0].to(device)
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]

    Y_ary = []
    for i in range(0, len(frames)-16):
        i_end = i+16

        X = torch.unsqueeze(frames[i:i_end], 0)
        X = (X / 255.0)

        for z in range(3):
            X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

        X = X.permute(0, 4, 2, 3, 1)
        emb, output = model(X)

        prob = torch.nn.functional.softmax(output)
        Y_ary.append(prob.cpu().detach().numpy()[0][1])

    score_filename = f'scores_start/{test_name}.pk'
    with open(score_filename, 'wb') as fout:
        pickle.dump(Y_ary, fout)

    '''
    score_filename = f'scores/{test_name}.pk'
    with open(score_filename, 'rb') as fout:
        Y_ary = pickle.load(fout)

    Y_ary = [0] * 16 + Y_ary
    Y_ary = [float(x) for x in Y_ary]

    json_filename = f'scores/{test_name}.json'
    with open(json_filename, 'w') as fout:
        json.dump(Y_ary, fout)
    
    continue
    '''
    video_filename = vis_file
    output_filename = f'scores_vis_start/{test_name}.mp4'
    visualize(score_filename, video_filename, output_filename)
    #executor.submit(visualize, score_filename, video_filename, output_filename)
    
print('All task has generated')
executor.shutdown()
print('All done')