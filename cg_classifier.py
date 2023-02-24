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
import numpy as np
import time
import os
import gc
import sklearn.metrics
import collections
from PIL import Image
from torchvision import transforms
import sklearn.svm
import pickle

use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda")
#device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

#pos is natural, neg is unnatural
class VideoDataset(Dataset):
    def __init__(self, pos_list, neg_list):
        super(VideoDataset, self).__init__()
        self.pos_list = pos_list
        self.neg_list = neg_list
        self.list = pos_list + neg_list
        self.preprocess = transforms.Compose([
            transforms.Resize([960, 540]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        if idx < len(self.pos_list):
            return [self.preprocess(Image.open(self.list[idx])), 1]
        else:
            return [self.preprocess(Image.open(self.list[idx])), 0]
batch_size = 32

pos_all = glob.glob('natural/*')
neg_all = glob.glob('unnatural/*')
train_dataset = VideoDataset(pos_all, neg_all)
train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=8, shuffle=False)

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
model.fc = nn.Identity()
model.cuda()

svm = sklearn.svm.SVC(C=2)

emb_ary = []
Y_ary = []
n_ary = []
for i_batch, sample in enumerate(train_loader):
    X, Y = sample
    print(X.shape)
    n_ary.append(len(X))
    X = X.to(device)
    emb = model(X)
    emb_ary.append(emb.cpu().detach().numpy())
    Y_ary.append(Y.numpy())

print(sum(n_ary))
print(len(train_dataset))
exit()
emb_ary = np.concatenate(emb_ary)
Y_ary = np.concatenate(Y_ary)
svm.fit(emb_ary, Y_ary)
print('svm score', svm.score(emb_ary, Y_ary))
#exit()
with open('svm.pickle', 'wb') as fout:
    pickle.dump(svm, fout)

#with open('svm.pickle', 'rb') as fin:
#    svm = pickle.load(fin)

unknown_all = glob.glob('cache_jpg2/*')
test_datasest = VideoDataset(unknown_all, [])
test_loader = DataLoader(test_datasest, batch_size = batch_size, num_workers=8, shuffle=False)
test_counter = 0
for i_batch, sample in enumerate(test_loader):
    X, Y = sample
    X = X.to(device)
    emb = model(X)
    emb_numpy = emb.detach().cpu().numpy()
    y_pred = svm.predict(emb_numpy)
    for j, l in enumerate(y_pred.tolist()):
        filename = unknown_all[test_counter]
        if l == 1:
            os.rename(filename, filename.replace('cache_jpg2', 'pred_n'))
        else:
            os.rename(filename, filename.replace('cache_jpg2', 'pred_un'))

        print(filename)
        test_counter += 1
print('123')

