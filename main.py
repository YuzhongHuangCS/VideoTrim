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

use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda")
#device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

def flatten_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def format_video(l, prefix):
    return [f'{prefix}{e}.mp4' for e in l]


class VideoDataset(Dataset):
    def __init__(self, pos_list, neg_list):
        super(VideoDataset, self).__init__()
        self.pos_list = pos_list
        self.neg_list = neg_list
        self.list = pos_list + neg_list

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        video_path = self.list[idx]
        if idx < len(self.pos_list):
            if os.path.exists(video_path):
                data = torchvision.io.read_video(video_path, pts_unit='sec')[0]
                #print(1, data.shape, video_path, start, end, duration)
                if len(data) < 16:
                    os.remove(video_path)
                    return self.__getitem__(max(idx-1, 0))
                else:
                    return [data[0:16, :, :, :], 1]
            else:
                return self.__getitem__(max(idx-1, 0))
        else:
            if os.path.exists(video_path):
                json_path = video_path.replace('.mp4', '.json')
                duration = json.loads(open(json_path).read())['duration']

                end = random.uniform(2.2, duration-0.2)
                start = end - 2
                data = torchvision.io.read_video(video_path, start, end, 'sec')[0]
                #print(0, data.shape, video_path, start, end, duration)
                if len(data) < 16:
                    os.remove(video_path)
                    return self.__getitem__(min(idx+1, len(self.list)-1))
                else:
                    return [data[0:16, :, :, :], 0]
            else:
                return self.__getitem__(min(idx+1, len(self.list)-1))

pos_dict = collections.defaultdict(list)
pos_json = json.loads(open('map_data/pos_map.json').read())
for key, value in pos_json.items():
    [source, name, end_time] = value
    pos_dict[name].append(key)

neg_dict = collections.defaultdict(list)
neg_json = json.loads(open('map_data/neg_map.json').read())
for key, value in neg_json.items():
    [source, name, start, end] = value
    neg_dict[name].append(key)

import pickle
with open('remove_set.pickle', 'rb') as fin:
    remove_set = pickle.load(fin)

names_common = sorted(list(set(pos_dict.keys()).intersection(set(neg_dict.keys()))))
batch_size = 64
n_valid = 128
n_test = 128

random.seed(1234)
print('names_common len', len(names_common))
random.shuffle(names_common)
names_valid = set(names_common[0:n_valid])
names_test = set(names_common[n_valid:(n_valid+n_test)])
names_valid_test = names_valid.union(names_test)
pos_valid = format_video(flatten_nested_list([v for k, v in pos_dict.items() if k in names_valid and k not in remove_set]), 'cache/pos_')
neg_valid = format_video(flatten_nested_list([v for k, v in neg_dict.items() if k in names_valid and k not in remove_set]), 'cache/neg_')
pos_test = format_video(flatten_nested_list([v for k, v in pos_dict.items() if k in names_test and k not in remove_set]), 'cache/pos_')
neg_test = format_video(flatten_nested_list([v for k, v in neg_dict.items() if k in names_test and k not in remove_set]), 'cache/neg_')

pos_train = format_video(flatten_nested_list([v for k, v in pos_dict.items() if k not in names_valid_test and k not in remove_set]), 'cache/pos_')
neg_train = format_video(flatten_nested_list([v for k, v in neg_dict.items() if k not in names_valid_test and k not in remove_set]), 'cache/neg_')

n_train_pos = len(pos_train)
n_train_neg = len(neg_train)
n_valid_pos = len(pos_valid)
n_valid_neg = len(neg_valid)
n_test_pos = len(pos_test)
n_test_neg = len(neg_test)

use_triplet = False
use_contrastive = False

train_dataset = VideoDataset(pos_train, neg_train)
valid_dataset = VideoDataset(pos_valid, neg_valid)
test_dataset = VideoDataset(pos_test, neg_test)

n_train_pos = len(pos_train)
n_train_neg = len(neg_train)
weights = [1.0/n_train_pos] * n_train_pos + [1.0/n_train_neg] * n_train_neg
sampler = WeightedRandomSampler(weights, len(weights))

weights_valid = [1.0/n_valid_pos] * n_valid_pos + [1.0/n_valid_neg] * n_valid_neg
sampler_valid = WeightedRandomSampler(weights_valid, len(weights_valid))

weights_test = [1.0/n_test_pos] * n_test_pos + [1.0/n_test_neg] * n_test_neg
sampler_test = WeightedRandomSampler(weights_test, len(weights_test))

train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=6, pin_memory=True, sampler=sampler)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, num_workers=6, pin_memory=True, sampler=sampler_valid)
test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=6, pin_memory=True, sampler=sampler_test)

print(f'#Train P/N: {n_train_pos}/{n_train_neg}, #Valid P/N: {len(pos_valid)}/{len(neg_valid)}, #Test P/N: {len(pos_test)}/{len(neg_test)}')

pretrain = torch.load('model/r3d18_K_200ep.pth', map_location='cuda')
model = resnet.generate_model(model_depth=18, n_classes=700)
model.load_state_dict(pretrain['state_dict'])
model.fc = nn.Linear(model.fc.in_features, 2)
#load 20
model_path = f'model/resnet_18_contrastive_epoch_99.pth'
model.load_state_dict(torch.load(model_path))
model.cuda()
#pdb.set_trace()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.AdamW(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
mean = [0.4345, 0.4051, 0.3775]
std = [0.2768, 0.2713, 0.2737]


for epoch in range(100):
    loss_ary = []
    acc_ary = []
    fetch_time_ary = []
    model_time_ary = []
    last_time = time.time()
    torch.enable_grad()
    model.train()
    for i_batch, sample in enumerate(train_loader):
        new_time = time.time()
        fetch_time_ary.append(new_time - last_time)
        X, Y = sample
        X = X.to(device)
        Y = Y.to(device)
        X = (X / 255.0)

        for z in range(3):
            X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

        X = X.permute(0, 4, 2, 3, 1)
        optimizer.zero_grad()
        emb, output = model(X)
        loss = criterion(output, Y)
        
        if use_triplet:
            emb_1 = emb[Y==1]
            emb_0 = emb[Y==0]
            n_group = min(len(emb_1) // 2, len(emb_0))

            emb_a = emb_1[0:n_group]
            emb_p = emb_1[n_group:n_group+n_group]
            emb_n = emb_0[0:n_group]

            d = nn.PairwiseDistance(p=2)
            distance = d(emb_a, emb_p) - d(emb_a, emb_n) + 0.5
            loss_triplet = torch.mean(torch.max(distance, torch.zeros_like(distance)))
            loss += loss_triplet

        if use_contrastive:
            emb_1 = emb[Y==1]
            emb_0 = emb[Y==0]
            n_group = min(len(emb_1) // 2, len(emb_0) // 2)

            emb_a = emb_1[0:n_group]
            emb_p = emb_1[n_group:n_group+n_group]
            emb_n1 = emb_0[0:n_group]
            emb_n2 = emb_0[n_group:n_group+n_group]

            d = nn.PairwiseDistance(p=2)
            distance_matrix = torch.stack([d(emb_a, emb_p), d(emb_a, emb_n1), d(emb_a, emb_n2)], dim=1)
            exp = torch.exp(distance_matrix)
            loss_contrastive = torch.mean(-torch.log(exp[:, 0]/torch.sum(exp, dim=1)))
            loss += loss_contrastive

        loss.backward()
        optimizer.step()

        Y_pred = np.argmax(output.cpu().detach().numpy(), axis=1)
        acc = np.mean(Y_pred == (Y.cpu().detach().numpy()))
        #print('\t', i_batch, loss.item(), acc)
        loss_ary.append(loss.item())
        acc_ary.append(acc)
        last_time = time.time()
        model_time_ary.append(last_time - new_time)

    del X, Y
    gc.collect()
 
    torch.no_grad()
    model.eval()
    valid_acc_ary = []
    valid_loss_ary = []
    for i_batch, sample in enumerate(valid_loader):
        X, Y = sample
        X = X.to(device)
        Y = Y.to(device)
        X = (X / 255.0)

        for z in range(3):
            X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

        X = X.permute(0, 4, 2, 3, 1)
        emb, output = model(X)
        loss = criterion(output, Y)

        Y_pred = np.argmax(output.cpu().detach().numpy(), axis=1)
        Y_cpu = Y.cpu().detach().numpy()

        acc = np.mean(Y_pred == Y_cpu)
        valid_loss_ary.append(loss.item())
        valid_acc_ary.append(acc)

    valid_loss = np.mean(valid_loss_ary)
    scheduler.step(valid_loss)
    print(f'Epoch: {epoch}, LR: {optimizer.param_groups[0]["lr"]}')
    del X, Y
    gc.collect()
    print(f'Epoch: {epoch}, Train Loss: {np.mean(loss_ary)}, Train Acc: {np.mean(acc_ary)}, Valid Loss: {valid_loss}, Valid Acc: {np.mean(valid_acc_ary)}, Avg fetch time: {np.mean(fetch_time_ary)}, Avg model time: {np.mean(model_time_ary)}')
     
    if True:
        test_acc_ary = []
        Y_true_ary = []
        Y_pred_ary = []
        for i_batch, sample in enumerate(test_loader):
            X, Y = sample
            Y_true_ary += Y
            X = X.to(device)
            Y = Y.to(device)
            X = (X / 255.0)

            for z in range(3):
                X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

            X = X.permute(0, 4, 2, 3, 1)
            emb, output = model(X)

            Y_pred = np.argmax(output.cpu().detach().numpy(), axis=1)
            Y_cpu = Y.cpu().detach().numpy()

            acc = np.mean(Y_pred == Y_cpu)
            #pdb.set_trace()
            test_acc_ary.append(acc)
            Y_pred_ary += list(Y_pred)
        del X, Y
        print(f'Epoch: {epoch}, Test Acc: {np.mean(test_acc_ary)}')
        print(sklearn.metrics.confusion_matrix(Y_true_ary, Y_pred_ary))
        torch.save(model.state_dict(), f'model/resnet_18_contrastive_epoch_{epoch}.pth')
        gc.collect()

#pdb.set_trace()
print('Training done')

pdb.set_trace()
print('Before Exit')