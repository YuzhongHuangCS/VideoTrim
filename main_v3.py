import torch
import torchvision
import torchvision.io
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
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
import argparse
import logging
import collections
from losses import SupConLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
log = logging.getLogger(__name__)

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self):
        pass

    def __call__(self, x):
        return [x, torch.transpose(x, 1, 2)]

tsfm = TwoCropTransform()

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
                log.debug(f'1, {data.shape}, {video_path}')
                if len(data) < 16:
                    #os.remove(video_path)
                    log.warning(f'Less than 16 frames: {video_path}')
                    
                    return self.__getitem__(max(idx-1, 0))
                else:
                    return [tsfm(data[0:16, :, :, :]), 1]
            else:
                log.warning(f'Missing Vidoe file: {video_path}')
                return self.__getitem__(max(idx-1, 0))
        else:
            if os.path.exists(video_path):
                json_path = video_path.replace('.mp4', '.json')
                duration = json.loads(open(json_path).read())['duration']

                end = random.uniform(2.2, duration-0.2)
                start = end - 2
                data = torchvision.io.read_video(video_path, start, end, 'sec')[0]
                log.debug(f'0, {data.shape}, {video_path}, {start}, {end}, {duration}')
                if len(data) < 16:
                    #os.remove(video_path)
                    log.warning(f'Less than 16 frames: {video_path}')
                    return self.__getitem__(min(idx+1, len(self.list)-1))
                else:
                    return [tsfm(data[0:16, :, :, :]), 0]
            else:
                log.warning(f'Missing Vidoe file: {video_path}')
                return self.__getitem__(min(idx+1, len(self.list)-1))

def flatten_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def format_video(l, prefix):
    return [f'{prefix}{e}.mp4' for e in l]


def main(args):
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

    names_common = sorted(list(set(pos_dict.keys()).intersection(set(neg_dict.keys()))))
    n_valid = args.n_valid
    n_test = args.n_test
    
    random.seed(args.seed)
    print('names_common len', len(names_common))

    random.shuffle(names_common)
    names_valid = set(names_common[0:n_valid])
    names_test = set(names_common[n_valid:(n_valid+n_test)])
    names_valid_test = names_valid.union(names_test)
    
    pos_valid = format_video(flatten_nested_list([v for k, v in pos_dict.items() if k in names_valid]), 'cache_large/pos_')
    neg_valid = format_video(flatten_nested_list([v for k, v in neg_dict.items() if k in names_valid]), 'cache_large/neg_')
    pos_test = format_video(flatten_nested_list([v for k, v in pos_dict.items() if k in names_test]), 'cache_large/pos_')
    neg_test = format_video(flatten_nested_list([v for k, v in neg_dict.items() if k in names_test]), 'cache_large/neg_')

    pos_train = format_video(flatten_nested_list([v for k, v in pos_dict.items() if k not in names_valid_test]), 'cache_large/pos_')
    neg_train = format_video(flatten_nested_list([v for k, v in neg_dict.items() if k not in names_valid_test]), 'cache_large/neg_')

    batch_size = args.batch_size

    use_triplet = args.loss in ('triplet', 'triplet_contrastive')
    use_contrastive = args.loss in ('contrastive', 'triplet_contrastive')

    train_dataset = VideoDataset(pos_train, neg_train)
    valid_dataset = VideoDataset(pos_valid, neg_valid)
    test_dataset = VideoDataset(pos_test, neg_test)

    n_train_pos = len(pos_train)
    n_train_neg = len(neg_train)
    n_valid_pos = len(pos_valid)
    n_valid_neg = len(neg_valid)
    n_test_pos = len(pos_test)
    n_test_neg = len(neg_test)
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
    #model_path = f'model/resnet_18_contrastive_epoch_15.pth'
    #model.load_state_dict(torch.load(model_path))
    model.cuda()
    #pdb.set_trace()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer2 = optim.Adam(model.fc.parameters(), lr=args.lr_finetune)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]
    d = nn.PairwiseDistance(p=2)

    n_epoch = args.n_epoch
    criterion_con = SupConLoss(temperature=0.07)
    for epoch in range(n_epoch):
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
            X = torch.cat(X, dim=0)
            
            X = X.to(device)    
            Y = Y.to(device)
            X = (X / 255.0)

            for z in range(3):
                X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

            X = X.permute(0, 4, 2, 3, 1)
            optimizer.zero_grad()
            emb_raw, output = model(X)
            loss = None

            if args.loss_method == 'joint' or epoch < args.n_epoch_pretrain:
                emb = F.normalize(emb_raw, dim=1)
                if args.loss_method == 'joint':
                    loss = args.xentropy_weight * criterion(output, Y)
                else:
                    loss = 0

                if use_triplet:
                    emb_1 = emb[Y==1]
                    emb_0 = emb[Y==0]
                    n_group = min(len(emb_1) // 2, len(emb_0))

                    if n_group > 0:
                        emb_a = emb_1[0:n_group]
                        emb_p = emb_1[n_group:n_group+n_group]
                        emb_n = emb_0[0:n_group]

                        distance = d(emb_a, emb_p) - d(emb_a, emb_n) + 0.05
                        distance_flip = d(emb_p, emb_a) - d(emb_p, emb_n) + 0.05
                        loss_triplet = torch.mean(F.relu(distance)) + torch.mean(F.relu(distance_flip))
                        loss += args.triplet_weight * loss_triplet
                    else:
                        loss += 0
                    
                    n_group2 = min(len(emb_1), len(emb_0) // 2)
                    if n_group2 > 0:
                        emb_a2 = emb_0[0:n_group2]
                        emb_p2 = emb_0[n_group2:n_group2+n_group2]
                        emb_n2 = emb_1[0:n_group2]

                        distance2 = d(emb_a2, emb_p2) - d(emb_a2, emb_n2) + 0.05
                        distance2_flip = d(emb_p2, emb_a2) - d(emb_p2, emb_n2) + 0.05
                        loss_triplet2 = torch.mean(F.relu(distance2)) + torch.mean(F.relu(distance2_flip))
                        loss += args.triplet_weight * loss_triplet2
                    else:
                        loss += 0

                if use_contrastive:
                    bsz = Y.shape[0]
                    f1, f2 = torch.split(emb, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    loss += args.contrastive_weight * criterion_con(features, Y)
                if loss == 0:
                    continue

                loss.backward()
                optimizer.step()
            else:
                loss = criterion(output, Y)
                loss.backward()
                optimizer2.step()

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
            X = X[0]
            
            X = X.to(device)
            Y = Y.to(device)
            X = (X / 255.0)

            for z in range(3):
                X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

            X = X.permute(0, 4, 2, 3, 1)
            emb_raw, output = model(X)
            
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

        if epoch % args.test_interval == 0 or epoch == (n_epoch-1):
            test_acc_ary = []
            Y_true_ary = []
            Y_pred_ary = []
            for i_batch, sample in enumerate(test_loader):
                X, Y = sample
                X = X[0]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Auto Trim")
    parser.add_argument('-loss', default='xentropy', choices=('xentropy', 'triplet', 'contrastive', 'triplet_contrastive'))
    parser.add_argument('-loss_method', default='joint', choices=('joint', 'separate'))
    parser.add_argument('-xentropy_weight', type=float, default=1.0)
    parser.add_argument('-triplet_weight', type=float, default=1.0)
    parser.add_argument('-contrastive_weight', type=float, default=1.0)
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-lr_finetune', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=0.07)
    parser.add_argument('-seed', default=1234, type=int, help='random seed')
    parser.add_argument('-gpu', default='0', help='which gpu')
    
    parser.add_argument('-n_epoch', default=100, type=int, help='n_epoch')
    parser.add_argument('-n_epoch_pretrain', default=80, type=int, help='n_epoch_pretrain')
    parser.add_argument('-test_interval', default=5, type=int, help='test_interval')
    parser.add_argument('-n_valid', default=512, type=int, help='n_valid')
    parser.add_argument('-n_test', default=512, type=int, help='n_test')
    parser.add_argument('-batch_size', default=64, type=int, help='batch_size')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(args)
    main(args)
