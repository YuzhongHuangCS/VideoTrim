import torch
import torchvision
import torchvision.io
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
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

class VideoDataset(Dataset):
    def __init__(self, start_list, end_list, neg_list):
        super(VideoDataset, self).__init__()
        self.start_list = start_list
        self.end_list = end_list
        self.neg_list = neg_list
        self.list = start_list + end_list + neg_list

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        video_path = self.list[idx]
        if idx < len(self.start_list):
            if os.path.exists(video_path):
                data = torchvision.io.read_video(video_path, pts_unit='sec')[0]
                log.debug(f'2, {data.shape}, {video_path}')
                if len(data) < 16:
                    #os.remove(video_path)
                    log.warning(f'Less than 16 frames: {video_path}')
                    return self.__getitem__(max(idx-1, 0))
                else:
                    return [data[0:16, :, :, :], 2]
            else:
                log.warning(f'Missing Vidoe file: {video_path}')
                return self.__getitem__(max(idx-1, 0))
        elif idx < (len(self.start_list) + len(self.end_list)):
            if os.path.exists(video_path):
                data = torchvision.io.read_video(video_path, pts_unit='sec')[0]
                log.debug(f'1, {data.shape}, {video_path}')
                if len(data) < 16:
                    #os.remove(video_path)
                    log.warning(f'Less than 16 frames: {video_path}')
                    return self.__getitem__(max(idx-1, len(self.start_list)))
                else:
                    return [data[0:16, :, :, :], 1]
            else:
                log.warning(f'Missing Vidoe file: {video_path}')
                return self.__getitem__(max(idx-1, len(self.start_list)))
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
                    return [data[0:16, :, :, :], 0]
            else:
                log.warning(f'Missing Vidoe file: {video_path}')
                return self.__getitem__(min(idx+1, len(self.list)-1))

def flatten_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def format_video(l, prefix):
    return [f'{prefix}{e}.mp4' for e in l]

def main(args):
    dataset_folder = 'cache_exa'
    names_common = set()
    names_all = os.listdir(dataset_folder)
    names_all_parts = []
    for name in names_all:
        if 'youtube' in name:
            if '.mp4' in name:
                parts = name.split('_')
                name_id = parts[1] + '_' + parts[2]
                names_common.add(name_id)

                parts[-1] = int(parts[-1].split('.')[0])
                names_all_parts.append(parts)

    names_common = sorted(list(names_common))
    names_all_parts = sorted(names_all_parts)

    n_valid = args.n_valid
    n_test = args.n_test
    random.seed(args.seed)
    random.shuffle(names_common)                  
    print('names_common len', len(names_common))

    names_valid = set(names_common[0:n_valid])
    names_test = set(names_common[n_valid:(n_valid+n_test)])
    names_valid_test = names_valid.union(names_test)
    names_train = set(names_common) - names_valid_test

    with open('split_config_2way.json', 'w') as fout:
        split = {
            'train': list(names_train),
            'valid': list(names_valid),
            'test': list(names_test)
        }
        json.dump(split, fout)
 
    start_train = []
    end_train = []
    neg_train = []
    start_valid = []
    end_valid = []
    neg_valid = []
    start_test = []
    end_test = []
    neg_test = []
    
    for parts in names_all_parts:
        name_id = parts[1] + '_' + parts[2]
        parts[-1] = str(parts[-1])
        name = dataset_folder + '/' + '_'.join(parts) + '.mp4'

        if name_id in names_valid:
            if 'start' in name:
                start_valid.append(name)
            elif 'end' in name:
                end_valid.append(name)
            else:
                assert 'neg' in name
                neg_valid.append(name)
        elif name_id in names_test:
            if 'start' in name:
                start_test.append(name)
            elif 'end' in name:
                end_test.append(name)
            else:
                assert 'neg' in name
                neg_test.append(name)
        else:
            if 'start' in name:
                start_train.append(name)
            elif 'end' in name:
                end_train.append(name)
            else:
                assert 'neg' in name
                neg_train.append(name)

    batch_size = args.batch_size
 
    train_dataset = VideoDataset(start_train, end_train, neg_train)
    valid_dataset = VideoDataset(start_valid, end_valid, neg_valid)
    test_dataset = VideoDataset(start_test, end_test, neg_test)

    n_train_start = len(start_train)
    n_train_end = len(end_train)
    n_train_neg = len(neg_train)

    n_valid_start = len(start_valid)
    n_valid_end = len(end_valid)
    n_valid_neg = len(neg_valid)

    n_test_start = len(start_test)
    n_test_end = len(end_test)
    n_test_neg = len(neg_test)

    weights = [1.0/n_train_start] * n_train_start + [1.0/n_train_end] * n_train_end + [1.0/n_train_neg] * n_train_neg
    sampler = WeightedRandomSampler(weights, len(weights))

    weights_valid = [1.0/n_valid_start] * n_valid_start + [1.0/n_valid_end] * n_valid_end + [1.0/n_valid_neg] * n_valid_neg
    sampler_valid = WeightedRandomSampler(weights_valid, len(weights_valid))

    weights_test = [1.0/n_test_start] * n_test_start + [1.0/n_test_end] * n_test_end + [1.0/n_test_neg] * n_test_neg
    sampler_test = WeightedRandomSampler(weights_test, len(weights_test))
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=6, pin_memory=True, sampler=sampler)
    #train_loader = DataLoader(train_dataset, num_workers=6, pin_memory=True, batch_sampler=sampler_train)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, num_workers=6, pin_memory=True, sampler=sampler_valid)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=6, pin_memory=True, sampler=sampler_test)

    print(f'#Train S/E/N: {n_train_start}/{n_train_end}/{n_train_neg}, #Valid S/E/N: {n_valid_start}/{n_valid_end}/{n_valid_neg}, #Test S/E/N: {n_test_start}/{n_test_end}/{n_test_neg}')

    pretrain = torch.load('model/r3d18_K_200ep.pth', map_location='cuda')
    model = resnet.generate_model(model_depth=18, n_classes=700)
    model.load_state_dict(pretrain['state_dict'])
    model.fc = nn.Linear(model.fc.in_features, 3)
    #load 20
    model_path = f'model/exa_yt_resnet_18_xentropy_epoch_19.pth'
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    
    #pdb.set_trace()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]

    n_epoch = args.n_epoch
    criterion_con = SupConLoss(temperature=0.07)
    for epoch in range(n_epoch):
        #sampler_train.current_epoch = epoch
        
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
            #X = torch.cat(X, dim=0)
            
            X = X.to(device)    
            Y = Y.to(device)
            X = (X / 255.0)

            for z in range(3):
                X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

            X = X.permute(0, 4, 1, 2, 3)
            #X = X.permute(0, 4, 2, 3, 1)
            optimizer.zero_grad()
            emb_raw, output = model(X)
            
            #start->2, end->1, neg->0
            output_start = output[:, [0, 2]][Y != 1]
            Y_start = Y[Y != 1] // 2
            
            output_end = output[:, [0, 1]][Y != 2]
            Y_end = Y[Y != 2]
            
            #print(output_start.shape, Y_start.shape, output_end.shape, Y_end.shape)
            loss = criterion(output, Y) + criterion(output_start, Y_start) + criterion(output_end, Y_end)
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
            #X = X[0]
            
            X = X.to(device)
            Y = Y.to(device)
            X = (X / 255.0)

            for z in range(3):
                X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

            X = X.permute(0, 4, 1, 2, 3)
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
            Y_start_true_ary = []
            Y_end_true_ary = []
            Y_start_ary = []
            Y_end_ary = []
            Y_end_prob_ary = []
            for i_batch, sample in enumerate(test_loader):
                X, Y = sample
                #X = X[0]
                Y_start_true_ary += (Y.numpy() == 2).astype(int).tolist()
                Y_end_true_ary += (Y.numpy() == 1).astype(int).tolist()
    
                X = X.to(device)
                Y = Y.to(device)
                X = (X / 255.0)

                for z in range(3):
                    X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

                X = X.permute(0, 4, 1, 2, 3)
                emb_raw, output = model(X)

                start_prob_ary = torch.nn.functional.softmax(output[:, [0,2]]).cpu().detach().numpy()
                Y_start_ary += (start_prob_ary[:, 1] >= 0.5).astype(int).tolist()
            
                end_prob_ary = torch.nn.functional.softmax(output[:, 0:2]).cpu().detach().numpy()
                Y_end_ary += (end_prob_ary[:, 1] >= 0.5).astype(int).tolist()
                Y_end_prob_ary += end_prob_ary[:, 1].tolist()

            del X, Y
            print(f'Epoch: {epoch}, Start Test Acc: {sklearn.metrics.accuracy_score(Y_start_true_ary, Y_start_ary)}')
            print(sklearn.metrics.confusion_matrix(Y_start_true_ary, Y_start_ary))
            
            print(f'Epoch: {epoch}, End Test Acc: {sklearn.metrics.accuracy_score(Y_end_true_ary, Y_end_ary)}')
            print(sklearn.metrics.confusion_matrix(Y_end_true_ary, Y_end_ary))
              
            torch.save(model.state_dict(), f'model/exa_yt_resnet_18_xentropy_epoch_{epoch}.pth')

    #pdb.set_trace()
    print('Training done')

    #pdb.set_trace()
    print('Before Exit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Auto Trim")
    #parser.add_argument('-loss', default='contrastive', choices=('xentropy', 'triplet', 'contrastive', 'triplet_contrastive'))
    #parser.add_argument('-loss_method', default='separate', choices=('joint', 'separate'))
    #parser.add_argument('-xentropy_weight', type=float, default=1.0)
    #parser.add_argument('-triplet_weight', type=float, default=1.0)
    #parser.add_argument('-contrastive_weight', type=float, default=1.0)
    parser.add_argument('-lr', type=float, default=0.01)
    #parser.add_argument('-lr_finetune', type=float, default=0.1)
    #parser.add_argument('-temperature', type=float, default=0.07)
    parser.add_argument('-seed', default=1234, type=int, help='random seed')
    parser.add_argument('-gpu', default='0', help='which gpu')
    
    parser.add_argument('-n_epoch', default=100, type=int, help='n_epoch')
    #parser.add_argument('-n_epoch_pretrain', default=20, type=int, help='n_epoch_pretrain')
    parser.add_argument('-test_interval', default=1, type=int, help='test_interval')
    parser.add_argument('-n_valid', default=128, type=int, help='n_valid')
    parser.add_argument('-n_test', default=128, type=int, help='n_test')
    parser.add_argument('-batch_size', default=64, type=int, help='batch_size')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(args)
    main(args)
