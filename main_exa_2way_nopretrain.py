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

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self):
        pass

    def __call__(self, x):
        return [x, torch.flip(x, (2, ))]

tsfm = TwoCropTransform()

class VideoDataset(Dataset):
    def __init__(self, end_list, neg_list):
        super(VideoDataset, self).__init__()
        self.end_list = end_list
        self.neg_list = neg_list
        self.list = end_list + neg_list

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        video_path = self.list[idx]
        if idx < len(self.end_list):
            if os.path.exists(video_path):
                data = torchvision.io.read_video(video_path, pts_unit='sec')[0]
                log.debug(f'1, {data.shape}, {video_path}')
                if len(data) < 16:
                    #os.remove(video_path)
                    log.warning(f'Less than 16 frames: {video_path}')
                    return self.__getitem__(max(idx-1, 0))
                else:
                    return [data[0:16, :, :, :], 1]
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
                    return [data[0:16, :, :, :], 0]
            else:
                log.warning(f'Missing Vidoe file: {video_path}')
                return self.__getitem__(min(idx+1, len(self.list)-1))

def flatten_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def format_video(l, prefix):
    return [f'{prefix}{e}.mp4' for e in l]

class VideoSampler(Sampler):
    def __init__(self, data, pos_dict, neg_dict, pos_train, neg_train, batch_size, names_train):
        self.num_samples = len(data) // batch_size
        self.pos_dict = pos_dict
        self.neg_dict = neg_dict
        self.pos_names = list(pos_dict.keys())
        self.neg_names = list(neg_dict.keys())
        self.batch_size = batch_size
        self.names_train = names_train
        self.total_epoch = 0
        self.current_epoch = 0
        
        cache_dict = {}
        for i, v in enumerate(pos_train):
            key = v.split('/')[1].split('.')[0]
            cache_dict[key] = i
        n_pos = len(pos_train)
        for i, v in enumerate(neg_train):
            key = v.split('/')[1].split('.')[0]
            cache_dict[key] = i + n_pos
        self.cache_dict = cache_dict

    def positive_hard(self, c, select=None):
        if select == None:
            select = random.sample(self.names_train, c)
        pos_idx = [self.cache_dict['pos_' + random.choice(self.pos_dict[n])] for n in select]
        return pos_idx
    
    def positive_easy(self, c, select=None):
        if select == None:
            select = random.sample(self.names_train, c)
        pos_idx = []
        for n in select:
            for v in self.pos_dict[n]:
                pos_idx.append(self.cache_dict['pos_' + v])
                if len(pos_idx) == c:
                    return pos_idx
    
    def negative_easy(self, c, select=None):
        if select == None:
            select = random.sample(self.names_train, c)
        neg_idx = [self.cache_dict['neg_' + random.choice(self.neg_dict[n])] for n in select]
        return neg_idx
        
    def positive_hard_negative_easy(self, c):
        return self.positive_hard(c//2) + self.negative_easy(c//2)

    def positive_easy_negative_easy(self, c):
        return self.positive_easy(c//2) + self.negative_easy(c//2)

    def positive_hard_negative_medium(self, c):
        select = random.sample(self.names_train, c//2)
        return self.positive_hard(c//2, select) + self.negative_easy(c//2, select)
    
    def positive_easy_negative_medium(self, c):
        select = random.sample(self.names_train, c//2)
        return self.positive_easy(c//2, select) + self.negative_easy(c//2, select)
    
    # currently use index based approach
    def positive_hard_negative_hard(self, c):
        select = random.sample(self.names_train, c//2)
        pos_idx = []
        neg_idx = []
        for n in select:
            n_pos = len(self.pos_dict[n])
            n_neg = len(self.neg_dict[n])
            
            i = random.randint(0, min(n_pos, n_neg))
            pos_idx.append(self.pos_dict[n][i])
            neg_idx.append(self.neg_dict[n][i])
        return pos_idx + neg_idx
        
    def __iter__(self):
        '''
        0-25% epoch:
            p_easy_n_easy: 25%
            p_hard_n_easy: 25%
            p_easy_n_medium: 25%
            p_hard_n_medium: 25%
        25-50% epoch:
            p_easy_n_easy: 25%
            p_hard_n_easy: 25%
            p_hard_n_medium: 25%
            p_hard_n_hard: 25%
        50-75% epoch:
            p_hard_n_easy: 25%
            p_hard_n_medium: 50%
            p_hard_n_hard: 25%
       75-100% epoch:
            p_hard_n_medium: 50%
            p_hard_n_hard: 50%
        '''
        '''
        if self.current_epoch <= self.total_epoch * 0.25:
            print('sampling mode 1')
            for i in range(self.num_samples):
                yield self.positive_easy_negative_easy(self.batch_size // 4) +\
                      self.positive_hard_negative_easy(self.batch_size // 4) +\
                      self.positive_easy_negative_medium(self.batch_size // 4) +\
                      self.positive_hard_negative_medium(self.batch_size // 4)
                    
        elif self.current_epoch <= self.total_epoch * 0.5:
            print('sampling mode 2')
            for i in range(self.num_samples):
                yield self.positive_easy_negative_easy(self.batch_size // 4) +\
                      self.positive_hard_negative_easy(self.batch_size // 4) +\
                      self.positive_hard_negative_medium(self.batch_size // 4) +\
                      self.positive_hard_negative_hard(self.batch_size // 4)
            
        elif self.current_epoch <= self.total_epoch * 0.75:
            print('sampling mode 3')
            for i in range(self.num_samples):
                yield self.positive_hard_negative_easy(self.batch_size // 4) +\
                      self.positive_hard_negative_medium(self.batch_size // 2) +\
                      self.positive_hard_negative_hard(self.batch_size // 4)
                
        else:
            print('sampling mode 4')
            for i in range(self.num_samples):
                yield self.positive_hard_negative_medium(self.batch_size // 2) +\
                      self.positive_hard_negative_hard(self.batch_size // 2)
        
        if self.current_epoch <= self.total_epoch * 0.5:
            print('sampling mode 3')
            for i in range(self.num_samples):
                yield self.positive_hard_negative_easy(self.batch_size // 4) +\
                      self.positive_hard_negative_medium(self.batch_size // 2) +\
                      self.positive_hard_negative_hard(self.batch_size // 4)
                
        else:
            print('sampling mode 4')
            for i in range(self.num_samples):
                yield self.positive_hard_negative_medium(self.batch_size // 2) +\
                      self.positive_hard_negative_hard(self.batch_size // 2)
        '''
        for i in range(self.num_samples):
            yield self.positive_hard_negative_medium(self.batch_size // 2)

    def __len__(self):
        return self.num_samples

def main(args):
    dataset_folder = 'cache_exa'
    names_common = set()
    names_all = os.listdir(dataset_folder)
    names_all_parts = []
    for name in names_all:
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

    with open('split_config.json', 'w') as fout:
        split = {
            'train': list(names_train),
            'valid': list(names_valid),
            'test': list(names_test)
        }
        json.dump(split, fout)
 

    end_train = []
    neg_train = []

    end_valid = []
    neg_valid = []

    end_test = []
    neg_test = []
    
    for parts in names_all_parts:
        name_id = parts[1] + '_' + parts[2]
        parts[-1] = str(parts[-1])
        name = dataset_folder + '/' + '_'.join(parts) + '.mp4'

        if name_id in names_valid:
            if 'start' in name:
                pass
            elif 'end' in name:
                end_valid.append(name)
            else:
                assert 'neg' in name
                neg_valid.append(name)
        elif name_id in names_test:
            if 'start' in name:
                pass
            elif 'end' in name:
                end_test.append(name)
            else:
                assert 'neg' in name
                neg_test.append(name)
        else:
            if 'start' in name:
                pass
            elif 'end' in name:
                end_train.append(name)
            else:
                assert 'neg' in name
                neg_train.append(name)

    batch_size = args.batch_size
 
    train_dataset = VideoDataset(end_train, neg_train)
    valid_dataset = VideoDataset(end_valid, neg_valid)
    test_dataset = VideoDataset(end_test, neg_test)

    n_train_end = len(end_train)
    n_train_neg = len(neg_train)

    n_valid_end = len(end_valid)
    n_valid_neg = len(neg_valid)

    n_test_end = len(end_test)
    n_test_neg = len(neg_test)

    weights = [1.0/n_train_end] * n_train_end + [1.0/n_train_neg] * n_train_neg
    sampler = WeightedRandomSampler(weights, len(weights))

    weights_valid = [1.0/n_valid_end] * n_valid_end + [1.0/n_valid_neg] * n_valid_neg
    sampler_valid = WeightedRandomSampler(weights_valid, len(weights_valid))

    weights_test = [1.0/n_test_end] * n_test_end + [1.0/n_test_neg] * n_test_neg
    sampler_test = WeightedRandomSampler(weights_test, len(weights_test))
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=6, pin_memory=True, sampler=sampler)
    #train_loader = DataLoader(train_dataset, num_workers=6, pin_memory=True, batch_sampler=sampler_train)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, num_workers=6, pin_memory=True, sampler=sampler_valid)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=6, pin_memory=True, sampler=sampler_test)

    print(f'#Train E/N: {n_train_end}/{n_train_neg}, #Valid E/N: {n_valid_end}/{n_valid_neg}, #Test E/N: {n_test_end}/{n_test_neg}')

    pretrain = torch.load('model/r3d18_K_200ep.pth', map_location='cuda')
    model = resnet.generate_model(model_depth=18, n_classes=700)
    #model.load_state_dict(pretrain['state_dict'])
    model.fc = nn.Linear(model.fc.in_features, 2)
    #load 20
    #model_path = f'model/resnet_18_xentropy_epoch_40.pth'
    #model.load_state_dict(torch.load(model_path))
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
            
            loss = criterion(output, Y)
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
            Y_true_ary = []
            Y_pred_ary = []
            for i_batch, sample in enumerate(test_loader):
                X, Y = sample
                #X = X[0]
                Y_true_ary += Y
                X = X.to(device)
                Y = Y.to(device)
                X = (X / 255.0)

                for z in range(3):
                    X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

                X = X.permute(0, 4, 1, 2, 3)
                emb_raw, output = model(X)

                Y_pred = np.argmax(output.cpu().detach().numpy(), axis=1)
                Y_cpu = Y.cpu().detach().numpy()

                acc = np.mean(Y_pred == Y_cpu)
                #pdb.set_trace()
                test_acc_ary.append(acc)
                Y_pred_ary += list(Y_pred)
            del X, Y
            print(f'Epoch: {epoch}, Test Acc: {np.mean(test_acc_ary)}')
            print(sklearn.metrics.confusion_matrix(Y_true_ary, Y_pred_ary))
            torch.save(model.state_dict(), f'model/exa_2way_nopretrain_resnet_18_xentropy_epoch_{epoch}.pth')
            #torch.save(fcn.state_dict(), f'model/resnet_18_xentropy_epoch_fcn_{epoch}.pth')
            gc.collect()

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
    parser.add_argument('-n_valid', default=512, type=int, help='n_valid')
    parser.add_argument('-n_test', default=1024, type=int, help='n_test')
    parser.add_argument('-batch_size', default=64, type=int, help='batch_size')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(args)
    main(args)
