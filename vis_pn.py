import os
import pdb
import json
import glob
from jinja2 import Template
from collections import Counter
import time
import datetime
import numpy as np
import copy
from concurrent.futures import ProcessPoolExecutor
import torch
import torchvision
import ffmpeg
import random

executor = ProcessPoolExecutor(max_workers=4)
class DatasetReader(object):
    def __init__(self, source, path, video_path, shotcut_path, vis_prefix, model_prefix, shotcut_suffix, model_name):
        if type(self) == DatasetReader:
            raise Exception("DatasetReader is an abstract class")
        super(DatasetReader, self).__init__()
        self.source = source
        self.path = path
        self.video_path = video_path
        self.shotcut_path = shotcut_path
        self.vis_prefix = vis_prefix
        self.model_prefix = model_prefix
        self.shotcut_suffix = shotcut_suffix
        self.model_name = model_name

    @classmethod
    def create(cls, source, path, video_path='/videos/', shotcut_path='/shotcut/', vis_prefix='vis', model_prefix='', shotcut_suffix='_shotcut.json', model_name='Model'):
        if source == 'youtube':
            return YoutubeReader(source, path, video_path, shotcut_path, vis_prefix, model_prefix, shotcut_suffix, model_name)
        elif source == 'vimeo90k':
            return VimeoReader(source, path, video_path, shotcut_path, vis_prefix, model_prefix, shotcut_suffix, model_name)
        else:
            raise NotImplementedError("Unsupported source type")
        
    # subclass need to filter out unwanted ids
    def get_ids(self):
        pass

    def read_id(self, name):
        json_filename = self.path + self.shotcut_path + name + self.shotcut_suffix
        video_glob = glob.glob(self.path + self.video_path + name + '.*')
        assert len(video_glob) == 1
        probe = ffmpeg.probe(video_glob[0])
        duration = float(probe['format']['duration'])
        with open(json_filename) as fin:
            data = json.load(fin)
            ret = {
                'video': video_glob[0],
                'frames': [x['frame'] for x in data['atoms']],
                'start_times': [0] + [x['startTime']/1000 for x in data['atoms']],
                'duration': duration,
            }
            return ret

class YoutubeReader(DatasetReader):
    def get_ids(self):
        video = [x.split('.')[0] for x in os.listdir(self.path + self.video_path) if x not in ('.ipynb_checkpoints', '.DS_Store')]
        shotcut = [x.replace(self.shotcut_suffix, '') for x in os.listdir(self.path + self.shotcut_path) if x not in ('.ipynb_checkpoints', '.DS_Store')]

        video_set = set(video)
        shotcut_set = set(shotcut)
        assert len(video) == len(video_set)
        assert len(shotcut) == len(shotcut_set)
        assert len(video_set - shotcut_set) == 0
        assert len(shotcut_set - video_set) == 0
        return video

class VimeoReader(DatasetReader):
    def get_ids(self):
        video = [x.split('.')[0] for x in os.listdir(self.path + self.video_path) if x not in ('.ipynb_checkpoints', '.DS_Store')]
        counter = Counter(video)

        video = [x[0] for x in counter.most_common() if x[1] == 1]
        shotcut = [x.replace(self.shotcut_suffix, '') for x in os.listdir(self.path + self.shotcut_path) if x not in ('.ipynb_checkpoints', '.DS_Store')]
        video_set = set(video)
        shotcut_set = set(shotcut)

        assert len(video) == len(video_set)
        assert len(shotcut) == len(shotcut_set)
        assert len(video_set - shotcut_set) == 0
        return video

#generate start clips:
def generate_visualization(source, r):
    names = r.get_ids()

    for name in names[:20]:
        prefix = f'{r.vis_prefix}/{source}/{name}'
        os.makedirs(prefix, exist_ok=True)

        config = r.read_id(name)
        video_path = config['video']
        start_times = config['start_times']
        duration = config['duration']

        if len(start_times) == 2:
            print(f'{name} contains 0 cuts')
            continue

        # anything smaller than -2 works
        pos_times = []
        last_time = -10
        for start_time in start_times:
            if start_time - last_time > 2:
                pos_times.append(start_time)
                last_time = start_time
        if duration - pos_times[-1] < 2:
            pos_times.pop()

        n_total = 9
        n_pos = random.randint(1, 8)
        pos_times_copy = copy.deepcopy(pos_times)
        random.shuffle(pos_times_copy)
        pos_times_copy = pos_times_copy[:n_pos]
        for idx, start_time in enumerate(pos_times_copy):
            cmd = f'ffmpeg -y -threads 1 -ss {start_time} -t 2 -i {video_path} -threads 1 -vcodec libx264 -an {prefix}/{r.model_prefix}{idx}_pos.mp4'
            print(cmd)
            #executor.submit(os.system, cmd)

        pos_times.append(duration)
        neg_times = []
        for idx, start_time in enumerate(pos_times[:-1]):
            neg_start_time = start_time + 2
            neg_end_time = pos_times[idx+1]
            if neg_end_time - neg_start_time > 2:
                neg_times.append([neg_start_time, neg_end_time])

        neg_times_copy = copy.deepcopy(neg_times)
        random.shuffle(neg_times_copy)
        neg_times_copy = neg_times_copy[:(n_total-n_pos)]
        for idx, pair in enumerate(neg_times_copy):
            start, end = pair
            start = random.uniform(start, end-2)
            end = start + 2

            cmd = f'ffmpeg -y -threads 1 -ss {start} -t {end-start} -i {video_path} -threads 1 -vcodec libx264 -an {prefix}/{r.model_prefix}{idx}_neg.mp4'
            print(cmd)
            #executor.submit(os.system, cmd)

        render_data = []
        for i, v in enumerate(pos_times_copy):
            config = {
                'label': 'pos',
                'index': i
            }
            render_data.append(config)
        for i, v in enumerate(neg_times_copy):
            config = {
                'label': 'neg',
                'index': i
            }
            render_data.append(config)

        random.shuffle(render_data)
        answer = [v['label'] for v in render_data]
        html_filename = f'{r.vis_prefix}/{source}/{name}/index.html'
        template = Template(open('template_pn.html').read())
        with open(html_filename, 'w') as fout:
            fout.write(template.render(enumerate=enumerate, name=name, render_data=render_data, reader=r))


if __name__ == "__main__":
    youtube1 = DatasetReader.create('youtube', '/Users/yhuang/video_trim/data/youtube', shotcut_path='/shotcut/', model_prefix='', model_name='Model 1', vis_prefix='vis_pn', answer=json.dumps(answer))
    generate_visualization('youtube', youtube1)
    print('All task has generated')
    executor.shutdown()
    print('All done')