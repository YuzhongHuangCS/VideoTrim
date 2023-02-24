import os
import pdb
import json
import glob
import ffmpeg
from collections import Counter

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
    def create(cls, source, path, video_path='/videos/', shotcut_path='/shotcut/', vis_prefix='vis', model_prefix='', shotcut_suffix='.json', model_name='Model'):
        if source == 'youtube':
            return YoutubeReader(source, path, video_path, shotcut_path, vis_prefix, model_prefix, shotcut_suffix, model_name)
        elif source == 'vimeo90k':
            return VimeoReader(source, path, video_path, shotcut_path, vis_prefix, model_prefix, shotcut_suffix, model_name)
        elif source == 'roughcut':
            return RoughcutReader(source, path, video_path, shotcut_path, vis_prefix, model_prefix, shotcut_suffix, model_name)
        else:
            raise NotImplementedError("Unsupported source type")

    # subclass need to filter out unwanted ids
    def get_ids(self):
        pass

    def read_id(self, name):
        #json_filename = self.path + self.shotcut_path + name + self.shotcut_suffix
        video_glob = glob.glob(self.path + self.video_path + name + '.*')
        assert len(video_glob) == 1
        probe = ffmpeg.probe(video_glob[0])
        duration = float(probe['format']['duration'])

        json_glob = glob.glob(self.path + self.shotcut_path + name + '*')
        assert len(json_glob) == 1
        with open(json_glob[0]) as fin:
            data = json.load(fin)
            ret = {
                'video': video_glob[0],
                'frames': [x['frame'] for x in data['atoms']],
                'start_times': [x['startTime']/1000 for x in data['atoms']],
                'duration': duration
            }
            return ret

class YoutubeReader(DatasetReader):
    def get_ids(self):
        #pdb.set_trace()
        video = [x.split('.')[0] for x in os.listdir(self.path + self.video_path) if x not in ('.ipynb_checkpoints', '.DS_Store')]
        shotcut = [x.replace(self.shotcut_suffix, '') for x in os.listdir(self.path + self.shotcut_path) if self.shotcut_suffix in x]

        video_set = set(video)
        shotcut_set = set(shotcut)

        assert len(video) == len(video_set)
        assert len(shotcut) == len(shotcut_set)
        #assert len(video_set - shotcut_set) == 0
        #assert len(shotcut_set - video_set) == 0
        return sorted(list(video_set.intersection(shotcut_set)))

class VimeoReader(DatasetReader):
    def get_ids(self):
        video = [x.split('.')[0] for x in os.listdir(self.path + self.video_path) if x not in ('.ipynb_checkpoints', '.DS_Store')]
        counter = Counter(video)

        video = [x[0] for x in counter.most_common() if x[1] == 1]
        shotcut = [x.replace(self.shotcut_suffix, '') for x in os.listdir(self.path + self.shotcut_path) if x not in ('.ipynb_checkpoints', '.DS_Store')]
        video_set = set(video)
        shotcut_set = set(shotcut)

        #assert len(video) == len(video_set)
        assert len(shotcut) == len(shotcut_set)
        assert len(video_set - shotcut_set) == 0
        return sorted(video)

class RoughcutReader(DatasetReader):
    def get_ids(self):
        video = [x.split('.')[0] for x in os.listdir(self.path + self.video_path) if x not in ('.ipynb_checkpoints', '.DS_Store')]
        counter = Counter(video)

        video = [x[0] for x in counter.most_common() if x[1] == 1]
        shotcut = [x.replace(self.shotcut_suffix, '').replace('.json', '') for x in os.listdir(self.path + self.shotcut_path) if x not in ('.ipynb_checkpoints', '.DS_Store')]
        video_set = set(video)
        shotcut_set = set(shotcut)

        assert len(video) == len(video_set)
        assert len(shotcut) == len(shotcut_set)

        return sorted(shotcut)
