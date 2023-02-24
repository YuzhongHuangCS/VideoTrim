import os
import pdb
import json
import glob
from jinja2 import Template
from collections import Counter

class DatasetReader(object):
    def __init__(self, path):
        if type(self) == DatasetReader:
            raise Exception("DatasetReader is an abstract class")
        super(DatasetReader, self).__init__()
        self.path = path

    @classmethod
    def create(cls, source, path):
        if source == 'youtube':
            return YoutubeReader(path)
        elif source == 'vimeo90k':
            return VimeoReader(path)
        elif source == 'roughcut':
            return RoughcutReader(path)
        else:
            raise NotImplementedError("Unsupported source type")
        
    # subclass need to filter out unwanted ids
    def get_ids(self):
        pass
    
    def read_id(self, name):
        json_filename = f'{self.path}/shotcut/{name}_shotcut.json'
        video_glob = glob.glob(f'{self.path}/videos/{name}.*')
        assert len(video_glob) == 1

        with open(json_filename) as fin:
            data = json.load(fin)
            ret = {
                'video': video_glob[0],
                'cuts': [x['frame'] for x in data['atoms']],
                'start_times': [x['startTime']/1000 for x in data['atoms']],
            }
            return ret

class VimeoReader(DatasetReader):
    def __init__(self, path):
        super(VimeoReader, self).__init__(path)

    def get_ids(self):
        video = [x.split('.')[0] for x in os.listdir(self.path + '/videos')]
        counter = Counter(video)

        video = [x[0] for x in counter.most_common() if x[1] == 1]
        shotcut = [x.replace('_shotcut.json', '') for x in os.listdir(self.path + '/shotcut')]
        video_set = set(video)
        shotcut_set = set(shotcut)

        assert len(video) == len(video_set)
        assert len(shotcut) == len(shotcut_set)
        assert len(video_set - shotcut_set) == 0
        return video

class YoutubeReader(DatasetReader):
    def __init__(self, path):
        super(YoutubeReader, self).__init__(path)

    def get_ids(self):
        video = [x.split('.')[0] for x in os.listdir(self.path + '/videos')]
        shotcut = [x.replace('_shotcut.json', '') for x in os.listdir(self.path + '/shotcut')]

        video_set = set(video)
        shotcut_set = set(shotcut)
        assert len(video) == len(video_set)
        assert len(shotcut) == len(shotcut_set)
        assert len(video_set - shotcut_set) == 0
        assert len(shotcut_set - video_set) == 0
        return video

def generate_visualization(source, path):
    reader = DatasetReader.create(source, path)
    ids = reader.get_ids()

    for name in ids[:20]:
        config = reader.read_id(name)
        video_path = config['video']
        cuts = config['cuts']
        if len(cuts) == 0:
            print(f'{name} contains 0 cuts')
            continue

        os.makedirs(f'vis_end/{source}/{name}', exist_ok=True)
        start_times = config['start_times']
        for idx, start_time in enumerate(start_times):
            cmd = f'ffmpeg -y -ss {start_time-2} -t 1.91 -i {video_path} -vf "fps=fps=10" -f gif -s 720x480 vis_end/{source}/{name}/{idx}.gif'
            print(cmd)
            os.system(cmd)
        
        n_cuts = len(start_times)
        html_filename = f'vis_end/{source}/{name}/index.html'
        url_prefix = f'https://senseij01-or2-proxy.infra.adobesensei.io/Int-a2841a75-f81a-44bf-935e-4d693ecd7df13/jupyter/files/user_space/{source}/'
        video_ext = video_path.split('.')[-1]

        template = Template(open('template.html').read())
        with open(html_filename, 'w') as fout:
            fout.write(template.render(url_prefix=url_prefix, name=name, cuts=start_times, n_cuts=n_cuts, video_ext=video_ext))

if __name__ == "__main__":
    generate_visualization('vimeo90k', '/home/code-base/data_space/vimeo90k')
    generate_visualization('youtube', '/home/code-base/data_space/youtube')

