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
from dataset_reader import DatasetReader
import sys

if sys.platform == 'darwin':
    sys_prefix = '/Users/yhuang/video_trim/data/'
    num_cpus = 8
else:
    sys_prefix = '/home/code-base/data_space/'
    num_cpus = 4

def print_and_run(cmd):
    print(cmd)
    os.system(cmd)

def generate_jpg_and_rename(frames, video_path, prefix):
    select_str = '+'.join([f'eq(n\,{f})' for f in frames[:10]])
    cmd = f'ffmpeg -y -threads 1 -i {video_path} -threads 1 -s 720x480 -vf select="{select_str}" -vsync 0 {prefix}/%d.jpg'
    print(cmd)
    os.system(cmd)
    for i, f in enumerate(frames):
        os.rename(f'{prefix}/{i+1}.jpg', f'{prefix}/f_{f}.jpg')

executor = ProcessPoolExecutor(max_workers=num_cpus)
def generate_visualization(source, readers):
    reader_ids = [r.get_ids() for r in readers]

    ok_set = set(reader_ids[0])
    ok_set = ok_set.intersection(set(reader_ids[1]))
    ok_set = ok_set.intersection(set(reader_ids[2]))
    ok_set = ok_set.intersection(set(reader_ids[3]))
    names = list(ok_set)
    #assert all(x == reader_ids[0] for x in reader_ids)
    n_readers = len(readers)

    for name in names:
        n_cuts_ary = []
        frames_ary = []
        start_times_ary = []
        start_stamps_ary = []

        jpg_frames = []
        mp4_frames_times = {}

        for r in readers:
            prefix = f'{r.vis_prefix}/{source}/{name}'
            os.makedirs(prefix, exist_ok=True)
            config = r.read_id(name)
            video_path = config['video']
            frames = config['frames']
            start_times = config['start_times']
            if len(frames) == 0:
                print(f'{name} contains 0 cuts')
                continue
            
            jpg_frames += frames
            jpg_frames += [f-1 for f in frames]

            '''
            select_str_before = '+'.join([f'eq(n\,{f-1})' for f in frames])
            cmd = f'ffmpeg -y -threads 1 -i {video_path} -threads 1 -s 720x480 -vf select="{select_str_before}" -vsync 0 {prefix}/{r.model_prefix}b_%d.jpg'
            executor.submit(print_and_run, cmd)

            select_str_after = '+'.join([f'eq(n\,{f})' for f in frames])
            cmd = f'ffmpeg -y -threads 1 -i {video_path} -threads 1 -s 720x480 -vf select="{select_str_after}" -vsync 0 {prefix}/{r.model_prefix}a_%d.jpg'
            executor.submit(print_and_run, cmd)

            for idx, start_time in enumerate(start_times):
                    cmd = f'ffmpeg -y -threads 1 -ss {start_time-1} -t 2 -i {video_path} -threads 1 -s 720x480 -vcodec libx264 -an {prefix}/{r.model_prefix}{idx}.mp4'
                    executor.submit(print_and_run, cmd)
            '''
            for idx, pair in enumerate(zip(start_times, frames)):
                start_time, frame = pair
                mp4_frames_times[frame] = start_time - 1

            start_stamps = [datetime.datetime.utcfromtimestamp(x).strftime('%H:%M:%S.%f')[:-3] for x in start_times]
            n_cuts_ary.append(len(frames))
            frames_ary.append(frames)
            start_times_ary.append(start_times)
            start_stamps_ary.append(start_stamps)

        jpg_frames = sorted(set(jpg_frames))
        executor.submit(generate_jpg_and_rename, jpg_frames, video_path, prefix)

        for frame, start_time in mp4_frames_times.items():
            cmd = f'ffmpeg -y -threads 1 -ss {start_time} -t 2 -i {video_path} -threads 1 -s 720x480 -vcodec libx264 -an {prefix}/f_{frame}.mp4'
            executor.submit(print_and_run, cmd)

        render_data = []
        frames_stack = copy.deepcopy(frames_ary)
        while any(l > 0 for l in [len(x) for x in frames_stack]):
            frame_peek = np.asarray([x[0] if x else np.nan for x in frames_stack])
            column_index = np.where(frame_peek == np.nanmin(frame_peek))[0]
            row_index = [len(frames_ary[i]) - len(frames_stack[i]) for i in column_index]
            frame_values = [frames_stack[i].pop(0) for i in column_index]
            model_ary = [-1] * n_readers
           
            for i, c in enumerate(column_index):
                r = row_index[i]
                model_ary[c] = r

            if len(frame_values) == 0:
                pdb.set_trace()

            config = {
                'frame': frame_values[0],
                'time': start_times_ary[column_index[0]][row_index[0]],
                'stamp': start_stamps_ary[column_index[0]][row_index[0]],
                'models': model_ary
            }
            render_data.append(config)

        html_filename = f'{readers[0].vis_prefix}/{source}/{name}/index.html'
        url_prefix = f'https://senseij01-or2-proxy.infra.adobesensei.io/Int-a2841a75-f81a-44bf-935e-4d693ecd7df13/jupyter/files/user_space/{source}'
        video_ext = video_path.split('.')[-1]

        template = Template(open('template.html').read())
        with open(html_filename, 'w') as fout:
            fout.write(template.render(enumerate=enumerate, url_prefix=url_prefix, name=name, video_ext=video_ext, n_cuts_ary=n_cuts_ary, render_data=render_data, readers=readers))

if __name__ == "__main__":
    youtube1 = DatasetReader.create('youtube', sys_prefix + 'youtube', shotcut_path='/compare_shotcut/', vis_prefix='vis', shotcut_suffix='_jumpcut_baseline.json', model_prefix='jumpcut_', model_name='jumpcut_baseline')
    youtube2 = DatasetReader.create('youtube', sys_prefix + 'youtube', shotcut_path='/compare_shotcut/', vis_prefix='vis', shotcut_suffix='_no_jumpcut_4_10.json', model_prefix='nojump410_', model_name='no_jumpcut_4_10')
    youtube3 = DatasetReader.create('youtube', sys_prefix + 'youtube', shotcut_path='/compare_shotcut/', vis_prefix='vis', shotcut_suffix='_no_jumpcut_5_28.json', model_prefix='nojump528_', model_name='no_jumpcut_5_28')
    youtube4 = DatasetReader.create('youtube', sys_prefix + 'youtube', shotcut_path='/compare_shotcut/', vis_prefix='vis', shotcut_suffix='_fewer_jumpcut_5_28.json', model_prefix='fewerjumpcut_', model_name='fewer_jumpcut_5_28')
    generate_visualization('youtube', [youtube1, youtube2, youtube3, youtube4])
    #vimeo90k1 = DatasetReader.create('vimeo90k', sys_prefix + 'vimeo90k', shotcut_path='/shotcut/', model_prefix='model1_', model_name='Model 1')
    #vimeo90k2 = DatasetReader.create('vimeo90k', sys_prefix + 'vimeo90k', shotcut_path='/shotcut2/', model_prefix='model2_', model_name='Model 2')
    #vimeo90k3 = DatasetReader.create('vimeo90k', sys_prefix + 'vimeo90k', shotcut_path='/shotcut3/', model_prefix='model3_', model_name='Model 3')
    #generate_visualization('vimeo90k', [vimeo90k1, vimeo90k2, vimeo90k3])
    print('All task has generated')
    executor.shutdown()
    print('All done')
