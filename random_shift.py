import os
import json
import random
import ffmpeg
import pdb
import math

source = 'vimeo90k'
output_path = 'shoutcut2'

prefix = f'{source}/videos/'
files = os.listdir(prefix)

for i, f in enumerate(files):
    if i % 100 == 0:
        print(f'{i}/{len(files)}')
    video_path = prefix + f
    probe = ffmpeg.probe(video_path)
    #if probe['streams'][0]['r_frame_rate'] != probe['streams'][0]['avg_frame_rate']:
    #    pdb.set_trace()
    #assert probe['streams'][0]['r_frame_rate'] == probe['streams'][0]['avg_frame_rate']
    fps_string = ''
    for s in probe['streams']:
        if s['r_frame_rate'] != '0/0':
            fps_string = s['r_frame_rate']
            break

    fps = eval(fps_string)
    duration = float(probe['format']['duration']) * 1000 - 2000

    data = json.loads(open(f'{source}/shotcut/{f}'.split('.')[0] + '_shotcut.json').read())
    min_frame = 0
    for a in data['atoms']:
        for z in range(100):
            neg_shift = a['frame'] - min_frame
            least_shift = max(math.ceil(-neg_shift)+1, 0)
            frames_shift = random.randint(math.ceil(-neg_shift)+1, 100) if random.random() < 0.9 else least_shift
            time_shift = (frames_shift / fps) * 1000

            assert a['frame'] + frames_shift > min_frame
            if a['frame'] + frames_shift > min_frame:
                if a['startTime'] + time_shift < duration:
                    a['startTime'] += time_shift
                    a['frame'] += frames_shift
                    min_frame = a['frame']
                    break

    with open(f'{source}/{output_path}/{f}'.split('.')[0] + '_shotcut.json', 'w') as fout:
        json.dump(data, fout)

