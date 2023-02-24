import os
import pdb
import json
from collections import Counter

youtube_folder = '/home/code-base/data_space/youtube/'
youtube_mp4 = set([x.replace('.mp4', '') for x in os.listdir(youtube_folder + 'videos')])
youtube_json = set([x.replace('_shotcut.json', '') for x in os.listdir(youtube_folder + 'shotcut')])
print('Youtube: Has mp4 but no json:', youtube_mp4 - youtube_json)
print('Youtube: Has json but no mp4:', youtube_json - youtube_mp4)


vimeo_folder = '/home/code-base/data_space/vimeo90k/'
vimeo_mp4 = set([x.split('.')[0] for x in os.listdir(vimeo_folder + 'videos')])
vimeo_json = set([x.replace('_shotcut.json', '') for x in os.listdir(vimeo_folder + 'shotcut')])
print('Vimeo: Has mp4 but no json:', vimeo_mp4 - vimeo_json)
print('Vimeo: Has json but no mp4:', vimeo_json - vimeo_mp4)

roughcut_folder = '/home/code-base/data_space/roughcut/'
roughcut_mp4 = set([x.split('.')[0] for x in os.listdir(roughcut_folder + 'videos')])
roughcut_json = []

for i, x in enumerate(os.listdir(roughcut_folder + 'meta')):
    #if i % 1000 == 0:
    #    print(i)
    meta_path = f'{roughcut_folder}meta/{x}'
    if os.path.exists(meta_path):
        with open(meta_path) as fin:
            source_id = json.load(fin)['source_id']
            shotcut_path = f'{roughcut_folder}shotcut/{source_id}.json'
            if os.path.exists(shotcut_path):
                roughcut_json.append(x.replace('.json', ''))

print('Roughcut: Has mp4 but no json:', vimeo_mp4 - vimeo_json)
print('Roughcut: Has json but no mp4:', vimeo_json - vimeo_mp4)


print(123)
'''
vimeo_ids = [x.split('.')[0] for x in os.listdir(vimeo_folder + 'videos')]
counter = Counter(vimeo_ids)
print(counter.most_common(45))

for name, count in counter.most_common(45):
    assert count == 2
    
    mp4_file = f'{vimeo_folder}videos/{name}.mp4'
    mov_file = f'{vimeo_folder}videos/{name}.mov'

    if not os.path.exists(mp4_file):
        pdb.set_trace()
    if not os.path.exists(mov_file):
        pdb.set_trace()

    print('='*20 + name)        
    mp4_cmd = f'ffmpeg -i {mp4_file} 2>&1 | grep "Duration\|fps"'
    os.system(mp4_cmd)
    
    mov_cmd = f'ffmpeg -i {mov_file} 2>&1 | grep "Duration\|fps"'
    os.system(mov_cmd)

vimeo_mp4 = set([x.split('.')[0] for x in os.listdir(vimeo_folder + 'videos')])
vimeo_json = set([x.replace('_shotcut.json', '') for x in os.listdir(vimeo_folder + 'shotcut')])
pdb.set_trace()

'''
