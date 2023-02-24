import json
import os
import collections
import pdb
import pickle

with open('map_data/pos_map.json', 'r') as fin:
	pos_map = json.load(fin)

with open('map_data/neg_map.json', 'r') as fin:
	neg_map = json.load(fin)

with open('map_data/pos_map_jpg.json', 'r') as fin:
	pos_map_jpg = json.load(fin)

with open('map_data/neg_map_jpg.json', 'r') as fin:
	neg_map_jpg = json.load(fin)

with open('video_name_counter.pickle', 'rb') as fin:
	video_name_counter = pickle.load(fin)
    

total_counter = collections.Counter()
for idx, ary in pos_map.items():
    total_counter[ary[1]] += 1
for idx, ary in neg_map.items():
    total_counter[ary[1]] += 1

counter = 0
remove_set = set()
for pair in video_name_counter.most_common():
    name = pair[0]
    total = total_counter[name]
    if pair[1]/total > 0.5:
        counter += 1
        remove_set.add(name)

with open('remove_set.pickle', 'wb') as fout:
    pickle.dump(remove_set, fout)

for idx, ary in pos_map.items():
    if ary[1] in remove_set:
        fname = f'cache/pos_{idx}.mp4'
        if os.path.exists(fname):
            os.rename(f'cache/pos_{idx}.mp4', f'cache_cg/pos_{idx}.mp4')
for idx, ary in neg_map.items():
    if ary[1] in remove_set:
        fname = f'cache/neg_{idx}.jpg'
        if os.path.exists(fname):
            os.rename(f'cache/neg_{idx}.mp4', f'cache_cg/neg_{idx}.mp4')
print(len(total_counter))
