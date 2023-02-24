import json
import os

with open('dataset_names', 'w') as fout:
    for dataset in ['youtube', 'vimeo90k', 'roughcut']:
        for part in ['videos', 'shotcut']:
            print(f'{dataset}/{part}')
            names = os.listdir(f'/home/code-base/data_space/{dataset}/{part}')
            for name in names:
                fout.write(f'{dataset}/{part}/' + name + '\n')
