import os
import pdb
import json
from jinja2 import Template

fname = os.listdir('cache_test')
template = Template(open('template_start.html').read())
for f in fname:

	#X_all = torchvision.io.read_video('cache_yt/pos_0.mp4', pts_unit='sec')[0]
	path = f'cam3_start/{f}'
	if os.path.exists(path):
		files = os.listdir(path)
		idx = sorted([int(t.split('.')[0].split('_')[1]) for t in files])

		json_filename = f.replace('.mp4', '.json')
		f_json = f'scores_new/start_{json_filename}'
		if os.path.exists(f_json):
			#pdb.set_trace()
			json_data = json.loads(open(f_json).read())
			html_filename = f'cam3_start/{f}.html'
			with open(html_filename, 'w') as fout:
				fout.write(template.render(name=f, enumerate=enumerate, idx = idx, json_data = json_data))


	#print(files)
