import json

filename = 'scores_f1_both.json'
data = json.loads(open(filename, 'r').read())

html_str = '''<!DOCTYPE html>
<html>
<head>
    <title>List of video</title>
</head>
<body>
<p>Raw footage</p>
<ol>
<li><a href='http://34.221.247.37/chart/?file=surf'>http://34.221.247.37/chart/?file=surf</a></li>
<li><a href='http://34.221.247.37/chart/?file=driving'>http://34.221.247.37/chart/?file=driving</a></li>
<li><a href='http://34.221.247.37/chart/?file=skateboard.mp4'>http://34.221.247.37/chart/?file=skateboard.mp4</a></li>
<li><a href='http://34.221.247.37/chart/?file=skateboard_practice'>http://34.221.247.37/chart/?file=skateboard_practice</a></li>
<li><a href='http://34.221.247.37/chart/?file=basketball'>http://34.221.247.37/chart/?file=basketball</a></li>
<li><a href='http://34.221.247.37/chart/?file=basketball_practice'>http://34.221.247.37/chart/?file=basketball_practice</a></li>
</ol>

<p>Test video: name - avg f1</p>
<ol>
'''

for pair in data:
    score, name = pair
    html_str += f"<li><a href='http://34.221.247.37/chart/?file={name}'>http://34.221.247.37/chart/?file={name}</a> - {score}</li>"
    
html_str += '''
</ol>
</body>
</html>
'''

with open(filename.replace('.json', '.html'), 'w') as fout:
    fout.write(html_str)