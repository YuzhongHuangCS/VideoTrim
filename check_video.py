import os
import av
import pdb
import ffmpeg

prefix = 'data/youtube/videos/'
for f in os.listdir(prefix):
	print(f)
	video_name = prefix + f
	container = av.open(video_name)
	if len(container.streams.video) != 1:
		pdb.set_trace()
		print('More than 1 video stream')

	frame_count = 0
	fps = float(container.streams.video[0].average_rate)
	for frame in container.decode(video=0):
		frame_count += 1
	probe = ffmpeg.probe(video_name)
	duration = float(probe['format']['duration'])
	print(duration, fps, frame_count, duration*fps)
	if abs(frame_count - duration*fps) > 0.4:
		pdb.set_trace()
		print(f)

