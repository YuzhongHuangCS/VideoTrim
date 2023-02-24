import pdb
import pandas as pd
import ffmpeg
import timecode
import json

df = pd.read_csv('editstock/clips_score_clean_concat.csv')
time_start_ary = []
time_end_ary = []
score_start_ary = []
score_end_ary = []
for idx, row in df.iterrows():
    video_filename = row['video_filename']
    main_name = video_filename.split('/')[1].split('.')[0]

    probe = ffmpeg.probe('editstock/' + video_filename)

    for s in probe['streams']:
        if s['codec_type'] == 'video':
            fps = s['avg_frame_rate']
            tc_base = s['tags']['timecode']
            break

    tc_base = timecode.Timecode(fps, tc_base)
    tc_start = str(timecode.Timecode(fps, row['start_timecode']) - tc_base)
    tc_end = str(timecode.Timecode(fps, row['end_timecode']) - tc_base)
    fps = float(tc_base.framerate)
    sample_ratio = fps // 8

    token_start = tc_start.rsplit(':', 1)
    tc_start = timecode.Timecode(8, token_start[0] + ':' + str(int(int(token_start[1]) // sample_ratio)))

    token_end = tc_end.rsplit(':', 1)
    tc_end = timecode.Timecode(8, token_end[0] + ':' + str(int(int(token_end[1]) // sample_ratio)))

    time_start_ary.append(tc_start)
    time_end_ary.append(tc_end)
    data = json.loads(open(f'scores/start_{main_name}.json', 'r', encoding='utf-8').read())
    if tc_start.frame_number >= len(data['start']):
        score_start = -1
    else:
        score_start = data['start'][tc_start.frame_number]

    if tc_end.frame_number >= len(data['end']):
        score_end = -1
    else:
        score_end = data['end'][tc_end.frame_number]
    print(main_name, score_start, score_end)
    score_start_ary.append(score_start)
    score_end_ary.append(score_end)

df['score_start'] = score_start_ary
df['score_end'] = score_end_ary
df['start_timecode_true'] = time_start_ary
df['end_timecode_true'] = time_end_ary
df.to_csv('clips_score_concat.csv', index=False)
print('Done')
