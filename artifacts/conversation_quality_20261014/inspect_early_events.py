import json, sys
for line in open(sys.argv[1], encoding='utf-8'):
 e=json.loads(line)
 if e.get('type') == 'duplex_control6.decision' and e.get('frame_id') in (0,1,2,3,4,5,6):
  print('DECISION', e.get('at_ms'), 'frame',e.get('frame_id'), 'keys',list(e.keys()))
  print(json.dumps(e, ensure_ascii=False, separators=(',',':'))[:3500])
 if e.get('type') == 'probe.input_frame_sent' and e.get('frame_index') < 10:
  print('INPUT',e.get('at_ms'),e.get('frame_index'),e.get('rms'),e.get('scheduled_capture_at_ns'),e.get('dispatch_started_at_ns'))
