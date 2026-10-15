import json,pathlib
p=pathlib.Path('/workspace/artifacts/conversation_quality_20261014/pair_long_20261015_owner_pin_fix/events.jsonl')
for s in p.open(encoding='utf-8'):
 e=json.loads(s)
 if e.get('speaker')=='B' and ((e.get('type')=='input_audio_buffer.consumed' and 140<=e.get('frame_id',-1)<=146) or e.get('type') in ('response.created','response.action_selected') and e.get('response_id')=='resp_a5d71e00da36'):
  print(json.dumps(e,ensure_ascii=False))
