import json, sys
from collections import Counter
from pathlib import Path
p=Path(sys.argv[1]); counts=Counter(); rows=[]
for line in p.open(encoding='utf-8'):
 try: e=json.loads(line)
 except Exception: continue
 t=e.get('type','?'); counts[t]+=1
 if t in {'response.created','response.action_selected','response.output_text.delta','response.done','response.cancelled','probe.response_relayed','probe.peer_audio_frame_sent','probe.peer_output_speech_started'} or 'relay' in t or 'response' in t and t.endswith('completed'):
  row={k:e.get(k) for k in ('at_ms','type','speaker','participant','response_id','action','text','delta','source_response_id','target_session_id','generation_id') if e.get(k) is not None}
  rows.append(row)
print('EVENT_TYPES',json.dumps(counts,sort_keys=True))
print('RESPONSE_AND_RELAY_EVENTS')
for row in rows[-40:]: print(json.dumps(row,ensure_ascii=False))
