import argparse, json
from collections import Counter
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('events'); a = p.parse_args()
counts = Counter()
for line in Path(a.events).open(encoding='utf-8'):
    try: e=json.loads(line)
    except Exception: continue
    t=e.get('type','?'); counts[t]+=1
print('EVENT_COUNTS', json.dumps(counts, ensure_ascii=False, sort_keys=True))
interesting = ('response.', 'conversation.item', 'input_audio_buffer.speech', 'session.', 'control.', 'duplex.')
for line in Path(a.events).open(encoding='utf-8'):
    try: e=json.loads(line)
    except Exception: continue
    t=e.get('type','')
    if not (t.startswith(interesting) or t.endswith('consumed')): continue
    row={k:e.get(k) for k in ('at_ms','type','response_id','frame_id','action','selected_action','effective_action','native_action','status','text','transcript','listener_frame_id','policy_decision_frame_id','policy_decision_at_ns','input_source_kind','audio_samples','first_audio_ns','timestamp_ns') if e.get(k) is not None}
    if t.endswith('consumed') and not any(k in row for k in ('action','selected_action','effective_action','native_action')): continue
    print(json.dumps(row, ensure_ascii=False))
