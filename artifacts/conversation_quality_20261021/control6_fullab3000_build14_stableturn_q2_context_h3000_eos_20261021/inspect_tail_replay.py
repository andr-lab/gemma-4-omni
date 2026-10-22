import json
from pathlib import Path
p=Path(__file__).with_name('events.jsonl')
rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def payload(row):
    x=row.get('event')
    return x if isinstance(x,dict) else row
for kind in ('input_audio_buffer.consumed','duplex_control6.decision','response.thinker_metrics','response.done','response.cancelled','response.audio.segment_done','error','server_error'):
    found=[r for r in rows if r.get('type')==kind]
    print(f'## {kind} count={len(found)}')
    for row in found[-20:]:
        x=payload(row)
        interesting={k:v for k,v in x.items() if any(token in k.lower() for token in ('time','ms','frame','queue','cache','fork','replay','context','response','error','action','confid','eos','pcm','event','scheduler','callback','cursor','generation','sample','status'))}
        print(json.dumps({'outer_ts':row.get('timestamp_ns',row.get('received_ns')),'payload':interesting},ensure_ascii=False,default=str))
