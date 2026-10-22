import json
from pathlib import Path
p=Path(__file__).with_name('events.jsonl')
rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def payload(row):
    x=row.get('event')
    return x if isinstance(x,dict) else row
wanted_prefix=('listener_batch_','scheduler_')
wanted_exact={'frame_id','action','native_action','selected_action','confidence','accepted','rejection_reason','speaker_active','native_control6_head_scored','queue_audio_for_next_turn','listener_cache_position_before','listener_cache_position_after','native_cache_current_position_after','native_cache_valid_kv_len_after','server_listener_queue_ms','server_listener_step_ms','response_id','generation_id','status','error','start_sample','end_sample','response_done_at_monotonic_ns'}
def flat(o):
  return {k:v for k,v in o.items() if k in wanted_exact or (k.startswith(wanted_prefix) and not isinstance(v,(dict,list)))}
for kind in ('input_audio_buffer.consumed','duplex_control6.decision'):
  found=[r for r in rows if r.get('type')==kind]
  print(f'## {kind} total={len(found)} last20')
  for row in found[-20:]: print(json.dumps({'ts':row.get('timestamp_ns',row.get('received_ns')),'f':flat(payload(row))},ensure_ascii=False,default=str))
for row in rows:
  if row.get('type') in {'response.done','response.cancelled'}:
    x=payload(row); rm=x.get('runtime_metrics',{})
    talker=rm.get('talker',{}) if isinstance(rm,dict) else {}
    print('## terminal',json.dumps({'type':row.get('type'),'ts':row.get('timestamp_ns',row.get('received_ns')),'response_id':x.get('response_id'),'generation':x.get('generation_id'),'talker':{k:v for k,v in talker.items() if not isinstance(v,(dict,list))},'talker_nested':{k:{a:b for a,b in v.items() if not isinstance(b,(dict,list))} for k,v in talker.items() if isinstance(v,dict) and ('eos' in k or 'budget' in k or 'frame' in k)}}))
