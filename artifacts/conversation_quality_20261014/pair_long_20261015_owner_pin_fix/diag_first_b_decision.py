import json, pathlib, collections
p = pathlib.Path('/workspace/artifacts/conversation_quality_20261014/pair_long_20261015_owner_pin_fix/events.jsonl')
es = [json.loads(x) for x in p.open(encoding='utf-8')]
first_b = next(e for e in es if e.get('type')=='response.created' and e.get('speaker')=='B')
decisions = [e for e in es if e.get('type')=='duplex_control6.decision' and e.get('speaker')=='B' and e.get('at_ms',0)<=first_b.get('at_ms',0)]
print('B_FIRST_CREATED', json.dumps(first_b, ensure_ascii=False))
if decisions:
    e = decisions[-1]
    print('B_PRECEDING_DECISION_ALL_FIELDS', json.dumps(e, ensure_ascii=False))
    print('B_PRECEDING_DECISION_KEYS', sorted(e.keys()))
    print('B_PRECEDING_DECISION_RELATED', json.dumps({k:v for k,v in e.items() if any(s in k.lower() for s in ('voice','speech','silence','endpoint','onset','turn','action','legal','pending','answer','reserv','pin','response','frame'))}, ensure_ascii=False))
print('NEARBY_B_EVENTS')
for e in es:
    if e.get('speaker')=='B' and 11000 <= e.get('at_ms',0) <= 12500 and e.get('type') not in ('probe.input_frame_ack','input_audio_buffer.consumed','probe.input_frame_sent'):
        print(json.dumps({k:v for k,v in e.items() if k not in ('runtime_metrics',)}, ensure_ascii=False))
print('MATCHING_EVENT_TYPES', sorted({e.get('type') for e in es if any(s in str(e.get('type','')).lower() for s in ('voice','speech','silence','endpoint','onset','turn','pin'))}))
