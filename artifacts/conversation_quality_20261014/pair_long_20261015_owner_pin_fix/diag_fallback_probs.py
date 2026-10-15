import json, pathlib
p=pathlib.Path('/workspace/artifacts/conversation_quality_20261014/pair_long_20261015_owner_pin_fix/events.jsonl')
rows=[json.loads(s) for s in p.open(encoding='utf-8')]
first_b=next(e for e in rows if e.get('type')=='response.created' and e.get('speaker')=='B')
first_decision=max((e for e in rows if e.get('type')=='duplex_control6.decision' and e.get('speaker')=='B' and e.get('at_ms',0)<=first_b['at_ms']), key=lambda e:e['at_ms'])
print('FIRST_B_ACTION',json.dumps({k:first_decision.get(k) for k in ('at_ms','frame_id','action','native_action','selected_action','confidence','accepted','rejection_reason','legal_actions','native_action_probabilities','native_control6_raw_token','native_control6_selected_token')},ensure_ascii=False))
a=[e for e in rows if e.get('type')=='duplex_control6.decision' and e.get('speaker')=='A' and e.get('at_ms',0)>=56000 and e.get('legal_actions')==['LISTEN','SPEAK']]
ps=[float(e['confidence']) for e in a if e.get('confidence') is not None]
wins=[e for e in a if float(e.get('confidence',1.0))<0.5]
print('A_LEGAL_LISTEN_SPEAK_AFTER_56S',len(a),'confidence_min_max', (min(ps),max(ps)) if ps else None,'masked_speak_would_win',len(wins))
for e in a:
    print(json.dumps({k:e.get(k) for k in ('at_ms','frame_id','action','native_action','selected_action','confidence','accepted','rejection_reason','legal_actions','native_action_probabilities','native_control6_raw_token','native_control6_selected_token')},ensure_ascii=False))
