import json, pathlib, statistics
p = pathlib.Path('/workspace/artifacts/conversation_quality_20261014/pair_long_20261015_owner_pin_fix/events.jsonl')
events = [json.loads(line) for line in p.open(encoding='utf-8')]
print('EVENTS', len(events), 'LAST_MS', events[-1].get('at_ms'))

def brief(e):
    keys = ('at_ms','speaker','type','response_id','frame_id','action','native_action','selected_action','gate_applied_action','accepted','rejection_reason','legal_actions','status','error','text_delta')
    out = {k:e.get(k) for k in keys if e.get(k) is not None}
    if isinstance(out.get('text_delta'), str): out['text_delta'] = out['text_delta'][:260]
    for k,v in e.items():
        if any(s in k for s in ('onset','answered','reserved','pending','pin','generation','response_active','user_turn')) and k not in out:
            out[k] = v
    return out

responses = [e for e in events if e.get('type') in ('response.created','response.done','response.output_text.delta')]
print('RESPONSES')
for e in responses:
    if e.get('type') == 'response.done':
        m = e.get('runtime_metrics') or {}
        out = {k:e.get(k) for k in ('at_ms','speaker','type','response_id','status')}
        out['audio_protocol'] = m.get('audio_response_protocol')
        out['talker'] = {k:(m.get('talker') or {}).get(k) for k in ('active_pcm_duration_ms','terminated_by','active_pcm_samples')}
        out['archive'] = {k:(m.get('audio_history_archive_assistant') or {}).get(k) for k in ('status','assistant_text_committed')}
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(json.dumps(brief(e), ensure_ascii=False))

for speaker in ('A','B'):
    ds = [e for e in events if e.get('type') == 'duplex_control6.decision' and e.get('speaker') == speaker]
    print('DECISIONS', speaker, len(ds))
    for e in ds[-10:]: print(json.dumps(brief(e), ensure_ascii=False))

# Measure the actual initial A PCM relayed to B, including synthetic/live silence frames.
first_a = next((e for e in events if e.get('type')=='response.created' and e.get('speaker')=='A' and e.get('action')=='SPEAK'), None)
if first_a:
    rid = first_a.get('response_id')
    done = next((e for e in events if e.get('type')=='response.done' and e.get('speaker')=='A' and e.get('response_id')==rid), None)
    bframes = [e for e in events if e.get('type')=='probe.input_frame_sent' and e.get('speaker')=='B' and e.get('source_response_id')==rid]
    silence = [e for e in events if e.get('type')=='probe.input_frame_sent' and e.get('speaker')=='B' and e.get('source_response_id')=='live_silence' and first_a['at_ms'] <= e.get('at_ms',0) <= (done or {}).get('at_ms', 0)]
    print('INTRO_RELAY', json.dumps({'response_id':rid,'created_ms':first_a.get('at_ms'),'done_ms':(done or {}).get('at_ms'),'frames_to_B':len(bframes),'first_frame_ms':bframes[0].get('at_ms') if bframes else None,'last_frame_ms':bframes[-1].get('at_ms') if bframes else None,'max_gap_ms':round(max((bframes[i]['at_ms']-bframes[i-1]['at_ms'] for i in range(1,len(bframes))),default=0),1),'gaps_over_300ms':[round(bframes[i]['at_ms']-bframes[i-1]['at_ms'],1) for i in range(1,len(bframes)) if bframes[i]['at_ms']-bframes[i-1]['at_ms']>300],'live_silence_frames_during_A_response':len(silence),'live_silence_first_last_ms':[silence[0].get('at_ms'),silence[-1].get('at_ms')] if silence else None}, ensure_ascii=False))
    bfirst = next((e for e in events if e.get('type')=='response.created' and e.get('speaker')=='B'), None)
    if bfirst:
        delivered_before = [e for e in bframes if e.get('at_ms',0) <= bfirst.get('at_ms',0)]
        print('B_FIRST_RESPONSE_VS_A_PCM', json.dumps({'B_created_ms':bfirst.get('at_ms'),'B_action':bfirst.get('action'),'A_source_frames_already_sent_to_B':len(delivered_before),'last_A_source_frame_before_B_create_ms':delivered_before[-1].get('at_ms') if delivered_before else None},ensure_ascii=False))

print('TERMINAL_TICKS')
for e in [e for e in events if e.get('type')=='duplex_control6.decision'][-20:]:
    print(json.dumps(brief(e), ensure_ascii=False))
