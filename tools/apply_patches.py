with open('tools/real_dialogue_simulator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove duplicate _record_kv_values in response.unified_native_finalized
target1 = """            elif kind == "response.unified_native_finalized" and record is not None:
                record.finalizer_event = event
                record.final_observation_ns = int(received_ns)
                record.finalizer_signal.set()
                self._record_kv_values(event)"""

repl1 = """            elif kind == "response.unified_native_finalized" and record is not None:
                record.finalizer_event = event
                record.final_observation_ns = int(received_ns)
                record.finalizer_signal.set()"""

# 2. Check speaking_created_ids != set(assignments)
target2 = """    if set(created_by_id) != set(response_reports) or set(created_by_id) != set(assignments):
        gaps.append("response IDs in report, raw events, and scenario assignments do not match")"""

repl2 = """    speaking_created_ids = {
        rid for rid, ev in created_by_id.items()
        if str(ev.get("effective_action", ev.get("action", ""))).upper() == "SPEAK"
    }
    if set(created_by_id) != set(response_reports) or speaking_created_ids != set(assignments):
        gaps.append("response IDs in report, raw events, and scenario assignments do not match")"""

# 3. Account for backpressure wait in offline pacing revalidation
target3 = """        sent_times = [int(payload["sent_ns"]) for payload in sent_payloads]
        for sent_ns in sent_times:
            pacing.record_send(sent_ns)"""

repl3 = """        for payload in sent_payloads:
            sent_ns = int(payload["sent_ns"])
            backpressure_wait_ns = int(payload.get("backpressure_wait_ns", 0))
            if payload.get("waited_for_backpressure"):
                pacing.deadline_ns = sent_ns
            pacing.record_send(sent_ns, backpressure_wait_ns=backpressure_wait_ns)"""

assert target1 in content, "target1 not found"
assert target2 in content, "target2 not found"
assert target3 in content, "target3 not found"

content = content.replace(target1, repl1, 1)
content = content.replace(target2, repl2, 1)
content = content.replace(target3, repl3, 1)

with open('tools/real_dialogue_simulator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully applied patches to tools/real_dialogue_simulator.py")
