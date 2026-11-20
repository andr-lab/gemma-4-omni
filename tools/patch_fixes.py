with open('duplex_kernel/listener_microbatch_worker.py', 'r', encoding='utf-8') as f:
    c = f.read()

target = '        if command.operation == "submit_tick":\n            frames, cuda_dependencies = command.args'
target_crlf = '        if command.operation == "submit_tick":\r\n            frames, cuda_dependencies = command.args'
replacement = '''        if command.operation == "submit_tick":
            if len(command.args) == 2:
                frames, cuda_dependencies = command.args
            else:
                (frames,) = command.args
                cuda_dependencies = ()'''

if target_crlf in c:
    c = c.replace(target_crlf, replacement.replace('\n', '\r\n'))
    print('Patched worker.py (CRLF)')
elif target in c:
    c = c.replace(target, replacement)
    print('Patched worker.py (LF)')
else:
    print('Target not found in worker.py')

with open('duplex_kernel/listener_microbatch_worker.py', 'w', encoding='utf-8') as f:
    f.write(c)

# Now patch shared_listener_scheduler.py: remove unconditional self._fatal_error = exc on line 810
with open('duplex_kernel/shared_listener_scheduler.py', 'r', encoding='utf-8') as f:
    s = f.read()

target_s = '''            except BaseException as exc:
                result_error = exc
                outputs = {}
                self._fatal_error = exc
                if not dependencies_completed:'''

target_s_crlf = target_s.replace('\n', '\r\n')

replacement_s = '''            except BaseException as exc:
                result_error = exc
                outputs = {}
                if _is_irrecoverable_gpu_error(exc):
                    self._fatal_error = exc
                if not dependencies_completed:'''

if target_s_crlf in s:
    s = s.replace(target_s_crlf, replacement_s.replace('\n', '\r\n'))
    print('Patched shared_listener_scheduler.py (CRLF)')
elif target_s in s:
    s = s.replace(target_s, replacement_s)
    print('Patched shared_listener_scheduler.py (LF)')
else:
    print('Target not found in shared_listener_scheduler.py')

with open('duplex_kernel/shared_listener_scheduler.py', 'w', encoding='utf-8') as f:
    f.write(s)
