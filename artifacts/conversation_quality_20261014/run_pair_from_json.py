import json, subprocess, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
client = '/workspace/.tmp/listener_batch_source_overlay/client/duplex_stream_e2e.py'
args = [sys.executable, client, '--mode', 'pair', '--ws-url', cfg['ws_url'], '--audio', cfg['audio'], '--max-turns', str(cfg['max_turns']), '--max-created-responses', str(cfg['max_created_responses']), '--max-duration-s', str(cfg['max_duration_s']), '--endpoint-silence-frames', str(cfg['endpoint_silence_frames']), '--instructions-a', cfg['instructions_a'], '--instructions-b', cfg['instructions_b'], '--output-dir', cfg['output_dir']]
for key, flag in (
    ('bargein_wav', '--bargein-wav'),
    ('bargein_target', '--bargein-target'),
    ('bargein_delay_ms', '--bargein-delay-ms'),
):
    if cfg.get(key) is not None:
        args.extend([flag, str(cfg[key])])
raise SystemExit(subprocess.run(args, check=False).returncode)

