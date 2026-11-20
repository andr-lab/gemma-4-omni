import subprocess

proc = subprocess.run(["docker", "logs", "gemma4-listener-attempt9-implementation-fixes-20261024"], capture_output=True, text=True)
lines = proc.stdout.splitlines() + proc.stderr.splitlines()
target = "resp_65a4d035069a"
found = False
for idx, line in enumerate(lines):
    if target in line:
        print(f"[{idx}] {line[:200]}")
        for j in range(max(0, idx-5), min(len(lines), idx+30)):
            if "playback drain/cache finalization failed" in lines[j] or "Traceback" in lines[j]:
                print("\n".join(lines[j:j+25]))
                found = True
                break
        if found:
            break
