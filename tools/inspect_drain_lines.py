import subprocess

proc = subprocess.run(["docker", "logs", "gemma4-listener-attempt9-implementation-fixes-20261024"], capture_output=True, text=True)
lines = proc.stdout.splitlines() + proc.stderr.splitlines()
target = "resp_65a4d035069a"
for idx in range(80885, min(len(lines), 81145)):
    line = lines[idx]
    if "duplex_control6.trace" not in line and "runtime-telemetry" not in line:
        print(line[:250])
