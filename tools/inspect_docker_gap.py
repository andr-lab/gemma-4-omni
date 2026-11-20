import subprocess, json

proc = subprocess.run(["docker", "logs", "gemma4-listener-attempt9-implementation-fixes-20261024"], capture_output=True, text=True)
print("Docker logs between 20:19:03 and 20:19:06:")
for line in proc.stdout.splitlines() + proc.stderr.splitlines():
    if any(ts in line for ts in ["20:19:03", "20:19:04", "20:19:05", "20:19:06"]):
        if "duplex_control6.trace" not in line and "runtime-telemetry" not in line:
            print(line[:250])
