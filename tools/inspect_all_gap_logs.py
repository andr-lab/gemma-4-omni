import subprocess

proc = subprocess.run(["docker", "logs", "gemma4-listener-attempt9-implementation-fixes-20261024"], capture_output=True, text=True)
print("All docker logs between 20:19:03.000 and 20:19:05.500:")
for line in proc.stdout.splitlines() + proc.stderr.splitlines():
    if any(ts in line for ts in ["20:19:03", "20:19:04", "20:19:05"]):
        print(line[:250])
