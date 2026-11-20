import sys
import json

log_file = sys.argv[1]
ts_start = int(sys.argv[2]) # e.g. 6532550373560327
ts_end = int(sys.argv[3])   # e.g. 6532550600000000

with open(log_file) as f:
    for line in f:
        # Check if line contains monotonic timestamps or frame references
        if "412" in line or "413" in line or "hold_ns" in line or "lock" in line:
            # check if line has ns in range
            found = False
            for token in line.split():
                if token.isdigit() and len(token) >= 15:
                    v = int(token)
                    if ts_start - 50_000_000 <= v <= ts_end + 50_000_000:
                        found = True
                        break
            if found or ('"frame_id": 412' in line or '"frame_id": 413' in line):
                print(line.strip()[:300])
