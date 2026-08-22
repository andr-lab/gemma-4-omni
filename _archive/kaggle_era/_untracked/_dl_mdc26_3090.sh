#!/bin/bash
# Download MDC Common Voice 26 Italian via datacollective SDK (chiave da env)
set -e
cd ~/gemma-4-omni
mkdir -p mdc26_it
export MDC_API_KEY=c77d2b51d520d6e126d898f69291e32e98a42a223d1cb57b4285983302be6e31
export MDC_DOWNLOAD_PATH=/home/administrator/gemma-4-omni/mdc26_it

# usa il pacchetto datacollective ufficiale: download_dataset scarica l'archivio
# ma archive_filename e' None nei details -> usiamo load_dataset che gestisce id+slug
/opt/venv_phase1/bin/python - <<'PYEOF'
import os, sys, json
os.environ.setdefault("MDC_API_KEY", "c77d2b51d520d6e126d898f69291e32e98a42a223d1cb57b4285983302be6e31")
os.environ.setdefault("MDC_DOWNLOAD_PATH", "/home/administrator/gemma-4-omni/mdc26_it")

from datacollective import get_dataset_details
try:
    d = get_dataset_details("cmqini14100vmnq07309ocknr")
    print("DETAILS OK:", d.id, d.name, flush=True)
except Exception as e:
    print("DETAILS FAIL:", type(e).__name__, e, flush=True)
    sys.exit(1)

# download session diretto (archive_filename e' None nel model -> usiamo API diretta)
import requests
r = requests.post("https://mozilladatacollective.com/api/datasets/cmqini14100vmnq07309ocknr/download",
                  headers={"Authorization": f"Bearer {os.environ['MDC_API_KEY']}"}, timeout=60)
r.raise_for_status()
plan = r.json()
print("PLAN:", {k: (str(v)[:80] if k != "downloadUrl" else "URL...") for k, v in plan.items()}, flush=True)
url = plan["downloadUrl"]
# salva l'URL firmato per il download con curl (piu' robusto del resume del SDK)
with open("/home/administrator/gemma-4-omni/mdc26_it/_download_url.txt", "w") as f:
    f.write(url)
print("URL salvato", flush=True)
PYEOF
echo "=== download con curl (resumable) ==="
URL=$(cat mdc26_it/_download_url.txt)
# estrae il nome file dall'URL
FNAME=$(echo "$URL" | sed -n 's#.*/\(cv-corpus[^?]*\).*#\1#p')
echo "FNAME=$FNAME"
curl -L --retry 5 --retry-delay 5 -C - -o "mdc26_it/${FNAME:-cv-corpus-26.0-2026-06-12-it.tar.gz}" "$URL"
ls -la mdc26_it/
echo "DOWNLOAD_DONE"
