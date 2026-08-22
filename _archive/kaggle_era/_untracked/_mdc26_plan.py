import os, requests
key = os.environ["MDC_API_KEY"]
r = requests.post("https://mozilladatacollective.com/api/datasets/cmqini14100vmnq07309ocknr/download",
                  headers={"Authorization": "Bearer " + key}, timeout=60)
r.raise_for_status()
plan = r.json()
with open("/workspace/mdc26_it/_download_url.txt", "w") as f:
    f.write(plan["downloadUrl"])
print("PLAN OK, sizeBytes:", plan.get("sizeBytes"), "accessRecord:", plan.get("accessRecordId"))
