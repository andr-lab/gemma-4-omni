import json
import glob
import os

def main():
    dirs = glob.glob("/home/administrator/gemma-4-omni/artifacts/acceptance_suite_*/run_01")
    dirs.sort(key=lambda d: os.path.getmtime(d))
    run_dir = dirs[-1]

    with open(os.path.join(run_dir, "report.json"), "r") as f:
        rep = json.load(f)
    print("acceptance_parameters:")
    for k, v in rep.get("acceptance_parameters", {}).items():
        if "budget" in k:
            print(f"  {k}: {v}")

    with open(os.path.join(run_dir, "runtime_telemetry.json"), "r") as f:
        rt = json.load(f)
    vram = rt.get("vram", {})
    baseline = vram.get("post_warmup_baseline")
    peak = vram.get("peak")
    print(f"baseline: {baseline}")
    print(f"peak: {peak}")
    print(f"allocated peak growth: {peak['allocated_mib'] - baseline['allocated_mib']:.2f} MiB")
    print(f"reserved peak growth: {peak['reserved_mib'] - baseline['reserved_mib']:.2f} MiB")

if __name__ == "__main__":
    main()
