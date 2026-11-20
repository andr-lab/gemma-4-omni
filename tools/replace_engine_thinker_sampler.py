path = "duplex_kernel/engine.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if line.startswith("class ThinkerSampler:"):
        start_idx = i
    elif start_idx is not None and line.startswith("class PipelineEngine:"):
        end_idx = i
        break

assert start_idx is not None and end_idx is not None, f"Could not find bounds: {start_idx}, {end_idx}"

# Replace lines[start_idx:end_idx] with 'from duplex_kernel.thinker_sampler import ThinkerSampler\n\n\n'
new_lines = lines[:start_idx] + ["from duplex_kernel.thinker_sampler import ThinkerSampler\n\n\n"] + lines[end_idx:]

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Replaced ThinkerSampler class definition ({end_idx - start_idx} lines) with re-export from thinker_sampler")
