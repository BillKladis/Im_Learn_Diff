"""training_status.py — Quick dashboard for all active training runs."""
import glob, os, re, sys

LOGS = [
    ("/tmp/hw_v2_continue.log",    "v2_cont  (σ noise,   u=0.15)", r"\[\s*(\d+)\].*?([\d.]+%|nan).*?([\d.]+%|—)"),
    ("/tmp/hw_v3_u010.log",        "v3_u010  (double,    u=0.10)", r"\[\s*(\d+)\].*?([\d.]+%|—)"),
    ("/tmp/hw_v4_u007.log",        "v4_u007  (double,    u=0.07)", r"\[\s*(\d+)\].*?([\d.]+%|—)"),
    ("/tmp/hw_v5_sa015.log",       "v5_sa015 (single,    u=0.15)", r"\[\s*(\d+)\].*?([\d.]+%|—)"),
    ("/tmp/hw_v6_sa010.log",       "v6_sa010 (single,    u=0.10)", r"\[\s*(\d+)\].*?([\d.]+%|—)"),
    ("/tmp/eval_noise_compare.log","noise_eval (v1 vs v2)",         r"(obs σ=[\d.]+|Done\.)"),
]

CHECKPOINTS = "saved_models"

print(f"\n{'='*70}")
print(f"  TRAINING STATUS")
print(f"{'='*70}")

for log_path, name, pattern in LOGS:
    try:
        with open(log_path) as f:
            lines = f.readlines()
        # Find last training row (starts with spaces + [ or obs)
        last_ep = None
        last_f01 = "—"
        for line in reversed(lines):
            m = re.search(r'\[\s*(\d+)\]', line)
            if m:
                last_ep = int(m.group(1))
                f_match = re.search(r'(\d+\.\d+%)', line)
                if f_match:
                    last_f01 = f_match.group(1)
                break
        noise_m = None
        if "noise_eval" in name:
            for line in reversed(lines):
                if "obs σ" in line or "Done" in line:
                    noise_m = line.strip()
                    break
        if noise_m:
            print(f"  {name:<35}  {noise_m}")
        elif last_ep is not None:
            print(f"  {name:<35}  ep={last_ep:>4}  f01={last_f01:>7}")
        else:
            print(f"  {name:<35}  (no data yet)")
    except FileNotFoundError:
        print(f"  {name:<35}  (log not found)")

# Checkpoints
print(f"\n  {'─'*65}")
dirs = sorted(glob.glob(f"{CHECKPOINTS}/hw_v[2-6]*"), key=os.path.getmtime, reverse=True)
print(f"  Saved checkpoints (newest first):")
for d in dirs[:8]:
    size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) // 1024
    print(f"    {os.path.basename(d):<52}  {size:>5} KB")

print(f"{'='*70}\n")
