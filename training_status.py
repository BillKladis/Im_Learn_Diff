"""training_status.py — Quick dashboard for all active training runs."""
import glob, os, re, sys

LOGS = [
    ("/tmp/hw_v2_continue.log",    "v2_cont  (σ noise,   u=0.15)"),
    ("/tmp/hw_v3_u010.log",        "v3_u010  (double,    u=0.10)"),
    ("/tmp/hw_v4_u007.log",        "v4_u007  (double,    u=0.07)"),
    ("/tmp/hw_v5_sa015.log",       "v5_sa015 (single,    u=0.15)"),
    ("/tmp/hw_v6_sa010.log",       "v6_sa010 (single,    u=0.10)"),
    ("/tmp/eval_noise_compare.log","noise_eval (v1 vs v2)        "),
]

CHECKPOINTS = "saved_models"

print(f"\n{'='*80}")
print(f"  TRAINING STATUS")
print(f"{'='*80}")
print(f"  {'Run':<35}  {'ep':>4}  {'last f01':>8}  {'best f01':>8}  {'best ep':>7}")
print(f"  {'─'*72}")

for log_path, name in LOGS:
    try:
        with open(log_path) as f:
            lines = f.readlines()

        if "noise_eval" in name:
            noise_m = None
            for line in reversed(lines):
                if "obs σ" in line or "Done." in line:
                    noise_m = line.strip()
                    break
            status = noise_m if noise_m else "(no data yet)"
            print(f"  {name:<35}  {status}")
            continue

        last_ep = None
        last_f01 = None
        best_f01 = None
        best_ep  = None

        for line in lines:
            m = re.search(r'\[\s*(\d+)\]', line)
            if m:
                ep = int(m.group(1))
                f_match = re.search(r'(\d+\.\d+)%', line)
                if f_match:
                    f = float(f_match.group(1))
                    if best_f01 is None or f > best_f01:
                        best_f01 = f
                        best_ep  = ep
                    last_f01 = f
                last_ep = ep

        ep_str   = f"{last_ep:>4}" if last_ep is not None else "   —"
        lf_str   = f"{last_f01:>7.1f}%" if last_f01 is not None else "       —"
        bf_str   = f"{best_f01:>7.1f}%" if best_f01 is not None else "       —"
        bep_str  = f"{best_ep:>7}" if best_ep is not None else "      —"
        print(f"  {name:<35}  {ep_str}  {lf_str}  {bf_str}  {bep_str}")

    except FileNotFoundError:
        print(f"  {name:<35}  (log not found)")

# Checkpoints
print(f"\n  {'─'*72}")
dirs = sorted(glob.glob(f"{CHECKPOINTS}/hw_v[2-6]*"), key=os.path.getmtime, reverse=True)
print(f"  Saved checkpoints (newest first):")
for d in dirs[:10]:
    size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) // 1024
    print(f"    {os.path.basename(d):<55}  {size:>5} KB")

print(f"{'='*80}\n")
