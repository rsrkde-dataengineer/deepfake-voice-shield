"""Fair comparison: fix each detector's threshold on dev-clean real voices (clean + phone line) at a target false-alarm
rate, then count modern fakes caught (Voice samples: GPT/Gemini/Grok + app clones, clean + phone) and test-clean
false alarms at that same threshold."""
import json, math
import numpy as np
d = json.load(open("jegan_check_scores.json"))
logit = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))
AGG = {   # higher = more human
    "Shield today (AASIST mean)": lambda r: -logit(r["ours_p"]),
    "AASIST3 mean": lambda r: float(np.mean(r["v3"])),
    "Jegan ensemble (worst window)": lambda r: min(min(r["v1"]), min(r["v3"])),
    "Both models, mean of windows": lambda r: min(float(np.mean(r["v1"])), float(np.mean(r["v3"]))),
}
def real(keys): return [r for k in keys for r in d[k] if r["label"] == "bonafide"]
dev = real(["dev-clean/clean", "dev-clean/telephone"])
tst = real(["test-clean/clean", "test-clean/telephone"])
modern = [r for k in ("samples/clean", "samples/telephone") for r in d[k] if r["label"] == "spoof"]
convo = [r for k in ("samples/clean", "samples/telephone") for r in d[k] if r["label"] == "bonafide"]
ours_fakes = [r for k in ("test-clean/clean", "test-clean/telephone") for r in d[k] if r["label"] == "spoof"]
print(f"dev real {len(dev)}, test real {len(tst)}, modern fakes {len(modern)} (16 clips x 2 channels), our test fakes {len(ours_fakes)}")
print("| Detector | dev false-alarm target | threshold | test-clean real flagged | modern fakes caught | app clones | our fakes | team real flagged |")
print("|---|---|---|---|---|---|---|---|")
for name, f in AGG.items():
    s = sorted(f(r) for r in dev)
    for target in (0.0, 0.01, 0.02, 0.05):
        k = int(math.floor(target * len(s)))            # allow k dev false alarms
        t = s[k] if k < len(s) else s[-1]                # flagged if score < t  -> exactly k dev clips below
        fp_t = sum(f(r) < t for r in tst)
        cat = sum(f(r) < t for r in modern)
        app = sum(f(r) < t for r in modern if "clone" in r["id"])
        of = sum(f(r) < t for r in ours_fakes)
        cv = sum(f(r) < t for r in convo)
        print(f"| {name} | {target:.0%} | {t:+.2f} | {fp_t}/{len(tst)} | {cat}/{len(modern)} | {app}/4 | {of}/{len(ours_fakes)} | {cv}/{len(convo)} |")
