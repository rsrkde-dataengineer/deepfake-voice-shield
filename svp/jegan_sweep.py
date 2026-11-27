"""Sweep aggregations/thresholds over jegan_check_scores.json: modern fakes caught (Voice samples) vs real voices flagged."""
import json, numpy as np
d = json.load(open("jegan_check_scores.json"))
mean = lambda v: float(np.mean(v))
kth = lambda v, k=2: sorted(v)[min(k - 1, len(v) - 1)]
aggs = {
    "v1 mean": lambda r: mean(r["v1"]), "v3 mean": lambda r: mean(r["v3"]),
    "v1 min": lambda r: min(r["v1"]), "v3 min": lambda r: min(r["v3"]),
    "v1 2nd-min": lambda r: kth(r["v1"]), "v3 2nd-min": lambda r: kth(r["v3"]),
    "min(v1min,v3min)": lambda r: min(min(r["v1"]), min(r["v3"])),
    "min(v1mean,v3mean)": lambda r: min(mean(r["v1"]), mean(r["v3"])),
    "avg of v1,v3 per window, min": lambda r: min((a + b) / 2 for a, b in zip(r["v1"], r["v3"])),
    "avg of v1,v3 per window, mean": lambda r: mean([(a + b) / 2 for a, b in zip(r["v1"], r["v3"])]),
}
def stats(key, f, t):
    rows = d.get(key, [])
    real = [r for r in rows if r["label"] == "bonafide"]; fake = [r for r in rows if r["label"] == "spoof"]
    return sum(f(r) < t for r in real), len(real), sum(f(r) < t for r in fake), len(fake), \
           sum(f(r) < t for r in fake if "clone" in r["id"])
sets = [k for k in ("samples/clean", "samples/telephone", "test-clean/clean", "test-clean/telephone",
                    "dev-clean/clean", "dev-clean/telephone") if k in d]
print("config | thr | " + " | ".join(sets))
for name, f in aggs.items():
    for t in (0.0, -0.5, -1.0, -1.140625, -1.5, -2.0):
        cells = []
        for k in sets:
            fp, nr, tp, nf, cl = stats(k, f, t)
            cells.append(f"FP {fp}/{nr} TP {tp}/{nf}" + (f" app-clones {cl}/2" if k.startswith("samples") else ""))
        print(f"{name} | {t:+.2f} | " + " | ".join(cells))
