"""Summarise jegan_check_scores.json: real voices wrongly flagged and fakes caught, per detector configuration.

    python3 "/Users/shivashankar/Downloads/Shield own voices/jegan_check_report.py"
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "jegan_check_scores.json").read_text())

CONFIGS = {   # name -> clip is called synthetic?
    "Shield today (AASIST, mean, -1.14)": lambda r: r["ours_p"] >= 0.5,
    "AASIST, worst window, -1.14": lambda r: min(r["v1"]) < -1.140625,
    "AASIST, worst window, 0": lambda r: min(r["v1"]) < 0,
    "AASIST3, worst window, 0": lambda r: min(r["v3"]) < 0,
    "AASIST3, mean, 0": lambda r: sum(r["v3"]) / len(r["v3"]) < 0,
    "Jegan: either model, worst window, 0": lambda r: min(min(r["v1"]), min(r["v3"])) < 0,
}


def main() -> None:
    lines = ["# Jegan's ensemble vs Shield's detector", "",
             "Real flagged = genuine human clips called synthetic (false alarms). Fakes caught = synthetic clips called "
             "synthetic. Phone line = simulated 8 kHz G.711.", ""]
    for key, rows in DATA.items():
        real = [r for r in rows if r["label"] == "bonafide"]
        fake = [r for r in rows if r["label"] == "spoof"]
        attacks = sorted({r["attack"] for r in fake})
        lines += [f"## {key}  ({len(real)} real, {len(fake)} fake)", "",
                  "| Configuration | Real flagged | Fakes caught | " + " | ".join(attacks) + " |",
                  "|---|---|---|" + "---|" * len(attacks)]
        for name, flag in CONFIGS.items():
            by = defaultdict(lambda: [0, 0])
            for r in fake:
                by[r["attack"]][0] += flag(r)
                by[r["attack"]][1] += 1
            fp = sum(flag(r) for r in real)
            lines.append(f"| {name} | {fp}/{len(real)} | {sum(flag(r) for r in fake)}/{len(fake)} | "
                         + " | ".join(f"{by[a][0]}/{by[a][1]}" for a in attacks) + " |")
        ens = sorted(real, key=lambda r: min(min(r["v1"]), min(r["v3"])))[:3]
        lines += ["", "Lowest real clips under Jegan's rule (ensemble worst logit; < 0 = flagged): "
                  + ", ".join(f"{r['id']} {min(min(r['v1']), min(r['v3'])):+.2f} ({len(r['v1'])} win)" for r in ens), ""]
    report = "\n".join(lines)
    print(report)
    (HERE / "jegan_check_report.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
