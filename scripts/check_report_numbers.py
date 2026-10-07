"""Flag numbers written in report/main.tex that do not trace to any saved result.

Every decimal number in the LaTeX source (comments stripped) is looked up among the values in the summary-level
results/**/*.json and *.csv files (per-example files are skipped), rounded to the
number of decimals written (also as a percentage, and with the sign dropped). A match is necessary, not sufficient:
check that a matched number comes from the right file. Unmatched numbers are printed with their line so they can be checked by hand; integers are skipped (counts, years, section numbers).
    python -m scripts.check_report_numbers [path/to/main.tex]
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w.])-?\d+\.\d+")


def harvest(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            harvest(v, out)
    elif isinstance(obj, list):
        for v in obj:
            harvest(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, str):
        try:
            out.append(float(obj))
        except ValueError:
            pass


# Per-example files hold thousands of values and would match almost any number by chance.
SKIP = ("per_problem", "pairs.csv", "per_prompt", "judge_cache", "judge_comparisons", "dynamic_steps", "manual_audit",
        "audit_disagreements", "variant_scores", "judge_raw_outputs")


def known_values():
    vals = []
    for p in (ROOT / "results").rglob("*.json"):
        if p.stat().st_size < 5_000_000 and not any(k in p.name for k in SKIP):
            try:
                harvest(json.loads(p.read_text(encoding="utf-8")), vals)
            except Exception:
                pass
    for p in (ROOT / "results").rglob("*.csv"):
        if p.stat().st_size < 5_000_000 and not any(k in p.name for k in SKIP):
            with open(p, encoding="utf-8", newline="") as f:
                for row in csv.reader(f):
                    harvest(row, vals)
    return vals


def main():
    tex = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "main.tex"
    vals = known_values()
    keys = {}
    for v in vals:
        for d in range(0, 5):
            for x in (v, abs(v), 100 * v, abs(100 * v), v / 1e-3):
                keys.setdefault(d, set()).add(f"{x:.{d}f}")
    bad, in_bib = 0, False
    for i, line in enumerate(tex.read_text(encoding="utf-8").splitlines(), 1):
        line = re.sub(r"(?<!\\)%.*", "", line)
        in_bib = (in_bib or "\\begin{thebibliography}" in line) and "\\end{thebibliography}" not in line
        if in_bib or "\\input{" in line:
            continue
        for m in NUM.finditer(line):
            s = m.group(0).lstrip("-")
            dec = len(s.split(".")[1])
            if s not in keys.get(min(dec, 4), set()):
                bad += 1
                print(f"line {i}: {m.group(0)}  | {line.strip()[:110]}")
    print(f"{bad} number(s) not found in results/" if bad else "All decimal numbers trace to results/.")


if __name__ == "__main__":
    main()
