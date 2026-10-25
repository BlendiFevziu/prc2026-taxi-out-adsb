"""Which constants in this codebase have ever been tuned, and which are bare guesses?

Three of the four gains on 22 September came from constants nobody had checked: ANTEIL_NETZ, the NM add-on's
SHRINK, and the size of the model behind it. Two others turned out to be measured after all (SCHWELLE_EXTREM was
swept on 04.09, ANTEIL_FLH and ANTEIL_CB are at their optimum). The difference was never visible without opening
each file, which is why this inventory exists: it is cheaper to read than to rediscover.

A constant counts as EVIDENCED when the comment block immediately above it cites a measurement: a number with a
unit, a grid, a date, or a comparison. Everything else is flagged. That is a heuristic on comments, not proof,
so a flag means "nobody wrote down a reason", not "wrong".

  python3 -B research_280/constant_audit.py
"""
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "research_280/results/constant_audit.json"
FILES = ["main_model.py", "neural_model.py", "submission.py", "features_stand_rate.py",
         "features_target_encoding.py", "research_250/release_common.py",
         "research_260/sequence_model.py", "research_260/sequence_features.py"]
NAME = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*=\s*(.+?)\s*(?:#.*)?$")
EVIDENCE = re.compile(r"\d[\d.,]*\s*(s\b|Sekunden|MSE|RMSE|Prozent|%)|"
                      r"\bgemessen\b|\bstatt\b|\bgegen\b|->|\d{2}\.\d{2}\.20\d\d|"
                      r"\bRangliste\b|\bHaelfte\b|\bFlaeche ist flach\b", re.I)


def main():
    rows = []
    for rel in FILES:
        path = PROJECT / rel
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            m = NAME.match(line)
            if not m or m.group(2).startswith(("[", "{")) and len(m.group(2)) > 90:
                continue
            block, j = [], i - 1
            while j >= 0 and lines[j].lstrip().startswith("#"):
                block.append(lines[j]); j -= 1
            comment = " ".join(reversed(block))
            rows.append({"file": rel, "name": m.group(1), "value": m.group(2)[:60],
                         "comment_lines": len(block), "evidenced": bool(EVIDENCE.search(comment)),
                         "hint": comment[:110]})
    flagged = [r for r in rows if not r["evidenced"]]
    print(f"{len(rows)} module-level constants scanned; {len(flagged)} carry no measurement in their comment\n")
    print(f"{'file':34s} {'name':24s} {'value':22s} comment?")
    for r in sorted(flagged, key=lambda r: (r["file"], r["name"])):
        print(f"{r['file']:34s} {r['name']:24s} {r['value']:22s} "
              f"{'none' if not r['comment_lines'] else str(r['comment_lines'])+' lines, no number'}")
    print(f"\nFor contrast, {len(rows)-len(flagged)} DO cite a measurement, e.g.:")
    for r in [x for x in rows if x["evidenced"]][:6]:
        print(f"  {r['name']:20s} {r['value']:14s} <- {r['hint'][:80]}")
    OUTPUT.write_text(json.dumps({"note": "heuristic on comments: a flag means nobody wrote down a reason, "
                                          "not that the value is wrong", "constants": rows}, indent=2) + "\n")
    print(f"\nwritten {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
