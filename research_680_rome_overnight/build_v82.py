"""Build v82 = v81 plus the Rome overnight rule of research_680_rome_overnight/NOTE_rome_overnight.md. NM-free, no fit.
No upload. Run from the repository root:  python3 -u -B research_680_rome_overnight/build_v82.py
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_500_midnight"):
    sys.path.insert(0, str(p))
import rule as R  # noqa: E402
import submission as SUB  # noqa: E402

ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 245.020
V81 = ROOT / ".cache/research_650/v81/candidate_v81.parquet"
OUT = ROOT / ".cache/research_680/v82"
REPORT = ROOT / "research_680_rome_overnight/results/v82_build.json"
LONG, MID = 14 * 3600.0, 8 * 3600.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def y24():
    d = pd.concat([pd.read_parquet(p, columns=R.COLS + [R.T]) for p in sorted((ROOT / "data").glob("training_2025-*.parquet"))], ignore_index=True)
    d = R.annotate(d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF") & d.AOBT_3_flt.isna()])
    kind = R.label_kind(d[R.T], d.delay)
    v = d.loc[kind == "shift24", R.T].to_numpy(float)
    return float(np.median(v)), int(len(v))


def rows_2026():
    k = pd.read_parquet(ROOT / "data/ranking.parquet", columns=R.COLS)
    k = R.annotate(k[k.PHASE_mvt.eq("DEP") & k.ADEP_mvt.eq("LIRF") & k.AOBT_3_flt.isna()]).reset_index(drop=True)
    k = k[k.crossing.eq(1) & k.delay.gt(MID)].reset_index(drop=True)
    k["part"] = np.where(k.delay >= LONG, "a", "b")
    return k


def main():
    rec81 = json.loads((ROOT / "research_650_eham_cat/results/v81_build.json").read_text())
    checks = {"v81_is_built_file": sha(V81) == rec81["candidate_sha256"]}
    Y24, n24 = y24()
    checks["y24"], checks["shift24_rows_2025"] = Y24, n24
    if n24 != 12:
        raise SystemExit(f"expected twelve 2025 +24 h rows, found {n24}")
    v81 = pd.read_parquet(V81)
    at81 = v81.set_index(ID)[T]
    k = rows_2026()
    k["old"] = at81.loc[k[ID]].to_numpy(float)
    d = k.delay.to_numpy(float)
    k["new"] = np.where(k.part.eq("a"), (d + Y24) / 2, d)
    checks["rows"] = {p: int(k.part.eq(p).sum()) for p in ("a", "b")}
    new = pd.Series(k.new.to_numpy(), index=k[ID].to_numpy())
    cand = v81.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v81.loc[~hit].reset_index(drop=True))
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v81=bool(untouched), all_finite=bool(np.isfinite(cand[T]).all()),
                  rows_changed=int((cand[T].to_numpy() != v81[T].to_numpy()).sum()))
    new_vals = cand.set_index(ID)[T].loc[k[ID]].to_numpy(float)
    k["new_written"] = new_vals
    # (b): valued at its rows being copies (y = delay)
    sse_b = float((((k.old - k.delay) ** 2) - ((k.new_written - k.delay) ** 2))[k.part.eq("b")].sum())
    # (a): outcome distribution, each row copy (y = delay) or +24 h (y = Y24)
    a = k[k.part.eq("a")]
    gain_copy = ((a.old - a.delay) ** 2 - (a.new_written - a.delay) ** 2).to_numpy()
    gain_24 = ((a.old - Y24) ** 2 - (a.new_written - Y24) ** 2).to_numpy()
    s = lambda g: round(float(np.sqrt(REF ** 2 - g / N)), 3)
    expect = lambda p: float(np.sum(p * gain_24 + (1 - p) * gain_copy)) + sse_b
    outcomes = sorted(float(sum(gain_24[i] if bit else gain_copy[i] for i, bit in enumerate(bits))) + sse_b
                      for bits in itertools.product((0, 1), repeat=len(a)))
    report = {"checks": checks, "base": {"version": "v81", "central": REF, "sha256": rec81["candidate_sha256"]},
              "rows": k[[ID, "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "delay", "part", "old", "new_written"]].astype(str).to_dict("records"),
              "gain_sse_b_if_copies": sse_b, "gain_sse_a_per_row": {"if_copy": gain_copy.tolist(), "if_plus24": gain_24.tolist()},
              "prediction_fixed_before_upload": {
                  "assumptions": "(a) each row +24 h with probability p, independent; (b) copies. Central p = 0.41, best p = 0.58, worst every (a) row a copy",
                  "central": s(expect(0.41)), "best": s(expect(0.58)), "worst": s(outcomes[0]), "all_plus24": s(outcomes[-1]),
                  "outcome_rmse_quantiles_p05": {q: s(float(np.quantile(outcomes, 1 - q))) for q in (0.1, 0.5, 0.9)}}}
    ok = not errors and untouched and checks["all_finite"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v82.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
