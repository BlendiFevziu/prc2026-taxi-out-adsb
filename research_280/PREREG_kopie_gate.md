# Pre-registration: copy-labelled fallback gate

Written 23.09.2026 **before any model was fitted**. Nothing here may be edited after the first
fit; changes go in a dated addendum below.

## The claim being tested

The fallback classifier learns `z = y > 5400 s`, the **consequence**. The main lane learns
`erf = |y - versp| <= 10 s`, the **mechanism**, at AUC 0.8759. Measured 23.09.2026
(`fallback_reachability.py`): of the 481 fallback rows above two hours that carry 92.88 % of
fallback squared error, **397 are copies** where `y == versp` exactly (max deviation 0.0 s) and
84 are broken records where `versp` says nothing. `main_model.py` already records this split as
"79 % auf vier Sekunden genau, fuer die uebrigen 21 % voellig falsch".

The consequence label is therefore wrong twice: it pools two mechanisms that need opposite
treatment, and it marks every copy with a small delay as a negative example.

If the mechanism label is right, then under squared error, with `h = E[y|copy] = versp` and
`m = E[y|not copy]`, the optimal blend weight is exactly the calibrated probability,
`w* = E[e*d]/E[d^2] = q`. Both tuned constants (`HOCH = 2.5`, `ANTEIL_HM = 0.3`) become
unnecessary. Both were set on the leaderboard on 09.09.2026 against a ~276 model and are the
last two leaderboard-tuned values in the system.

## Instrument calibration, runs first

`fallback_cv.py` carries its own probe: the scored set says HOCH 1.6 beats 0.8 by 0.509 s. If the
six-pair harness does not reproduce that ordering, **no arm is interpreted at all**.

## Rule calibration, runs second, and it is the point of this section

Previous fallback screens were judged partly by "does the pooled gain survive removing its best
100 rows". I claim that rule is structurally unusable on this lane, because its entire value sits
in under 500 rows. **I test that claim on a known positive control before applying the rule to my
own candidate.** The control is the height apparatus itself: the scored set measured that turning
it off (`HOCH = 50`) costs **6.53 s**, the largest single measured contribution in the model.

- If `HOCH = 2.5` against `HOCH = 50` **fails** the top-100 rule, then the rule rejects a lever the
  scored set says is worth 6.53 s, and it is reported as uninformative for this lane.
- If the control **passes**, the rule stands and my candidate must pass it too.

This is decided by the control, not by whether my own arms happen to pass.

## Arms

| arm | classifier label | blend weight | height branch | normal regressor |
|---|---|---|---|---|
| A0 prod | `y > 5400` | `p ** 2.5` | `0.7*versp + 0.3*hm` | rows with `z == 0` |
| A1 label | `\|y-versp\| <= 10` | `p ** 2.5` | `0.7*versp + 0.3*hm` | non-copy rows |
| A2 weight | `y > 5400` | `p` | `0.7*versp + 0.3*hm` | rows with `z == 0` |
| A3 full | `\|y-versp\| <= 10` | `q` | `versp` exactly | non-copy rows |
| A4 full+cal | `\|y-versp\| <= 10` | isotonic `q`, fitted out of fold on learning months only | `versp` exactly | non-copy rows |

A1 and A2 exist to attribute any effect to the label or to the exponent separately. A0 is the
scored recipe and must reproduce it.

## Decision rules, fixed now

Primary, low variance, and the reason this study exists:
1. **Classifier quality.** On the copy event, out of fold, the mechanism label must beat the
   consequence label on both AUC and Brier score. This is a binomial quantity with no tail.
2. **Calibration.** The winning classifier's reliability curve must be closer to the diagonal
   than `p ** 2.5` is, measured by expected calibration error over ten bins.

Secondary, high variance, reported for every arm:
3. Pooled six-pair leaderboard effect positive.
4. At least 4 of 6 pairs positive.
5. Ordinary rows (`y <= 5400`) not worse.
6. Top-100 removal, **interpreted only if the positive control above survives it**.

## Transfer check, no labels and no leaderboard

The 08.09.2026 sign error was diagnosed in `main_model.py` as a 2025 to 2026 calibration shift
("ist er dort zuversichtlicher oder anders geeicht"). It was never measured directly. It can be:
fit the classifier on all of 2025 and apply it to the 2026 fallback rows in `ranking.parquet`,
then compare the two **score distributions**. No labels are read and no submission is made.

## What happens with the result

Nothing is uploaded under any outcome. If the primary rules pass, a candidate is built and
reported, and the decision to submit is the owner's. If they fail, the line is closed and
recorded like the others.

---

## Addendum, 23.09.2026, written after the first six-pair run and before the A5/A6 fits

**Two things the run established, both recorded before any new arm was fitted.**

1. **The instrument fails its own calibration probe**, reproducing experiments.csv row 71 of
   08.09.2026. It ranks HOCH 0.8 above 1.6 by +1.311 s [+0.161, +2.541]; the scored set says 1.6
   wins by 0.509 s. The six-pair MSE comparison therefore cannot adjudicate any arm, and rules 3
   to 5 are void.
2. **The rule control settles rule 6 as pre-registered.** The height apparatus, which the scored
   set values at 6.53 s, gains 492,856 MSE per row pooled and loses 14,974 without its best 100
   rows. The top-100 rule rejects it, so the rule is uninformative on this lane, exactly as the
   pre-registration anticipated. This was decided by the control, not by my arms.

**A defect in my own arm design, named rather than quietly fixed.** A1 was described as "label
only". It is not. The label also defines the partition that trains the normal and height
regressors, so switching to `kopie` moves the 84 broken extremes into the normal regressor's
training data. A1, A3 and A4 all lose about 400,000 MSE per **ordinary** row, and ordinary rows
have bounded labels, so that is real damage and not a tail artefact. The conclusion is that those
three arms are genuinely bad as specified, and that the label change and the partition change were
confounded.

**Two arms added**, with the classifier target and the regressor partition separated:

| arm | classifier label | regressor partition | weight | height |
|---|---|---|---|---|
| A5 | `kopie` | `z` (production) | `p ** 2.5` | `0.7*versp + 0.3*hm` |
| A6 | `kopie` | `z` (production) | `p` | `0.7*versp + 0.3*hm` |

**Judged by `kopie_expected_gain.py`, not by the six-pair MSE**, since the latter is now known
broken here. Nothing is uploaded under any outcome.

---

## Second addendum, 23.09.2026. The central claim of this pre-registration is refuted.

**What the evidence says.** The mechanism label is the better copy detector, by a wide margin and
in a low-variance measurement: AUC on the copy event 0.9613 against 0.7566, Brier 0.02464 against
0.03088, and at a 0.5 operating point it has higher precision (0.888 vs 0.867), higher recall
(0.465 vs 0.410) and fewer catastrophic false positives (35 vs 49).

**And it is the worse decision rule anyway.** Mean blend weight assigned to TRUE copies, by size:

| label band | n | A0 (`y>5400`) | A5 (`kopie`) |
|---|---|---|---|
| 0 to 1,800 s | 392 | 0.003 | 0.066 |
| 1,800 to 5,400 s | 115 | 0.008 | 0.456 |
| 5,400 to 10,800 s | 432 | 0.716 | 0.696 |
| 21,600 to 50,000 s | 42 | 0.996 | 0.787 |
| above 50,000 s | 9 | 0.997 | 0.769 |

On the 51 largest copies A0 flags 51 of 51, A5 flags 41. A5 wins only below 5,400 s, where
`versp` is close to the normal prediction, `D = (height - normal)^2` is near zero and the weight
barely changes the answer.

**So the consequence label is not a modelling error, it is implicit loss weighting.** Training on
`y > SCHWELLE_EXTREM` concentrates the classifier on the rows where squared error lives. My
argument that it "mislabels 528 copies as negatives" was right about the label and wrong about the
consequence: those 528 rows carry almost no loss.

**Exploratory, generated after seeing the above, and marked as such.** Arms A7 and A8 keep the
mechanism label but weight the classifier's training rows by `1 + (|versp - airport median| /
3600)^g`, a model-free stand-in for the scale of D, with `g = 1` and `g = 2`. The pre-registered
acceptance criteria do not apply to arms chosen this way. They are judged on two bounded
quantities and reported as a hypothesis for a later confirmatory test, never as a result:
mean weight on the 51 largest true copies (must reach A0's 0.996) and mean weight on large
non-copies (must stay below A0's 0.238). Nothing is uploaded.
