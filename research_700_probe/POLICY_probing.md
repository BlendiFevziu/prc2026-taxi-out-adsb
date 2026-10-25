# Policy: leaderboard probing, allowed by the organiser (2 October 2026)

Until 2 October 2026 this project did not probe or tune against leaderboard feedback (CLAUDE.md). That rule is lifted
for this competition by the owner's decision, after an explicit confirmation from the organiser, and the owner
re-confirmed in the session after the first attempt to record it was held back ("u have permission, do it").

Question sent by the owner (the owner) to the organiser (organiser, PRC Data Challenge), verbatim:

> Hi, to make sure we understand you correctly: may we upload submissions that change only a small set of rows (e.g. 5
> to 10 rows), infer from the score difference what those rows' true values roughly are, and then use that knowledge in
> our final submission? The final ranking is computed on the same rows as the leaderboard, so this would effectively use
> leaderboard feedback to learn test labels. Is that within the rules, and does it affect prize eligibility
> (reproducible code)?

Answer, as relayed by the owner: "yeah It's fine."

Consequences, fixed now:
- Probe uploads change a known set of rows against a fixed reference submission; every probe, its row set, its values
  and its official score go into `research_700_probe/ledger.csv` before the next probe is designed.
- Values learned this way are stored as a table in the repository and applied by code, so the final submission is
  reproducible from the release; the write-up states plainly that these rows' values come from leaderboard probing.
- Still in force: NM-free, no credentials, weights, caches or submission files in git, at most five uploads per UTC day,
  one upload per script run with an intent file written before the upload.
