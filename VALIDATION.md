# Validating the behavior-adherence rater

The headline metric is only as trustworthy as the LLM-judge that produces it.
This is the plan to make the rater credible (and to drop the "experimental /
unvalidated" caveat).

## The gap
The released CONFLICTS dataset has gold **conflict-type** labels but **no human
behavior-adherence judgments on candidate answers**. The paper validated its
rater against humans on 100 examples (0.89 accuracy), but that rater-validation
set and the Appendix B judge prompts are **not in the release**. So we cannot
reproduce 0.89 off the shelf, and our prompts are a *reconstruction*.

## Two paths

### Path A — author cooperation (preferred, fast)
Obtain from the authors (Cattan et al.):
1. the 100-example rater-validation set (candidate responses + human adherence labels), and
2. the Appendix B judge-prompt templates.

Then: validate our reconstructed rater directly against their human labels, and
diff our prompts against theirs. If agreement is high, we ship a *faithful,
validated* open implementation with credit. This is days, not weeks.

### Path B — self-built validation (if no cooperation / while we wait)
Build our own human-validated set:
1. **Generate candidate answers** from 3–5 distinct systems/prompts over the
   dataset, **stratified by conflict type** (don't let No-conflict dominate).
2. **2 independent annotators, blind** to model + judge output, label behavior
   adherence per the rubric; **adjudicate** disagreements.
3. Report **human–human agreement** AND **judge-vs-adjudicated accuracy/F1 per
   type** (not just aggregate).
4. **Freeze** judge model + `PROMPT_VERSION` before annotating.
5. **Misinformation excluded** (n=5; also needs reliability evidence the judge
   lacks). ~120 examples: ~20 each for the 4 usable types + 20 No-conflict; 20
   synthetic stress cases as diagnostics only, not the headline number.

Multi-judge agreement is for prompt debugging only — correlated judges share
bias, so it does not validate correctness.

## Already possible now (no humans needed)
The **conflict-type classifier** can be benchmarked directly against the 458 gold
type labels (macro-F1 + confusion matrix). That is a separate, immediately
reportable number (classifier is a v2 utility, not in the oracle scoring path).

## What validation unlocks
- Drop the "experimental / unvalidated rater" caveat.
- A short empirical write-up (rater agreement + a small leaderboard of systems'
  behavior-adherence rates across types).
- Decision point: flip the repo public.
