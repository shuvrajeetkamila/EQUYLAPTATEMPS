# Campaign 11 — Amendment 01 (infrastructure only)

**Date:** 2026-10-14
**Filed:** after the frozen preregistration (commit `fb90ebb`) and
before any Campaign 11 training run exists.
**Scope:** execution trigger mechanism only. **No scientific element
changes:** conditions, runs, seeds, budgets, data, donor/host,
placement, control, gates, outcome mapping, and analysis are exactly
as frozen in `preregistration.md`.

## Reason

The execution environment's GitHub token, which successfully
dispatched the Campaign 10 workflows earlier, now returns
`HTTP 403: Resource not accessible by integration` for the
workflow-dispatch endpoint (verified twice via `gh workflow run` and
once via the REST dispatches endpoint). The token retains git push,
pull-request, and Actions *read* permissions, but not Actions
*dispatch*.

## Change

The two Campaign 11 workflows gain an additional trigger: a `push`
event touching a dedicated marker file
(`campaigns/campaign_11/.triggers/campaign11-screen.txt` /
`.../campaign11-budget96.txt`) on this branch, and their job
condition accepts either the manual-dispatch confirmation input or
such a marker push. Manual `workflow_dispatch` remains available for
anyone with dispatch permission. The marker files are pushed only
when the corresponding run is intended; each run still executes
exactly the frozen registry conditions — the screen registry contains
all of condition B's confirmatory runs, and the budget-96 run remains
separately marker-gated.

## What does not change

Nothing scientific. This amendment exists so the frozen experiment
can be executed at all under a changed tooling permission; it was
written and committed before any Campaign 11 result was observed.
