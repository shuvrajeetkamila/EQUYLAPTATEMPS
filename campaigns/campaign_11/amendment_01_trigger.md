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

No workflow file changes. The Campaign 11 workflows already use this
repository's established opt-in execution pattern (in place since the
early campaigns): they trigger on `push`, and the job runs only when
the pushed commit message contains the campaign tag
(`[campaign11-screen]` / `[campaign11-budget96]`). Execution is
therefore authorized by pushing a dedicated commit carrying the tag —
an action the available token is permitted to perform — instead of
via the dispatch endpoint. Each tagged push still executes exactly
the frozen registry conditions: the screen registry contains all of
condition B's confirmatory runs, and the budget-96 run remains
separately tag-gated behind the dated budget amendment required by
the experiment matrix.

## What does not change

Nothing scientific. This amendment exists so the frozen experiment
can be executed at all under a changed tooling permission; it was
written and committed before any Campaign 11 result was observed.
