# Campaign 13 — Control Design

## Pair integrity (validated before training, job `[campaign13-validate]`)

- Both parent snapshots acquired at the frozen revisions
  (`acquire` refuses if the Hub returns a different SHA).
- Provenance records `mode: same_family_positive_control` and
  `tokenizers_identical` (expected **false**: the Instruct
  tokenizer adds chat configuration; vocabulary size identical;
  the child uses the recipient tokenizer for all training and
  evaluation, so every confirmatory comparison is
  tokenizer-matched).
- Config comparison recorded (architectural fields identical;
  special-token metadata differences documented in
  `model_pair_selection.md`).
- Donor sector delta vs the recipient's own layers [24,30):
  5.64% mean relative RMS (probe) — the learned difference the
  scrambles ablate is real and quantified.

## Conditions B and C (scrambles of the Instruct sector)

Implementations are Campaign 11/12's, unchanged:
`row_shuffled_sector_control` (B) and `shuffled_sector_control`
(C), seed 20241014, eligibility = the 42 two-dimensional sector
matrices, 12 RMSNorm gains excluded and preserved. Because the
architecture is the same Llama block family, no adaptation was
needed; both controls were verified at construction in the probe
(42 tensors, exact multisets incl. per-row for B, fingerprints
changed, graft finite, parents unchanged).

## Condition D (random twin)

Established random-sector control, seed 1234, identical interface
treatment. Probe initial activity 0.0011 (near-silent at init, as
in Campaigns 9–12); trained activity will be reported per run and
D is never validity-gated evidence.

## Condition F (mechanical self-reconstruction)

Separate registry with donor := the recipient checkpoint itself
(`c13_self_42`, 32 steps, seed 42), executed in its own job. It
checks the pipeline reconstructs a functioning child when the
graft is the recipient's own sector. It is labeled
`F_self_reconstruction` everywhere and excluded from all contrasts
and gates.

## Activity comparability

No calibration forces activity matching. Pre-training validation
records initial graft activity per condition; each run records
trained graft relative update and bypass/zero ablations, checked
against the frozen validity list in the preregistration —
implemented verbatim in the publisher and unit-tested against
hand-computed examples before training (Campaign 12 lesson).
The stricter bypass<zero ordering is kept as a reported,
non-gating diagnostic per run.
