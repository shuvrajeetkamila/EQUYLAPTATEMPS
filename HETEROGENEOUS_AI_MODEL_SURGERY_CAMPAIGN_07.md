# HETEROGENEOUS AI MODEL SURGERY — Campaign 7 Final Report

**Campaign 7: Cross-Pairing Replication, Architecture Compatibility, and
Pretrained Capability Transfer** · Status: **COMPLETE** · Development-only
evaluation; held-out **NOT_RUN_NO_PROMOTION**.

## Final decision: Outcome C — valid experiment, no reliable useful pretrained advantage in the new pairing

Campaign 7 changed the donor family for the first time: the Mamba-130M host
received a contiguous six-block late sector [24,30) from **SmolLM-135M**
(Llama family: GQA, RoPE, SwiGLU, RMSNorm; width 576) through full-size dense
seams, against a matched random-sector twin, at two seeds, under a
preregistration frozen before training (`campaigns/campaign_07/`).

| Seed | Pretrained macro | Random twin macro | Benefit | G-B (≥2%) | G-C (usefulness) |
|---|---|---|---|---|---|
| 42 | 8.5719 | 7.5349 | **−13.76%** | FAIL | FAIL |
| 123 | 9.4023 | 7.8900 | **−19.17%** | FAIL | FAIL |

The pretrained sector was not merely unhelpful — it was reliably **worse** than
random blocks of identical shape, and the deficit grew at the second seed.

## Answers to the campaign questions

1. **Is the new pairing technically compatible?** Yes. Llama support was added
   natively (blocks, RMSNorm, RoPE, GQA, tied embeddings); the child
   (143,721,408 params; 888,768 trainable) trained and evaluated cleanly on
   CPU in ~12-minute jobs. See `compatibility_report.md`.
2. **Does the donor sector carry signal in the child?** Yes — bypass raises
   macro from 8.57→10.93 (seed 42), zeroing to 44.07; the sector's relative
   update is ≈54–58× its input norm (twin ≈0.47). The child depends on the
   pathway. Per the frozen interpretation, dependence is not benefit.
3. **Does pretrained content beat a matched random sector?** No — at neither
   seed (table above). This is the primary comparison and it is decisive.
4. **Is the child useful in absolute terms?** No. Language dev NLL is
   1.90×/2.22× the host (frozen ceiling 1.50×); macro 8.57/9.40 vs ceiling 6.00.
5. **Is the donor itself capable?** Yes, in its own frame: standalone SmolLM
   dev (own tokenizer) language NLL 4.0773, math 1.4689. Capability exists;
   surgical transfer through dense seams at this budget destroys its value.
6. **Does the Campaign 4–5 pattern generalize?** The qualitative pattern —
   donor dependence without donor benefit — now holds across **two donor
   families** (Qwen2 and Llama), two widths (1536, 576), and both mandatory
   and non-mandatory interfaces. The twin-beats-pretrained direction seen in
   Campaign 3 also recurs here at larger magnitude.
7. **What differs from Campaigns 1–6?** Only the donor family/checkpoint/width
   and the code paths to support it. Host, data, protocol, metric, budget, and
   gates were held constant, so the donor change is isolated.
8. **Were controls fair?** Yes: identical shape/native blocks/seams/data/
   budget/seed; twin differs only in sector initialization (native init, seed
   1234), verified frozen and shape-matched by tests.
9. **Was anything promoted?** No. Promotion required G-B ∧ G-C at seed 42 plus
   a ≥1% replicated benefit at seed 123; all failed. Held-out was never opened
   (`NOT_RUN_NO_PROMOTION`), exactly as frozen.
10. **Any integrity concerns?** None. Revisions pinned and enforced, parents
    fingerprinted unchanged, evidence bundles SHA-256-verified on import. No
    failed runs; two process incidents (probe-log retrieval, workflow commit
    race) documented in `failure_analysis.md`.
11. **What would Outcome A/B have required?** A: replicated G-B pass plus G-C
    pass plus one-shot held-out confirmation. B: replicated G-B pass with
    usefulness failure. Neither was approached.
12. **Is this a claim about all heterogeneous surgery?** No. It is a claim
    about dense-seam transplantation of late transformer sectors into an SSM
    host at ~32-step CPU budgets, now replicated across two donor families.
    Larger budgets, different placements in the Llama donor, trained-interface
    families beyond dense seams, and reverse pairings remain untested.
13. **What is the single best next hypothesis?** See the decision memo below.

## Decision memo — best next hypothesis (no Campaign 8 launched)

Across seven campaigns, every failure shares one shape: the interface is
trained briefly on token loss while the donor sector's *internal
representation geometry* is left to fend for itself — and a random sector,
which starts near-identity (relative update ≈0.47), is easier for the host to
absorb than a pretrained sector that aggressively transforms its input
(≈54–58). **Next hypothesis: the binding constraint is interface training
signal, not donor content — a sector whose input/output geometry is first
calibrated on unlabeled host-stream statistics (moments only, never hidden
targets) and then trained at the budget where Campaign 3's 96-step twin
reached 4.01 macro will show a pretrained-over-twin advantage that the
32-step dense protocol cannot express.** This is testable cheaply, keeps every
frozen guardrail (matched twins, two seeds, no TEST selection), and directly
attacks the one mechanism consistent with all seven campaigns' data. It is
recorded here as a recommendation only; no new campaign has been started.

## Verification and publication

- Code: Llama family support in `backbones.py`/`parents.py`/`controls.py`/
  `synthetic.py`/`recipe.py`/`model.py`; runner `scripts/campaign07_runner.py`;
  publisher `scripts/publish_campaign07.py`.
- Tests: `tests/test_campaign07_llama.py` (7) + `tests/test_campaign07_results.py`
  trace every published number to the SHA-verified raw records
  (`campaigns/campaign_07/screen_38542143640/`, `replication_38544932532/`).
- Records: audit, pairing selection, frozen preregistration (`.md`+`.json`),
  Amendment 01, both registries, `control_comparisons.csv`,
  `donor_ablation_results.csv`, integrity/resource/failure/test/recovery docs
  in `campaigns/campaign_07/`.

## Plain-English closing summary

We gave the Mamba "host brain" a new donor — a slice of a small but genuinely
capable Llama-family model — using the same careful, pre-registered test as
before, including a "random slice" twin as the fair comparison. The borrowed
slice definitely *did things* inside the child (removing it hurt badly), but
the child with random blocks actually performed **better**, twice in a row.
So: the transplant surgery works mechanically on a second donor family, the
donor is talented on its own, yet its talent still does not survive the
journey through today's simple connecting layers. That is a clean, honest
negative — **Outcome C** — and it points the next experiment squarely at the
connectors and their training, not at finding yet another donor.
