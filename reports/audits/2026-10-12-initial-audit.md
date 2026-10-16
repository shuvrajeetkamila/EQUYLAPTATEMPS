# HETEROGENEOUS AI MODEL SURGERY — initial audit, 2026-10-12

## Identity and independently checked starting state

- Repository: `shuvrajeetkamila/EQUYLAPTATEMPS`; fetch/push remote: `https://github.com/shuvrajeetkamila/EQUYLAPTATEMPS.git`.
- Branch: `arena/97efdf4a-equylaptatemps`; initial HEAD: `b077573adab0149834a17d7110dd477ea8afa8f8`. Initial working tree was clean.
- PR #1 was independently queried: **OPEN**, base `main`, same head branch/SHA, not merged. Both Python 3.10/3.11 push and PR checks were green at that SHA.
- Public project name is **HETEROGENEOUS AI MODEL SURGERY**. Existing repository, package, and CLI identifiers remain for backwards compatibility, not as a different project or roadmap.

## A. Verified existing functionality

1. The original regression suite reproduced: **106 passed, 0 failures, 0 errors, 0 skips**. Ruff lint/format and `pip check` passed. Raw JUnit output is preserved alongside this audit.
2. Native Mamba and Qwen2 whole-block execution, both transplant directions, different dimensions/depths, causal/right-padding masks, dense seam gradients, checkpointed gradients, parent freezing, float32/BF16 portable reload, and existing CLI workflows are covered by those tests.
3. Independently extracted and loaded the exact published ZIP, without parent snapshots. SHA-256: `a3af87ff2cc0d35cbbf4e0dcf2b67460e2af9aeafc2762439675470b5a4d5be6`; size **88,457 bytes**. Inference produced finite logits of shape `[1, 13, 259]`.
4. The distributed child has 20,376 unique parameters, 19,488 frozen parameters, and 888 trainable dense-interface parameters. Two intact Qwen2 blocks replace two Mamba blocks. No production bypass or weight interpolation is present.
5. Local-snapshot loading, supervised seam calibration, donor-window validation search, simple exact-match evaluation, native-parent comparisons, and safetensors/config/tokenizer integrity code exist and execute with tiny offline fixtures. These are implemented software features, not evidence of pretrained performance.
6. **Only randomly initialized/synthetic parents have been tested in the existing published work.** No real pretrained weight acquisition, math benchmark, language benchmark, or capability inheritance was established. Preserve the old demo/archive/reports unchanged as an engineering baseline.

## B. Independently reproduced weakness and necessary fix

The old `load_parents` accepted locally saved random tiny models/byte-tokenizer fixtures under pretrained model IDs if a caller supplied plausible-looking 40-character revision declarations. The loader reported `synthetic: false` while both snapshot records said `caller_declared_only`. This was reproduced using adversarial fixtures, not actual pretrained revisions.

**Fixed before the real-parent milestone:** pretrained paths now require a locally verified pinned snapshot inventory; synthetic byte-tokenizer fixtures cannot be labeled pretrained parents. Two added regression tests reproduce/reject those cases. After-fix suite: **108 passed**. Snapshot records are local integrity evidence, not signed proof of historical ancestry; later acquisition must verify real remote metadata/hashes.

No pre-existing regression test failed. Numerical execution of the synthetic child passed. Its pretrained-capability interpretation was never justified, and this provenance gap must not be used to manufacture one.

## C. Planned or missing at the initial revision

- Real immutable parent revision/config/license verification and actual pretrained-weight transplantation.
- Parent activation statistics, block dependency/tensor inventory, parent-sector ablation measurements, empirical functional-sector discovery.
- Established math answer normalization, real-text language modeling, dataset provenance/split hashes, and comprehensive no-training/random/recipient controls.
- Registered inheritance criteria, complete per-example benchmark records and uncertainty, honest stage/metric status tracking, resumable real-experiment orchestration.
- Fresh-process real-child load/generation verification and GPU/Colab execution workflow.
- Any measured benchmark superiority, retained pretrained capability, or useful complementary contribution.

Existing manual-sector search is not functional localization. Existing strict string matching is not a sufficient GSM8K scorer. Existing calibration only accepts prompt/answer JSONL, not a documented real-text corpus workflow.

## Resources and precise local blocker

Measured RAM: **4,131,278,848 bytes**; free disk at audit: **15,003,209,728 bytes**; GPU count **0**, CUDA unavailable; no cached/local pretrained parents found. Python 3.11.2, PyTorch 2.5.1, Transformers 4.48.3. Exact resource/package evidence is in `resources.json`.

Hugging Face metadata request to `https://huggingface.co/api/models/state-spaces/mamba-130m-hf` failed with `URLError: TLS/SSL connection has been closed (EOF)`. The sandbox supports GitHub/PyPI but not the required model host. A real-parent experiment therefore cannot be claimed locally. Model identifiers are starting candidates; immutable Hub revisions and license/config values remain unverified until fetched from a connected environment.

## D. New work introduced at this audit checkpoint

- Dated, traceable audit evidence and preserved before/after test output.
- Two provenance guards and regression tests described above.
- Correct public project title while retaining existing technical identifiers.

Subsequent milestone work must add a bounded connected-runner probe and an independently runnable GPU/Colab experiment, clearly labeling their actual measurements versus unavailable stages. It must not replace the real-parent objective with another synthetic success demonstration.
