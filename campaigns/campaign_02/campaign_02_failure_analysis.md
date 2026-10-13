# Campaign 2 failure analysis — HETEROGENEOUS AI MODEL SURGERY

Date:2026-10-13. Numerical findings below are **measured** in frozen replay38354043771, sourcea977e18, unless explicitly labeled as hypotheses. No hidden activations are training targets.

## Reproduction gate

The actual original64-window WikiText language scores reproduced exactly at the recorded floating precision: Mamba1.7921447898729175, Qwen2.34309137844324, child3.2795770048682704, matched random donor3.043678304653564. All11 original controls were rerun, identical aggregate scores. Original16-question completed-answer math predictions/counts were recovered:Qwen12/16, child0/16, Mamba0/16; all other scored controls0/16. Same prompts/fewshots/testIDs/256-token budgets/final-marker parser/native tokenizers. Raw generatedIDs/text/termination/timing are preserved; the campaign does not relabel incomplete answers correct.

Both real parents, published child and matched random control frozen values were invariant. Parent-reference fresh subprocess logits AND token/text samples reproduced identically. This reference reload loads exact pinned parents; it is not falsely called a parent-free standalone rerun. Prior child-only standalone proof is historical and kept separate. The reverse cache oracle in the replay is clearly a historical reverified record, not a new reverse experiment.

## Input and output distributions — actual unlabeled TRAIN text

Eight192-character WikiText TRAIN windows were passed through both original parents using their own native tokenizers and through the actual child using the host tokenizer. Buffers retained only float64 aggregate sums/second moments/covariance, not per-example or per-token vectors. Full per-channel means/variances, covariance trace/spectrum/effective rank/correlation, magnitude thresholds/position RMS/finite rates are in `baselines/replay_38354043771/failure_analysis/representation_statistics.json`.

| Stream | Global RMS | Mean | Fraction abs>10 |
|---|---:|---:|---:|
| Native Mamba input to removed sector |1.207607099704619|0.031491529162096105|0.001376589464123524|
| Native Mamba output of removed sector |1.433743829202458|0.041640032241754464|0.0014475476839237056|
| Native Qwen input to selected sector |29.59514231751426|0.026500282222221467|0.037494419642857144|
| Native Qwen output of selected sector |29.82151543724568|0.029567303525351835|0.059830729166666666|
| Child donor input after new interface |0.816505261543267|-0.009012682328143971|0|
| Child donor output |7.865372999824421|0.16726840434526727|0.19862447490917348|
| Child host suffix input after new output interface |1.7211309751766404|0.057885230240546516|0|
| Child host suffix output |45.560766184240244|-0.3642865591770256|0.6776616428247049|

The child donor input is roughly36× smaller in RMS than the donor's native input on this text. Its native frozen RMSNorm normalizes the branch, but the residual skip retains raw scale; this changes the branch-to-residual relationship. Output LayerNorm restores unit scale before projection but cannot restore lost semantic coding. These are actual distribution discrepancies, not proof that a scalar gain alone repairs meanings. Native and host BPE sequences differ; diagnostic aggregate comparisons are unpaired, not semantic correspondence labels. The large suffix output remains finite; absent a matched original suffix-output statistic, do not call it proven numerical instability. All observed nonfinite counts are zero. “Saturation” is not a universally meaningful threshold for SiLU/linear operations; abs>10 and nearzero thresholds are explicitly documented instead of invented.

## Architectural and optimization findings

Whole native blocks, residuals, norms, source layer indices, causal attention masks/config-derived RoPE and Mamba full-sequence convolution/SSM states remain. Parent/child cached paths are numerical-oracle gated; zero-input cached control remains explicitly blocked after its historical failed gate. Default training/full-prefix implementation is preserved, CPUfloat32/batch1 cache scope not generalized to GPU/padding. No added task router, host-residual donor bypass, parent-weight averaging or low-rank mechanism.

Actual one-example child next-token backward probe (no optimizer update) reached every new tensor. Input projection gradientL2 **11.323934555053711**, output projection **9.630228042602539**; norm/bias gradients also nonzero and finite. Parent tensors remained frozen and gradient-free. Thus total failure is not simply absence of gradient. New resumable optimizer telemetry records per-tensor mean/std/max/norm/zero fraction, pre-clip global norm, batch losses/order, parameter updates and exact safe checkpoint identity. A global clip1 can constrain updates; this is documented, not silently removed. Existing32/16-step data/parameter ratios are small, so under-calibration remains plausible.

Earlier actual new-interface SVD showed full numerical rank768, condition numbers~5.6/6.1 and~4.7% same-seed Frobenius change. That is compatible with modest adaptation, not rank collapse. Spectral/measured scalar diagnostics cannot prove a useful representation mapping.

## Justified limited comparisons

Registered before new development observations: same topology/seed/data/budget full-sized legacy randomLayerNorm projection versus independent residual-scale conditioning, source-covariance orthogonal conditioning and fixed orthogonal no-normalizer linear projection. Conditioning uses unpaired aggregate unlabeled TRAIN moments; no donor hidden vectors/distributions/answers are optimization targets. Actual training loss is the child's own next-token/human ground-truth loss. Native parent values stay frozen. Fixed projection is a dimensional control, not an identity of unequal768/1536 hidden spaces or different token semantics.

One-block, early-boundary and reverse-direction paths receive cheap untrained structural/development checks first. True identity is rejected at unequal dimensions. Two-layer interfaces/larger sectors are deferred because extra parameters/compute would confound attribution before these simpler hypotheses are tested. Selection uses development only; no TEST tuning or retrospective threshold changes. Final dual-capability OutcomeA also requires matched random-sector/ablation controls, meaningful math/language retention and independent seeds; execution alone cannot promote it.

## Remaining hypotheses

The measured scale/covariance discrepancies justify conditioning research, not its success. Token-semantic coding, loss of the donor's embedding/prior computation/output path, insufficient interface expressiveness/data/steps and generic nonlinear feature fitting remain alternative explanations. Random sector already outperforming the learned donor requires strong matched controls. If no candidate passes the registered development gate, do not blindly scale or claim inheritance; retain OutcomeC/D and preserve all failures/recovery checkpoints.
