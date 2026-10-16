# HETEROGENEOUS AI MODEL SURGERY — Campaign 2

## Plain-English summary

We did **not** restart from the synthetic demonstration. We audited the existing real pretrained Mamba–Qwen child, froze and reran its disappointing results, measured why its internal signals may be incompatible, and tested a small set of bridge-initialization ideas.

The child is real and executable: complete learned Qwen blocks are physically inside a Mamba-based model, the original pretrained tensors remain frozen, and a fresh process can reconstruct and repeat its output. That is successful **structural surgery and execution**.

It is not yet a useful hybrid brain. On fresh language text, the child is worse than both parents and worse than a matched random-sector control. On a completed-answer mathematics diagnostic, Qwen answers 12 of 16 questions while the child answers 0 of 16. A source-covariance-based bridge initialization improves development loss, but not enough to pass the criteria fixed before testing. We therefore classify the campaign as **Outcome C — structural surgery without useful inheritance**. We do not call this successful capability inheritance.

## 1. Initial repository and experiment audit

The repository is `shuvrajeetkamila/EQUYLAPTATEMPS`, branch `arena/97efdf4a-equylaptatemps`, PR #1. The verified starting HEAD was `84205a51a44206f08a2f1c8219f0684fc1779626`, not the older reported baseline commit. PR #1 was open and its Python 3.10/3.11 checks passed.

The local checkout initially pointed at an older commit while its saved workspace contained older campaign files. All 182 workspace files were compared with the live branch: 178 matched and four were older versions. They were preserved in a Git stash rather than overwritten, then the same branch was fast-forwarded to the verified commit. No repository restart, branch switch, force-push, or deletion occurred.

Local resources were 4,131,278,848 bytes of RAM, no GPU, and Hugging Face access blocked by a TLS error. Genuine parent execution therefore used separately opted-in connected GitHub CPU runners, as before. The audit was committed before new real-parent experiments.

## 2. Parents and child verified

The actual parents are:

| Role | Model | Immutable revision | Parameters |
|---|---|---:|---:|
| Host/recipient | `state-spaces/mamba-130m-hf` | `1e76775f628fbf1350fbe4dbb3d971ba64af25a1` | 129,135,360 |
| Donor | `Qwen/Qwen2.5-Math-1.5B` | `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2` | 1,543,714,304 |

Mamba uses selective state-space/convolution blocks; Qwen uses causal attention with rotary position embeddings and gated MLP blocks. They are materially different architecture families, not two fine-tunes of one family. Their configurations, tokenizers, and safetensors hashes were verified in the parent-loading records.

Qwen's snapshot has an Apache-2.0 license. The converted Mamba repository does not include a direct license document/tag; the official original Mamba checkpoint declares Apache-2.0. That is lineage evidence, not a direct license inside the converted snapshot. No frozen parent weights were added to Git.

The preserved child is:

```text
Mamba embedding and blocks [0,11)
→ new LayerNorm + full dense 768→1536 interface
→ complete learned Qwen blocks [20,22)
→ new LayerNorm + full dense 1536→768 interface
→ Mamba blocks [13,24), final norm and Mamba output head
```

It contains 217,553,920 parameters: 215,187,712 frozen and 2,366,208 newly introduced interface/norm parameters. Mamba blocks [11,13) are absent. The donor embedding and output head are absent. There is no LoRA, distillation, weight averaging, ensemble, or task router.

## 3. Reproduction of the original negative findings

Campaign connected run `38354043771` reran the frozen published child, both original parents, and the principal controls. The earlier language results reproduced exactly:

| Variant | WikiText bits per byte, lower is better |
|---|---:|
| Original Mamba | **1.7921447898729175** |
| Original Qwen | **2.34309137844324** |
| Published transplanted child | **3.2795770048682704** |
| Matched random-donor control | **3.043678304653564** |

The evaluation used the same 64 fresh WikiText TEST windows, the same UTF-8 byte denominator, each model’s native tokenizer, and byte-normalized likelihood. This is not official word-level WikiText perplexity.

The completed-answer GSM8K diagnostic also reproduced its earlier counts:

| Variant | Correct |
|---|---:|
| Qwen | **12/16** |
| Mamba | 0/16 |
| Published transplanted child | 0/16 |
| Matched random-donor control | 0/16 |

This diagnostic used 16 fresh questions, two human-written training few-shot examples, 256 output tokens, and credit only for a completed final answer marker. It is preliminary evidence, not a full benchmark ranking, but it is long enough to show that the donor has a usable capability that the current child does not retain.

Both parents remained unchanged. A fresh subprocess reconstructed the published reference and reproduced identical logits and generated tokens. That is reference-dependent reconstruction, not a standalone checkpoint.

## 4. Root-cause investigation

We measured aggregate activation statistics on eight unlabeled WikiText TRAIN windows. We did not save per-token hidden vectors and did not use hidden activations as training targets.

The most striking measured discrepancy is scale:

| Stream | RMS |
|---|---:|
| Native Qwen input to the selected sector | 29.595 |
| Hybrid input supplied to that sector | 0.817 |
| Native Mamba output of the removed sector | 1.434 |
| Hybrid output entering the Mamba suffix | 1.721 |

The transplanted Qwen blocks therefore receive a much smaller residual stream than in their original model. Qwen’s internal normalization partly compensates inside each block, but its residual pathway still receives a differently scaled branch contribution. A distribution-aware initialization is justified, but matching a scale does not prove that the meanings of the dimensions correspond.

Other findings:

- No nonfinite activations were observed in the diagnostic streams.
- Full-channel means, variances, covariance summaries, position-dependent RMS, near-zero rates, and large-magnitude rates are recorded.
- On a real-text backward probe, gradients reached every new interface tensor and were finite. The failure is not simply that no gradient reaches the bridge.
- The existing interface matrices have full numerical rank and no obvious rank collapse.
- Token semantics remain a major unresolved difference. Mamba and Qwen tokenize the same text differently, and the child receives Mamba token IDs while its Qwen blocks were pretrained inside a Qwen-token representation pathway.
- The donor sector was removed from its original embedding, earlier blocks, later blocks, final norm, and output head. A two-block sector is not an isolated “math organ.”
- Short budgets and only 2.37 million new interface parameters may be insufficient, but increasing them without controls would not identify the cause.

## 5. Bridge designs investigated

All first-stage designs retained the same complete donor blocks and trained only new interface parameters. The objective was the child’s own next-token loss on real training text and human GSM8K TRAIN answers. No parent-generated answers, parent logits, or hidden activations were used as targets.

| Design | What changed |
|---|---|
| Legacy bridge | Existing LayerNorm plus full dense projection, random initialization |
| Residual-scale bridge | Same architecture; input/output matrices initialized with measured RMS gains |
| Source-covariance bridge | Orthogonal initialization conditioned using source-side aggregate covariance, plus the same RMS gains |
| Fixed orthogonal linear bridge | Fixed orthogonal coordinate map without the learned input/output normalizers |

A true identity bridge was not run because the host width is 768 and the donor width is 1536. Calling a rectangular projection an “identity” would be misleading.

The covariance method used aggregate source statistics, not paired examples and not donor activations as labels. It is initialization/conditioning, not semantic-alignment proof.

## 6. Development calibration results

The four candidates used the same registered training pool, seed, 32 steps, optimizer settings, and maximum sequence length. Development data—not final test data—was used for screening.

| Candidate | Development language NLL | Development math NLL | Macro average |
|---|---:|---:|---:|
| Legacy random | 7.7657 | 7.4107 | 7.5882 |
| Residual scale | 7.8483 | 6.7534 | 7.3009 |
| Source covariance | **6.7650** | **4.5917** | **5.6783** |
| Fixed orthogonal linear | 7.4288 | 5.9296 | 6.6792 |

The covariance-conditioned bridge was the strongest development candidate. Compared with the legacy bridge, its language NLL improved by about 12.9%, and its mathematics-development NLL was much lower. This is a genuine development signal.

It nevertheless failed the promotion rule fixed beforehand:

- Development language NLL remained **1.539×** the original Mamba host, above the permitted 1.50×.
- No candidate met all registered development retention and improvement conditions.
- Therefore no candidate was promoted to a new held-out confirmation run.

This distinction matters: “better than the failed child” is not the same as “good enough to claim retained capability.”

## 7. Alternative topologies screened

Cheap, untrained structural/development screens were run before spending more training budget:

| Topology | Result |
|---|---|
| One Qwen block instead of two | Executed, but uncalibrated development loss was extremely poor |
| Earlier host/donor boundary | Executed, but uncalibrated development loss remained poor |
| Reverse direction: Mamba sector inside Qwen | Executed as one integrated model, but uncalibrated development loss remained poor |

None gave a reason to displace the primary topology without calibration. A larger sector, a two-layer bridge, a second seed, and a separate general-reasoning benchmark were deliberately not run after the promotion gate failed. Running them anyway would spend substantial compute while weakening attribution.

## 8. Scientific evaluation status

- Language reproduction: **completed**, negative scientific result.
- Mathematics diagnostic reproduction: **completed**, negative child result.
- New held-out confirmation for alignment candidates: **NOT_RUN**, because no candidate passed development promotion.
- General reasoning: **NOT_RUN**.
- Multiple independent seeds for a promoted candidate: **NOT_RUN**, because there was no promoted candidate.
- GPU/Colab execution: **NOT_RUN**.
- Reverse-direction capability training: **NOT_RUN**; only structural screening was performed.

The campaign intentionally did not use the held-out sets repeatedly to tune the bridge.

## 9. Does the child really use the donor?

The donor affects computation: bypassing it changes likelihood substantially. But that does not establish useful use. A harmful or disruptive block also changes the output.

The stronger controls point against useful transfer in the current child:

- A frozen random sector with the same interface-training budget performed better on the 64-window language evaluation.
- The learned child scored 0/16 on the mathematics diagnostic while the donor scored 12/16.
- The child was worse than both original parents on language.

Accordingly, we have evidence of causal influence, but not evidence of useful functional inheritance.

## 10. Failed experiments and blockers

- The original four-question, 32-token math pilot was too weak and could credit unfinished intermediate numbers. It was replaced by a completed-answer diagnostic rather than reinterpreted.
- A zero-input cached ablation failed its numerical cache-equivalence gate in an earlier run. That control remains explicitly blocked rather than silently accepted.
- The covariance candidate improved development loss but failed the absolute language-retention gate.
- Candidate interface binaries and resumable optimizer states were stored in the identified GitHub Actions artifact. The textual evidence bundle was imported and verified, but the large artifact archive could not be downloaded through this sandbox’s artifact-storage connection. Their remote hashes are recorded; local binary reverification is therefore incomplete.
- The current environment has no GPU and cannot download the parent weights directly. Real-parent work used connected CPU runners.

Checksums detect inconsistent records; they cannot prove that a maliciously forged complete evidence bundle is genuine.

## 11. Best verified child configuration

The best **development** configuration in this campaign is the unchanged primary topology—Mamba [0,11), Qwen [20,22), Mamba [13,24)—with source-covariance/RMS-conditioned full-sized interfaces.

It is not the best verified useful child, because it was not promoted to held-out confirmation and did not meet the development retention threshold. The previously published child remains the best fully reproduced configuration, and its measured performance is negative.

No new candidate is presented as a successful pretrained child.

## 12. Checkpoint integrity and reconstruction

- The original published child interfaces are unchanged: SHA-256 `5f2337211dd54ea0dd001244735cc3e7c3d2b74468eaea99053ad6a84411d537`.
- The earlier structural child interfaces are unchanged: SHA-256 `bcc43ee4fda51d35518fd79e0f27e23b4c503095dcde8320f4cdb06293f2a294`.
- The synthetic baseline archive is unchanged: SHA-256 `a3af87ff2cc0d35cbbf4e0dcf2b67460e2af9aeafc2762439675470b5a4d5be6`.
- Frozen original-parent tensors remained unchanged in the reproduced baseline and development matrix.
- Fresh-process reference reconstruction reproduced logits and generation for the published child.
- Candidate reference descriptors and training-checkpoint hashes are recorded. Candidate binary artifacts are in the named Actions artifact; see the limitation above.

## 13. Runtime and resources

- Frozen baseline replay: 48 minutes 31 seconds of job time; 2,850.56 seconds recorded experiment time.
- Sequential development matrix: 30 minutes 6 seconds of job time; 1,740.90 seconds recorded session time.
- Candidate training times were approximately 254 seconds (legacy), 239 seconds (residual scale), 392 seconds (source covariance), and 550 seconds (fixed orthogonal linear).
- All real-parent inference and training in this campaign used CPU float32 with two threads. There was no quantization.

## 14. Tests and integrity

The final local regression contains 211 automated tests. These include archive integrity, provenance, whole-block execution, checkpoint reconstruction, cache behavior, benchmark schemas, tampering detection, matched controls, resumable training, aggregate statistics, execution contracts, and campaign-registry validation.

The tests are software and evidence-consistency checks. They are not themselves evidence of model intelligence or capability transfer.

## 15. GitHub publication

All work remains on branch `arena/97efdf4a-equylaptatemps` and PR #1. Historical reports and checkpoints were preserved. No parent weights, credentials, raw large datasets, or newly generated candidate parent tensors were committed.

The central machine-readable campaign result is `campaigns/campaign_02/campaign_02_results.json`. Detailed records are under:

- `campaigns/campaign_02/campaign_02_initial_audit.md`
- `campaigns/campaign_02/campaign_02_failure_analysis.md`
- `campaigns/campaign_02/baselines/`
- `campaigns/campaign_02/matrix_38358962201/`
- `campaigns/campaign_02/experiment_registry.json`

## 16. Remaining limitations

1. Representation alignment remains unresolved; covariance/scale conditioning does not align token meanings by itself.
2. Only one principal calibration/random-sector seed was available for the strongest comparisons.
3. WikiText windows are correlated, so treating all 64 as fully independent would overstate certainty.
4. The mathematics diagnostic has only 16 questions.
5. The child uses slow cache-free training/full-prefix computation in several paths; the validated incremental decoder is CPU float32 and batch size 1.
6. No GPU execution, broader reasoning suite, or multi-seed confirmation was completed.
7. Parent pretraining contamination and undisclosed shared ancestry cannot be excluded.
8. Candidate binary recovery from GitHub artifact storage was not independently completed in this sandbox.

## 17. Final decision

### Outcome C — Structural surgery without useful inheritance

The transplanted pretrained blocks are present, execute in one integrated model, remain frozen, and can be reconstructed. However, the child does not retain the donor’s measured mathematics capability or the host’s language capability, and it does not beat the matched random-sector control. The new covariance-conditioned bridge is a promising development direction, but it fails the predeclared promotion threshold.

This is neither Outcome A nor Outcome B. It is also not Outcome D: the baseline and primary comparisons are sufficiently reproduced to support a clear negative conclusion about the current method.

## 18. Exact recommended next milestone

Conduct a **development-only semantic-alignment study** that does not train on test data or imitate parent hidden activations:

1. Compare several sector boundaries and bridge capacities while keeping all pretrained tensors frozen.
2. Add bridge designs that can transform sequence/token-level structure more expressively than a position-independent dense map, while ensuring the donor sector still performs essential computation.
3. Use larger, documented development calibration sets and multiple seeds.
4. Require, before any final test, both:
   - language development performance within the registered host-retention bound; and
   - a reproducible advantage over matched random-sector controls.
5. Only then run a larger held-out GSM8K evaluation and the 64-window language confirmation across at least two independent seeds.

Until those gates pass, do not scale to larger sectors, claim complementary capability, or describe the child as having inherited its parents’ skills.
