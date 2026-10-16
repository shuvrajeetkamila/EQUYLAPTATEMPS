# HETEROGENEOUS AI MODEL SURGERY — audited real-parent follow-up

**Research date:** 2026-10-13 (Asia/Calcutta). Run timestamps use the connected runner's UTC clock. **Outcome:** software execution/reconstruction PASS; registered language retention and learned-sector advantage FAIL. No successful inheritance claim.

## Plain-English finding

The real hybrid model exists and runs, but it is not a useful combination of its parents' skills. The previous small pilot already failed. This continuation added a stronger test: compare its actual learned Qwen sector with a randomly initialized sector of exactly the same native architecture, giving both interfaces the same training examples, optimizer, steps and seed.

On **64 fresh language windows**, the learned-sector child was worse than both original parents and worse than the matched random-sector control. This is stronger negative evidence than a changing ablation score: under this topology/calibration, the donor's learned parameters did not produce a language advantage. It does not establish that all heterogeneous surgery is impossible, that the original Qwen lacks math ability, or that a better alignment/calibration procedure could never work.

## Verified starting state and audit-first sequence

The supplied b077573/106-test identifiers were historical. Actual checkout/PR #1 was **20f456b8e935540fe572ddb6bbd56a3cbf8f94a1**, branch `arena/97efdf4a-equylaptatemps`, PR OPEN, latest Python3.10/3.11 checks green. Independent audit reproduced **137 passing tests**, lint/format/dependency checks, synthetic ZIP reload and all historical real pilot metric summaries/source identities.

The audit found two integrity gaps: an invented inheritance PASS could be substituted without rejection, and reference-only checkpoints checked weights but not altered RoPE/operator settings. Fixes added v2 execution contracts, verified-load contracts for old v1 references, ordered-sample consistency and registered verdict recomputation. Audit/fixes were committed and pushed **2c02c914d2cd206264b4da91374bba4528648501 before this new real-parent run**. Six additional tests passed (143 total). See `reports/audits/2026-10-13-initial-audit.md` and its raw before/after evidence. Neither original synthetic nor existing real interface artifacts was overwritten.

## Real parents, provenance, architecture and license status

No substitute or unrelated model was used:

| Role | Actual model/revision | Original unique parameters | Architecture/config facts |
|---|---|---:|---|
| Recipient | `state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1` | 129,135,360 | 24 blocks, width768, inner1536, selective state size16, causal convolution kernel4, dt rank48, SiLU, native RMSNorm epsilon1e-5 |
| Donor | `Qwen/Qwen2.5-Math-1.5B@4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2` | 1,543,714,304 | 28 blocks, width1536, MLP8960, GQA12 query/2 KV heads, head128, RoPE theta10000/max4096, gated SiLU MLP, RMSNorm epsilon1e-6 |

Mamba embedding/tied output vocabulary50280 belongs to the child; donor embedding/tied output vocabulary151936 is not transplanted. Native tokenizers have different BPE/vocabularies and fingerprints. Dimensional compatibility does not imply semantic compatibility. Real parent hashes match the previously pinned official safetensors: Mamba `1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea`; Qwen `f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b`. Full config/tokenizer/notice/source inventories are in this run's provenance.

These are materially different state-space versus attention-based architecture families, with disjoint declared weight roots, not two related fine-tunes. Hidden ancestry/pretraining overlap remains unknowable from these records. Qwen's pinned snapshot has Apache2 LICENSE/tag. Mamba's HF conversion omits direct license documentation/tag; official original `state-spaces/mamba-130m@c4dcd5d65dff7d0bff5a4c97e30d82e57190f1b0` declares Apache2. This lineage evidence is disclosed, not a substitute direct converted license. No frozen parent tensor is distributed.

## Exact real surgery and reconstruction

The unchanged published selected child is one executable path:

```text
Mamba tokenizer/embedding → Mamba blocks [0,11)
 → new LayerNorm/full dense768→1536
 → complete learned Qwen blocks [20,22)
 → new LayerNorm/full dense1536→768
 → Mamba blocks [13,24) → original Mamba norm/head
```

Mamba [11,13) is physically absent, not averaged with a donor. Qwen [20,22) contains **93,595,648 frozen parameters**; removed Mamba [11,13) contains **7,543,296**. The child has **217,553,920 unique parameters**, **215,187,712 frozen**, **2,366,208 new interface/norm parameters**. Native residuals, RMSNorm/MLP/attention/causal masks/config-derived FP32 RoPE and Mamba full-sequence convolution/SSM operations remain intact. No persistent cross-request state; generation is cache-free and slow. The default child is not a router/ensemble and never calls full parent models for an answer.

The actual child was reconstructed from `examples/pretrained-pilot-reference/` using new verified-load contracts. Its new-interface file remains **9,465,560 bytes**, SHA `5f2337211dd54ea0dd001244735cc3e7c3d2b74468eaea99053ad6a84411d537`. v2 reference descriptor is saved in the new run; it adds execution semantics without duplicating the interface binary in Git. A fresh subprocess independently loaded the original parents/verified reference, then generated output: **identical logits and identical generated token IDs/text**. This is reference-dependent reconstruction, not standalone loading. Earlier private standalone export/parent-forbidden reload is preserved as separate measured evidence.

New repeatable unscored generations: `The study of mathematics` → `,,,, the,,,`; `Language is a way to` → `,,,,,,,,`. These are valid token-level executions but not usable language answers. No learned frozen tensor changed; all original-parent, published-child frozen and random-control frozen invariants passed.

## Registration, data, controls, resources

Registration was committed before held-out scoring: `configs/pretrained-language-diagnostic.json`, source **7ca1a4dba38ec663065c8300601e8d44148b0758**. Connected Actions run **38301210044** completed in **14m51s**, with measured experiment wall-clock **841.945181451 seconds**. This was real CPU execution, not local GPU/Colab execution.

- The sandbox still has 4,131,278,848 RAM bytes, no GPU, Hub TLS EOF; no local parent download attempted.
- Connected CPU environment/RAM/disk/torch values are recorded in `results.json`; common float32, two threads, no quantization. Qwen's BF16 stored values are promoted exactly; native baseline execution uses the same float32 as the child/control.
- Dataset: immutable registered WikiText-2 PyTorch mirror, TEST nonoverlapping 192-character windows **16–79**, excluding all previous16 inspected windows. Source/byte hashes checked against the actual pinned dataset. BOS context and EOS targets; report bits per UTF-8 byte, not official word-level WikiText perplexity. Data terms CC-BY-SA3/GFDL, not the mirror software license; attribution/card/source commits preserved.
- Random-sector interface calibration uses the **same original 16 human GSM8K TRAIN references plus 8 WikiText TRAIN windows**, same seed42, LR0.0003, AdamW betas0.9/0.999/epsilon1e-8/weight_decay0, clip1, batch1, steps16, max512, checkpointingTrue. Dataset content/selection/budget checked against the original selected-child run. No test targets, teacher-generated answers or parent-activation targets used. GSM8K is MIT; model-generated solution files never acquired.
- Only eight full-sized interface/norm tensors train. Fresh native Qwen blocks use the pinned Transformer's standard initialization on only those two blocks (seed1042), then are frozen. Original pretrained parents remain untouched. Random sectors are controls only, explicitly labeled **random_sector_control** and cannot be exported as a pretrained child.
- All eleven variants use the same64 text strings and comparable precision/conditions. Different parent tokenizations retain native token counts; compare byte likelihood, not cross-tokenizer perplexity.

## Actual measured 64-window results

Lower bits/byte is better. Every score below is measured and recomputed from raw next-token likelihood records, not hardcoded into the evaluator:

| Variant | Bits per UTF-8 byte |
|---|---:|
| Original Mamba recipient | **1.7921447898729175** |
| Original Qwen donor | **2.34309137844324** |
| Published calibrated learned-sector child | **3.2795770048682704** |
| Same topology, uncalibrated interfaces | 6.384759399368934 |
| Independent random interfaces | 5.8836934613781295 |
| **Matched-budget frozen random donor sector** | **3.043678304653564** |
| Donor bypass | 11.236217523096013 |
| Recipient prefix bypass | 4.034932890418672 |
| Recipient suffix bypass | 10.179113167161882 |
| Zero input interface | 4.997706354256383 |
| Zero output interface | 3.862536469425896 |

The child/Mamba language BPB ratio is **1.8299732384350642** (derived). Registered retention requires at most1.10×Mamba and at most3.0BPB, min64: **FAIL**. Both ceilings are missed. The matched random sector is better than the learned donor sector: random-minus-learned BPB **-0.23589870021470663**, paired bootstrap95 **[-0.27793177999948926, -0.19742045716239084]** (derived from raw paired records, seed42/2000 resamples). Registered positive learned-sector advantage: **FAIL**. Single-seed/correlated windows and one random-sector seed prevent universal claims.

Ablation control-minus-full differences (paired bootstrap95): donor bypass **7.95664051822774 [6.997497529808451,8.893176919263222]**; prefix bypass **0.7553558855504015 [0.6706636030979336,0.8511261865338777]**; suffix bypass **6.899536162293611 [6.563097254055149,7.281454488047574]**. They prove dependence under this input distribution, **not useful inherited capability**; the trained random sector outperforming the learned one is precisely why ablation alone was insufficient.

Matched-control TRAIN token NLL **29.471969736602915 → 6.995113719593395**, calibration time **453.55893488899994 seconds**. Those are measured fitting diagnostics, not benchmark accuracy. Actual step journals, trainable-tensor names/shapes/counts and optimizer settings are preserved. No assertion that random weights inherited anything is made.

Mathematics was **NOT_RUN in this isolated follow-up**; no new math result inferred from a language score. The prior actual 4-case/32-token pilot remains unchanged: child0/4, flexible parent1/4 but strict0 for all; incomplete/intermediate numeric matches limit interpretation. Adequately sized long-output math/source-capability evidence remains NOT_RUN. Dual-capability inheritance/complementarity/improvement are **NOT_RUN / unproven**. Existing reverse finite structural check is preserved; no reverse training/scaling while primary retention fails.

## Failure diagnosis and sector-discovery improvements

Historical source observations are never overwritten. A derived inventory under `reports/pretrained/2026-10-13-derived-diagnosis/sector-selection.json` labels [20,22) SELECTED and [13,15) REJECTED according to actual development scores8.873143710709705 versus9.439523602347089. Future searches annotate those decisions directly; fabricated/nonfinite/wrong-winner selections fail.

Historical native-parent development ablation NLL increases (derived): Qwen[13,15) **0.10833836352731296**, Qwen[20,22) **0.18971215395789764**, Mamba[11,13) **0.18744933205928493**. These are model-specific own-vocabulary NLL sensitivities, not math-organ labels or directly comparable cross-tokenizer capability scales.

Actual saved new-interface matrices were independently analyzed locally without parents: both numerical rank768; condition numbers **5.595561981201172** (input1536×768) and **6.099851131439209** (output768×1536); same-seed relative Frobenius weight changes **0.047359537333250046** and **0.047744058072566986**. Norm scale means stayed near1. These are measured checkpoint statistics, not performance or proof of causal failure. They do not suggest numerical rank collapse; little interface adaptation may be insufficient for semantic alignment.

A scale-mismatch hypothesis is explicitly **estimated**, not observed child activation: unit-variance LayerNorm + Xavier predicts initial bridge RMSsqrt(2×768/(768+1536))≈**0.816496580927726**; measured native Qwen block20 input RMS from two historical development samples was **13.094068301171557**, a ratio≈**16.036892997511043**. Different tokenizations/contexts/covariance/calibration make this only a hypothesis. Further development-only residual-scale/layout diagnostics are justified; no activation-matching teacher loss or test-target training is introduced to force success.

## Integrity, reproducibility and publication

- New audit/verifier rejects unsupported edited inheritance PASS claims, mismatched ordered comparison cases, registered-count changes and modified native RoPE/norm execution semantics. Tensor hashes alone were insufficient.
- Diagnostic verifier recomputes every likelihood/paired summary/registered verdict and source text/byte identity. All checks passed independently after receiving the public source-bound bot evidence bundle. This is consistency/tamper evidence, not proof against malicious complete-bundle forgery.
- **158 offline tests passed**, ruff lint/format and dependency checks passed. Raw JUnit/current validation records are under `reports/pretrained/2026-10-13-derived-diagnosis/`.
- Standard regression remains offline/lightweight. The actual weight run is a separately opted-in job, not mandatory gigabyte CI downloads. No existing test weakened; old106/137 tests remain historical subsets.
- Both original synthetic and real interface files/checksums are unchanged. No frozen parent weights, private credentials or large raw datasets added to Git. Parent caches were reused within the run, including the fresh subprocess. No duplicate full parent/child checkpoint exported in this follow-up.
- All work remains on the original `arena/97efdf4a-equylaptatemps`, existing PR#1. Appropriate code/configs/raw lightweight records/audit/report are committed/pushed; exact final commit/check state is reported in PR/final response.

### Exact reproduction commands

On a connected CPU machine with >=12GiB RAM/free disk and repository installed (torch2.5.1, transformers4.48.3):

```bash
python scripts/run_language_diagnostic.py --config configs/pretrained-language-diagnostic.json --out artifacts/language-diagnostic-002
python scripts/verify_study.py artifacts/language-diagnostic-002 --datasets data/local/pretrained-benchmarks
```

Use a new output directory; preserve failures. For the original registered larger GPU/Colab study, see `docs/pretrained-study.md` and `notebooks/pretrained-study-colab.ipynb`. That GPU study has NOT_RUN status, not an implied execution. New reference contracts work with the unchanged published v1 files through fresh `load_parents` verification; v2 saves additionally include the canonical config/native operation contract.

## Exact next research milestone

Before more expensive or reverse surgery, diagnose and preregister **development-only representation/residual-scale alignment changes** while keeping intact real sectors and frozen pretrained tensors. Then require language retention versus the real parent AND advantage versus matched frozen random sectors across multiple seeds. Only after that, perform adequately sized long-output mathematics/source baselines and both-task controlled inheritance evaluation. The current evidence favors failure/generic feature fitting over useful pretrained transfer; do not rename structural transplantation as inherited capability.
