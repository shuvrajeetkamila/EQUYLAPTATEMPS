# Campaign 7 Pairing Selection (frozen 2026-10-14)

Host is fixed as the audited Mamba-130M (`state-spaces/mamba-130m-hf@1e76775…`,
SSM, width 768, 24 blocks) so Campaign 7 isolates the **donor** change against
six campaigns of host-side evidence. Donor candidates were probed at pinned
revisions (probe runs `38540675630`, `38540795693`; results posted to PR #1).

| Candidate | Pinned revision | Family / architecture | Width | Layers | Heads (Q/KV) | FFN | Positional | Tokenizer | License | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **HuggingFaceTB/SmolLM-135M** | `1d461723eec654e65efdc40cf49301c89c0c92f4` | Llama: GQA + RoPE + SwiGLU + RMSNorm | 576 | 30 | 9 / 3 (dim 64) | 1536 silu | RoPE θ=10000, 2048 ctx | GPT2-style BPE, vocab 49152, tied emb. | Apache-2.0 | **PRIMARY** |
| HuggingFaceTB/SmolLM-360M | `59f7ef243ee09a72cbc14cb054393a3e3b771d41` | Llama (same family) | 960 | 32 | 15 / 5 | 2560 | RoPE θ=10000 | same | Apache-2.0 | Rejected: same family as primary; ~2.7× donor compute for no new architectural information; retained as documented alternate only |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | Llama | 2048 | 22 | 32 / 4 | 5632 | RoPE θ=10000 | Llama BPE, vocab 32000 | Apache-2.0 | Rejected: 1.1B donor repeats the Campaign 1–6 CPU-feasibility problem (download + fp32 activation cost) with no width match benefit |

## Why SmolLM-135M

1. **Genuinely new pairing:** Llama-family donor (GQA, RoPE, SwiGLU, RMSNorm,
   tied embeddings) into the Mamba SSM host — a different donor family,
   checkpoint, width (576 vs Qwen's 1536), depth, and tokenizer from all prior
   campaigns. It tests the Campaign-5 memo's explicit next hypothesis.
2. **Feasible honestly:** 135M params, single safetensors, CPU-runnable at the
   frozen protocol scale; no silent budget shrinkage needed.
3. **Strong small donor:** pretrained on a large web/code/math mixture, so a
   late contiguous sector [24,30) is a meaningful capability probe.
4. **Width change is part of the test:** seams must bridge 768↔576, exercising
   the project's width-mismatch machinery rather than assuming equal widths.

Rejected alternative considered: reversing roles (transformer host, Mamba donor)
— deferred, not selected: it would require a new host pipeline (embedding/head
ownership, cached decoding) and confound "new pairing" with "new harness" in a
single campaign. Recorded here so the choice is auditable.
