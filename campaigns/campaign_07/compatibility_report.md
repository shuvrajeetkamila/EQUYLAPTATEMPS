# Campaign 7 Compatibility Report

**Verdict: compatible at the frozen design point.** The Mamba→Llama pairing
executed end-to-end with no architecture-specific hacks:

- `LlamaConfig/LlamaDecoderLayer/LlamaRMSNorm/LlamaRotaryEmbedding` support was
  added to `backbones.py` (native blocks, eager attention, RoPE inside
  `StackSegment`, sliding-window configs explicitly rejected). Synthetic
  Mamba→Llama children pass forward/backward/ablation/twin tests offline
  (`tests/test_campaign07_llama.py`, `scripts/campaign07_smoke_local.py`).
- Real pairing (screen job `38542143640`): child = 143,721,408 unique params,
  888,768 trainable seam params, 6 transplanted blocks, context limit 2048
  (SmolLM `max_position_embeddings`), widths 768↔576 bridged by the frozen dense
  seams. Parent loading enforced the pinned revisions and safetensors-only
  acquisition; SmolLM's tied embeddings and GQA (3 KV heads) required no special
  casing beyond the family branch.
- Known limitation (recorded, not hit): cached autoregressive decoding
  (`decoding.py`) supports only Mamba/Qwen2 caches; Campaign 7 evaluation is
  teacher-forced NLL, so this does not affect any reported number. Donor-native
  standalone dev (own tokenizer, not comparable to host-tokenizer metrics):
  language NLL 4.0773, math NLL 1.4689 — the donor is capable in its own frame;
  the failure is in surgical transfer, not donor quality.
