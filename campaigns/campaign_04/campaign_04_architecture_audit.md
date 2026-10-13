# Campaign 4 — Architecture Audit

Text → host (Mamba) tokenizer → frozen Mamba embedding (768) → frozen Mamba
blocks [0,11) → **input interface** → frozen intact Qwen blocks [20,22)
(1536; native GQA/RoPE/causal mask, positions derived at
`backbones.py:110-113`) → **output interface** → frozen Mamba blocks [13,24)
→ frozen final norm → frozen tied host head. One path; no skip around the
donor (`model.py:166-189`). Campaign 4 changes only the interfaces and, for
delta candidates, one subtraction at the donor boundary: the suffix receives
(donor output − donor input), so an identity donor contributes exactly zero.
Internal Qwen residuals are untouched (they are part of the pretrained
computation, not an external bypass).
