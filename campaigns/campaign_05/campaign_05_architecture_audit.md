# Campaign 5 — Architecture Audit

Verified Campaign 4 child (one path, no external bypass): host tokenizer → frozen Mamba
embedding (768) → Mamba blocks [0,11) → bottleneck-128 input interface (norm → down →
causal-attention mixer → up) → intact frozen Qwen sector → **donor-delta** (sector output
minus sector input) → bottleneck-128 output interface → Mamba blocks [13,24) → frozen norm
→ tied host head. Host partition [11,13) is held fixed for every Campaign 5 candidate; only
the donor sector [start, stop) varies, so width/placement is the single experimental
variable. Trainable parameters (893,696) are identical for every candidate and twin because
interfaces see only the 768/1536 boundary widths. Diagrams: this file is the source-controlled
description; the machine-readable graph is campaign_05_sector_matrix.json.
