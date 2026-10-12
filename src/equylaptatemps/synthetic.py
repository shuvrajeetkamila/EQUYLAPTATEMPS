"""Heterogeneous architecture smoke test, explicitly NOT a pretrained capability demo."""

from __future__ import annotations

import torch
import transformers
from transformers import AutoModelForCausalLM, MambaConfig, Qwen2Config

from .audit import frozen_fingerprint, module_fingerprint
from .backbones import extract_parent
from .data import Record
from .evaluation import evaluate_records
from .model import child_from_parents
from .recipe import LayerSpan, ParentSpec, Recipe
from .tokenization import ByteTokenizer, tokenizer_fingerprint
from .training import TrainingOptions, fit_seams


def demo_recipe(seed: int = 17, *, reverse: bool = False) -> Recipe:
    a, b = ("qwen2", "mamba") if reverse else ("mamba", "qwen2")
    return Recipe(
        ParentSpec(f"synthetic://{a}", None, None),
        ParentSpec(f"synthetic://{b}", None, None),
        LayerSpan(1, 3),
        LayerSpan(1, 3),
        seed,
    )


def tiny_parent(family: str, seed: int):
    if family == "mamba":
        config = MambaConfig(
            vocab_size=259,
            hidden_size=16,
            num_hidden_layers=4,
            state_size=4,
            expand=2,
            conv_kernel=3,
            time_step_rank=4,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            use_cache=False,
            use_mambapy=False,
        )
    elif family == "qwen2":
        config = Qwen2Config(
            vocab_size=311,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            use_cache=False,
            attention_dropout=0.0,
        )
    else:
        raise ValueError(f"Unknown synthetic family {family}")
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return AutoModelForCausalLM.from_config(config).eval()


def make_synthetic_child(recipe: Recipe | None = None):
    recipe = recipe or demo_recipe()
    if not recipe.synthetic:
        raise ValueError("Random initialization is reserved for explicitly synthetic recipes")
    recipient = tiny_parent(recipe.recipient.identity.family, recipe.seed + 100)
    donor = tiny_parent(recipe.donor.identity.family, recipe.seed + 200)
    child = child_from_parents(recipe, recipient, donor)
    tokenizer = ByteTokenizer()
    donor_tokenizer = ByteTokenizer(rotation=37)
    original = torch.nn.ModuleList(
        extract_parent(donor).layers[recipe.donor_sector.start : recipe.donor_sector.stop]
    )
    source_hash = module_fingerprint(original)
    if source_hash != module_fingerprint(child.graft.layers):
        raise RuntimeError("Synthetic donor tensors were not transplanted intact")
    provenance = recipe.stranger_report()
    provenance.update(
        {
            "initialization": "random; neither parent has learned any real-world capability",
            "tokenizers": {
                "recipient_sha256": tokenizer_fingerprint(tokenizer),
                "donor_sha256": tokenizer_fingerprint(donor_tokenizer),
            },
            "source_donor_sector_sha256": source_hash,
        }
    )
    return child, tokenizer, provenance, recipient


def run_demo(*, steps: int = 40, seed: int = 17):
    child, tokenizer, provenance, recipient = make_synthetic_child(demo_recipe(seed))
    records = [
        Record(f"{a}+{b}=", str(a + b), "arithmetic")
        for a, b in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5)]
    ]
    heldout = [Record("5+1=", "6", "arithmetic"), Record("5+2=", "7", "arithmetic")]
    before = frozen_fingerprint(child)
    metrics = fit_seams(
        child,
        tokenizer,
        records,
        TrainingOptions(steps, 4, 0.02, 64, seed, True),
        validation=heldout,
    )
    full = evaluate_records(child, tokenizer, heldout, max_length=64, batch_size=2)
    bypass = evaluate_records(
        child, tokenizer, heldout, max_length=64, batch_size=2, ablation="bypass_graft"
    )
    zero = evaluate_records(
        child, tokenizer, heldout, max_length=64, batch_size=2, ablation="zero_graft"
    )
    baseline = evaluate_records(recipient, tokenizer, heldout, max_length=64, batch_size=2)
    report = {
        "run_kind": "offline synthetic architecture and gradient smoke test",
        "pretrained_parent_weights_used": False,
        "real_world_capabilities_validated": False,
        "seed": seed,
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
        "stranger_checks": provenance,
        "structure": child.structure_report(),
        "calibration": metrics,
        "heldout_ablations": {"full_child": full, "bypass_graft": bypass, "zero_graft": zero},
        "random_recipient_baseline": baseline,
        "frozen_tensors_unchanged": frozen_fingerprint(child) == before,
        "limitation": "Random-parent loss changes do not establish real skill transfer.",
    }
    return child, tokenizer, provenance, report
