import os
from dataclasses import replace

# Set before importing Transformers: tests must never download a model or tokenizer.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytest
import torch

from equylaptatemps.audit import module_fingerprint
from equylaptatemps.model import child_from_parents
from equylaptatemps.synthetic import demo_recipe, tiny_parent
from equylaptatemps.tokenization import ByteTokenizer, tokenizer_fingerprint

torch.set_num_threads(1)


@pytest.fixture
def child_bundle():
    recipe = demo_recipe()
    recipient = tiny_parent("mamba", 117)
    donor = tiny_parent("qwen2", 217)
    child = child_from_parents(recipe, recipient, donor)
    tokenizer, donor_tokenizer = ByteTokenizer(), ByteTokenizer(37)
    provenance = recipe.stranger_report()
    provenance["source_donor_sector_sha256"] = module_fingerprint(child.graft.layers)
    provenance["tokenizers"] = {
        "recipient_sha256": tokenizer_fingerprint(tokenizer),
        "donor_sha256": tokenizer_fingerprint(donor_tokenizer),
    }
    return child, tokenizer, provenance, recipient, donor


@pytest.fixture
def local_parents(tmp_path, child_bundle):
    child, tokenizer, _, recipient, donor = child_bundle
    a_path, b_path = tmp_path / "mamba", tmp_path / "qwen"
    recipient.save_pretrained(a_path, safe_serialization=True)
    donor.save_pretrained(b_path, safe_serialization=True)
    tokenizer.save_pretrained(str(a_path))
    ByteTokenizer(37).save_pretrained(str(b_path))
    recipe = replace(
        child.recipe,
        recipient=replace(child.recipe.recipient, local_path=str(a_path)),
        donor=replace(child.recipe.donor, local_path=str(b_path)),
    )
    return recipe, a_path, b_path
