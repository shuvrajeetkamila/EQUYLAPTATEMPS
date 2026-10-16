"""Empirically rank complete donor sectors, rather than pretending 'math neurons' are known."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .audit import module_fingerprint
from .data import Record, ensure_disjoint
from .evaluation import evaluate_records
from .model import child_from_parents
from .parents import LoadedParents
from .recipe import LayerSpan, Recipe
from .training import TrainingOptions, fit_seams


def search_sectors(
    recipe: Recipe,
    parents: LoadedParents,
    train: list[Record],
    validation: list[Record],
    options: TrainingOptions,
    *,
    donor_depth: int,
    stride: int = 1,
    progress: Callable[[dict], None] | None = None,
):
    """Reuse frozen source tensors; keep only the winner's independent seam parameters."""
    recipe.validate(parents.recipient.config, parents.donor.config)
    options.validate()
    ensure_disjoint(train, validation)
    depth = parents.donor.config.num_hidden_layers
    if (
        type(donor_depth) is not int
        or not 1 <= donor_depth <= depth
        or type(stride) is not int
        or stride < 1
    ):
        raise ValueError("donor_depth must fit the donor; stride must be positive")
    starts = list(range(0, depth - donor_depth + 1, stride))
    if starts[-1] != depth - donor_depth:
        starts.append(depth - donor_depth)
    candidates = []
    best_child, best_score, best_training = None, float("inf"), None
    for start in starts:
        trial_recipe = replace(recipe, donor_sector=LayerSpan(start, start + donor_depth))
        child = child_from_parents(trial_recipe, parents.recipient, parents.donor)
        training = fit_seams(child, parents.tokenizer, train, options, validation=validation)
        score = training["validation"]["groups"]["overall"]["token_nll"]
        candidate = {
            "donor_sector": [start, start + donor_depth],
            "heldout_response_token_nll": score,
            "validation_by_domain": training["validation"]["groups"],
            "frozen_tensors_unchanged": training["frozen_tensors_unchanged"],
        }
        candidates.append(candidate)
        if progress:
            progress(candidate)
        if score < best_score:
            best_child, best_score, best_training = child, score, training
    best_child.eval()
    bypass = evaluate_records(
        best_child,
        parents.tokenizer,
        validation,
        max_length=options.max_length,
        batch_size=options.batch_size,
        ablation="bypass_graft",
    )
    report = {
        "selection_metric": "lowest held-out response-token NLL in the recipient vocabulary",
        "selected_sector": [
            best_child.recipe.donor_sector.start,
            best_child.recipe.donor_sector.stop,
        ],
        "candidate_count": len(candidates),
        "candidates": sorted(candidates, key=lambda c: c["heldout_response_token_nll"]),
        "selected_calibration": best_training,
        "selected_bypass_ablation": bypass,
        "bypass_minus_full_token_nll": bypass["groups"]["overall"]["token_nll"] - best_score,
        "warning": (
            "Selection uses validation data, NOT the final test set. Contiguous sectors contain "
            "distributed features; winning this search does not identify an isolated math organ. "
            "A positive bypass delta is only an ablation signal, "
            "not a general capability guarantee."
        ),
    }
    provenance = dict(parents.provenance)
    provenance["source_donor_sector_sha256"] = module_fingerprint(best_child.graft.layers)
    return best_child, provenance, report
