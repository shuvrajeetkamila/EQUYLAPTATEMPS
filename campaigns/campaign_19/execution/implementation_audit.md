# Campaign 19 execution — Implementation audit

Runner `scripts/campaign19_runner.py` (gates/train modes) executed
the frozen matrix via the Campaign 15/16 resumable training
procedure. One implementation failure (F-C19-001): the runner
initially changed `TrainingOptions.steps` between training
segments, which the resumable contract correctly rejected; fixed to
the registered `interrupt_after` pattern before any run produced
scores, design untouched, failed attempt preserved
(`train_38947619114/`). Publisher derives every number from the
two SHA-verified bundles.
