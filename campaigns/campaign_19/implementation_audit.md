# Campaign 19 — Implementation audit (preparation)

Implemented and offline-tested: foreign-graft builder, scramble and
random controls (see `implementation_design.md`,
`tests/test_campaign19_foreign_graft.py`, 6 tests). Verified:
execution of a synthetic Mamba sector inside a synthetic Llama
recipient across unequal widths; depth preserved; unrelated
recipient blocks fingerprint-identical; donor parent immutable
under scramble/randomize; trainable inventory = connectors only;
length-changing grafts rejected; bypass ablation executes.

Not implemented (requires authorization): the real-weight runner,
workflows, training, D0/B gate execution, held-out acquisition.
Historical code (`controls.py`, `child_from_parents`, prior
campaigns) untouched.
