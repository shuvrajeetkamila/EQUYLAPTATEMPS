# Campaign 4 — Test Report

Offline suite at publication: 227 tests passed locally (223 historical +
3 mandatory-use mechanics + 1 results-traceability), ruff lint and format
clean. Unit evidence includes: bottleneck causality under future-token
perturbation, donor-delta behavior (full vs donor-removed computations
differ; identity donor yields zero suffix signal by construction),
trainable/frozen contract with gradient isolation, and bridge-only
reference reconstruction with identical logits for bottleneck+delta
children. Connected Python 3.10/3.11 CI status is verified on the final
head and recorded in the PR. Real-parent correctness evidence comes from
the registered Actions jobs (frozen fingerprints unchanged; native-block
call traces exactly once per block; finite logits enforced in-runner).
