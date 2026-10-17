# Changelog

## v2.1
* Multi-section real-data pipeline (`multisection.py`, `run_multi.py`): per-section graphs, leave-one-donor/section-out, training-only preprocessing, optional per-section `restrict`.
* Per-spot outputs (loss, attention last-layer + rollout) from `fit_predict` (`want_spot_attn`, `return_model`).
* `Transformomics.forward(nbr_mask=...)` hides individual neighbour tokens (deletion test).
* Boundary analysis, marker enrichment, integrated gradients, attention-vs-baseline faithfulness, deletion test.
* `--null-coords` coordinate-permutation run to measure the real-vs-shuffled artefact floor.
* Synthetic multi-section generator, configs (DLPFC, 5xFAD), data scripts, dataset guide, pre-registration template.
* Not changed: model architecture and training protocol of v2.0.
