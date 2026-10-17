# Pre-specified hypotheses and decision rules (edit BEFORE running on real data, then commit)

Fill the thresholds in, commit this file, and only then run. Any change afterwards must be recorded as a
deviation. The point is that a null or negative result is an acceptable, reportable outcome.

Dataset(s): ____________   Task: label / program   Replication unit: donor / section   n groups: ___
Boundary definition: boundary = fraction-different >= ___ (default 2/6); interior = 0.   Labels from: ______
Marker sets (source + date fixed): ______________________

| # | Hypothesis | Supported only if ALL hold | Otherwise report as |
|---|---|---|---|
| H0 | Pipeline has an artefact floor | `--null-coords` run done with >= ___ seeds; floor = mean(real - shuffled) = ___ | (required before H1) |
| H1 | Neighbour expression carries information | real > shuffled AND real > none in every held-out group; mean gap >= ___ AND > null floor + 2 SD | No evidence neighbours help |
| H2 | Attention adds beyond smoothing | real > `*_own_nbr_mean` in >= ___ of ___ groups (also vs an MLP if added) | Smoothing suffices |
| H3 | Neighbours matter more at boundaries | adjusted effect > 0 for BOTH `attn_excess_roll` and `benefit_vs_shuffled`, positive in every group, shuffled-model attention shows no boundary effect | No boundary effect / confounded by ambiguity |
| H4 | Attribution recovers known markers | `ig_query` mean rank-AUC > `logreg_coef` and `de_auc` and q < 0.1 in >= half of classes with markers | Attribution no better than simple baselines |
| H5 | Attention is faithful | same-label mass > `expr_cosine` in every group AND deletion order top > random > bottom | Attention not shown to be informative |

Reporting rules
* Report every class/outcome/group, including the unfavourable ones.
* p-values are spot-level and anti-conservative (neighbouring spots are correlated); decisions use the
  per-group consistency columns, not p alone.
* If H0's floor is comparable to the observed H1 gap, H1 is not supported, whatever the p-value.
* Wording for the resume/README follows the table: "supported", "not supported", or "not tested" only.
