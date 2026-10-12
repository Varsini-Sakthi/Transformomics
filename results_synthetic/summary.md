# Transformomics v2 results

Primary metric: mean Pearson r across programmes; units of replication = 4 spatial-block folds (seeds averaged within fold).

## Mean score per method

| method               |   mean |   sd_over_folds |   n_folds |
|:---------------------|-------:|----------------:|----------:|
| ridge_own            | 0.6863 |          0.0779 |         4 |
| ridge_own_nbr_mean   | 0.7136 |          0.0732 |         4 |
| transformer_none     | 0.6671 |          0.0755 |         4 |
| transformer_real     | 0.7271 |          0.0665 |         4 |
| transformer_shuffled | 0.6715 |          0.0767 |         4 |

## Paired comparisons (A - B)

| A                  | B                    |   mean_diff_A_minus_B |   sd_over_folds | folds_A_better   |
|:-------------------|:---------------------|----------------------:|----------------:|:-----------------|
| transformer_real   | transformer_shuffled |                0.0556 |          0.0113 | 4/4              |
| transformer_real   | transformer_none     |                0.06   |          0.0111 | 4/4              |
| transformer_real   | ridge_own_nbr_mean   |                0.0135 |          0.0081 | 4/4              |
| transformer_real   | ridge_own            |                0.0408 |          0.0134 | 4/4              |
| ridge_own_nbr_mean | ridge_own            |                0.0273 |          0.0053 | 4/4              |

## Reading this honestly

- Real vs shuffled neighbors: diff = +0.0556 (4/4 folds favour real). Only a positive, consistent gap supports 'the model uses spatial context'.
- Transformer vs linear own+neighbor-mean baseline: diff = +0.0135 (4/4 folds favour transformer). If this is <= 0, simple neighborhood smoothing explains the benefit and the attention model is not needed.
- Attention weights are descriptive, not proof of importance; rely on the ablations above.
- With 4 folds from one section, differences are indicative only; confirm on held-out sections/donors before drawing biological conclusions.
- Mean CLS attention mass on neighbor tokens (last layer): 0.857 (uniform attention over all 8 tokens would give ~0.75; compare, don't over-read).
