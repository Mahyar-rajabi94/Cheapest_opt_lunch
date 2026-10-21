# Probe feature thresholds: evidence and caveats

The thresholds in the probing skill come from probe runs at d=5 and d=10 on
development problems of known types (separable bowl, curved valley,
ill-conditioned bowl, multimodal with and without global structure).
They are rules of thumb, not guarantees.

Observed typical values (d=10, default probe budget ≈ 12% of 2,000):

| Landscape type | P4 gain | P4 y_spread | P4 contraction | P5 cosine | P2 fdc | P3 R² (quad, rank) |
|---|---|---|---|---|---|---|
| smooth bowl | 1.0 | ~0 | ~0 | 1.0 | 0.85 | 0.95 |
| curved valley | 0.97 | 0.05 | 0.47 | 1.0 | 0.65 | 0.89 |
| ill-conditioned bowl | 0.40 | 0.60 | 0.92 | 1.0 | 0.43 | 0.88 |
| multimodal, global structure | 0.76 | 0.89 | 0.80 | 1.0 | 0.61 | 0.82 |
| multimodal, weak structure | 0.56 | 0.77 | 0.89 | 1.0 | 0.30 | 0.46 |

Caveats:
- **P5 cannot detect multimodality.** Most multimodal test landscapes are
  smooth at small scales, so cosine ≈ 1 there too. P5 tells you whether
  gradients are *usable*, not whether there is one basin. Use P4 for basins.
- **Ill-conditioning looks like several basins to P4.** Short local runs stop
  before reaching the bottom of a stretched valley, so their endpoints
  differ. A high P3 quadratic R² together with high y_spread suggests
  ill-conditioning rather than true multimodality.
- In low dimension (d ≤ 5) P4 runs converge more often, so y_spread is more
  reliable there.
