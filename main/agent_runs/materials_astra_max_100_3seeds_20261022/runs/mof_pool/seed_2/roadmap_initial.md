# First optimization block

## Problem and assumptions
Minimize the deterministic black-box response over exactly 1,000 public candidate rows, each with 193 features. Only unused row indices are valid; no continuous proposals or repeat purchases. The total hard budget is 100 objective evaluations. Campaign seed is 2. There are no prior observations or additional constraints, and no known objective lower bound. The objective identity and all unobserved labels remain sealed.

Public feature inspection found 165 varying columns and 1,000 unique feature rows. After bound normalization, 14, 20, and 42 principal components explain 90%, 95%, and 99% of feature variance. This is feature geometry only, not evidence about the objective. All public features may be used for experimental design; only purchased labels may train a response model.

## Method, initialization, and hyperparameters
Use a representative cluster design, tagged `cluster_design`. Remove the 28 constant columns from distance calculations and scale each varying feature by its public bound range. Fit KMeans with 16 clusters, k-means++ initialization, 10 initializations, max_iter=300, tol=1e-4, algorithm=lloyd, random_state=2. In each cluster select the actual member closest to its centroid, breaking ties by the first row index. Purchase the 16 representatives in a permutation from NumPy default_rng(2). No objective labels inform this selection.

Implementation: scikit-learn 1.9.1 KMeans; NumPy 2.4.6; SciPy 1.17.1. KMeans copy_x=True and other unspecified behavior use this installed version's defaults. The representative design gives coverage of populated feature regions while keeping initialization modest relative to the budget. Bound scaling prevents large-unit features from dominating the distance; correlations remain and their relevance to the target is unknown.

## This block only
Maximum 16 additional objective calls; pause at cumulative evaluation 16. Sixteen representatives provide an initial view of response variation across the pool's main feature structure. They are not intended to identify all 165 feature effects. This block leaves the next method and its budget undecided until the observed data have been inspected.

At the checkpoint inspect every purchased value, the best row, response range and quantiles, cluster sizes, and whether low-valued rows occupy related feature regions. Save the harness trace and metadata. Stop immediately on an evaluator error, a nonfinite result, or an invalid selection; never retry an unknown evaluation. There is no value-based early stopping threshold without prior response knowledge.

## Execution integrity
One persistent harness owns the budget and all labels. A separate per-block guard bounds calls, and each selection must belong to h.available_indices. Scripts are kept in work/. The harness alone writes trace.csv and meta.json. This initial decision is immutable; all subsequent decisions will be appended to changelog.md before further evaluations.
