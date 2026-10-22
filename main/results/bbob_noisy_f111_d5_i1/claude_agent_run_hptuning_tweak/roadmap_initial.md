# Roadmap — Initial Step

## Understood problem

- Black-box objective, sealed (identity unknown, not to be investigated).
- Dimension: 5. Bounds: [-5, 5] per coordinate (box-constrained).
- Direction: minimize.
- Noise: the objective is **noisy** — repeated `evaluate()` calls at the same
  `x` can return different `y`. This is real observation noise, not
  deterministic drift, so single-point evaluations are only a noisy estimate
  of the true value at that point. Implications: finite-difference gradients
  are unreliable without replication; "best observed so far" can be
  misleading if the current incumbent happened to get a lucky low noise
  draw; methods that assume noiseless evaluations (e.g. a plain
  improvement-based acquisition) can be fooled by that. I will keep this in
  mind throughout and consider replicating/re-checking promising points
  rather than trusting a single low observation blindly.
- Total evaluation budget: 100 `evaluate()` calls, hard-enforced by the
  harness. This is a small budget for dim=5 (~20x dim), so evaluations are
  precious; initialization, finite-difference probes, and repeated
  measurements all count against it.
- Harness: `AgentTaskHarness(budget=100)`, `harness.evaluate(x)`,
  `harness.set_active_optimizer(name)`, `harness.save()`. Budget is enforced
  in code (`BudgetExhausted`); trace/meta are written automatically.

## Method for this step

**Method:** GP-BO (`src/optimizers/gp_bo.py`), sequential Bayesian
optimization with a `SingleTaskGP` and the analytic `LogExpectedImprovement`
acquisition function (BoTorch).

**Why this method suits the stated problem:** The module's own docstring
documents that this implementation is sized for small budgets of order
"10 x dim", which matches this task almost exactly (budget=100, dim=5, i.e.
20x dim). Exact GP inference is affordable at this scale, and a
surrogate-model approach is well suited to a tight, expensive-feel budget
where every evaluation must be used efficiently, rather than relying on
population-based methods (CMA-ES, DE) that typically want many more
evaluations to converge in 5D. `SingleTaskGP` also fits an observation-noise
term from the data by default, which partially addresses the stated noise
(the GP will not literally interpolate noisy points), even though the LogEI
acquisition formula itself is the "noiseless" variant that conditions on the
single best observed `y`. I will watch for this at the first checkpoint
(e.g. by checking the initial design's value spread) and will consider
replicating the incumbent if a single low point looks like it could be a
noise outlier.

**This step's content — initial Sobol design:** Before any GP can be fit
meaningfully, GP-BO (like the project's own documentation of this method)
requires an initial space-filling design. I am using the method's own
built-in rule: `n_init = 2 * (dim + 1) = 12` scrambled Sobol' points
(`SobolSearch`, scipy's own default scrambling), which is what
`GPBO.ask()` returns automatically until 12 points have been told. This is
an inherited default from the implementation, not a new tuned value — I am
adopting it because the module's own docstring calls it "a standard rule of
thumb for how many points are needed before a GP fit is meaningful," which
matches the purpose of this first step (pure exploration, no model yet).

No other GP-BO hyperparameters apply yet at this stage (the GP, kernel,
priors, and acquisition optimizer settings only come into play once the
first acquisition-driven point is proposed, after the 12 initial points).

## Step length

**This step: 12 objective evaluations** (cumulative evaluation count 12 of
100 after this step). This is not an arbitrary choice: 12 is exactly
`2*(dim+1)`, the initial design size the chosen method defines as the
minimum needed before its surrogate model is meaningful. Stopping exactly
at the init design boundary is the natural place for a first checkpoint: it
is the smallest useful block (fewer points would leave the GP badly
under-determined for 5D), and it is cheap relative to the 100-call budget
(12%), leaving ample budget for the review-and-adapt loop that follows.

## What I want to learn at the checkpoint / pause conditions

At evaluation 12, I will inspect:
- The spread and rough shape of the 12 observed `y` values (range, whether
  one region looks clearly more promising), to judge whether the space
  looks smooth/unimodal or rugged, which bears on whether GP-BO's local
  refinement is likely to pay off versus a more global/population method.
- A rough sense of the noise level: although no point is replicated yet in
  this step, if later checkpoints call for replication I will set the
  granularity using what's observed about value spread here.
- Whether anything indicates the init design should be discarded (e.g. a
  bug/outlier such as a non-finite value), which would call for a different
  immediate reaction.

I do not pre-commit to continuing with GP-BO's acquisition phase after this
step — that decision, including its own hyperparameters (number of
acquisition-driven points before the next checkpoint, num_restarts/
raw_samples for `optimize_acqf`, whether to replicate points for noise), will
be made at the checkpoint in `changelog.md`, based on the 12 observations
collected here. There is no condition for pausing earlier than evaluation
12: these are independent space-filling points with no feedback loop within
the batch, so there is nothing to learn mid-batch that would justify
stopping sooner.
