# Figure Standard

Rules for figures that leave the repo (papers, slides, posters). Diagnostic plots from `scripts/make_comparison_plots.py` predate this standard and are not bound by it.

The rules are implemented in `scripts/figure_style.py`. A figure script imports it and only chooses tasks, series and titles. If a rule needs to change, change it in `figure_style.py` and in this file together.

Reference figure: `results_linux/crossover_rosenbrock_vs_mixint.png`, produced by `scripts/make_crossover_plot.py`.

## Content

- One message per figure. If the point needs a sentence of explanation, that sentence goes in the caption, not the figure.
- No annotations, callouts, markers or "X wins" text inside the plot area.
- Panel titles are one or two words naming what differs between panels (e.g. "Smooth", "Mixed Integer"), not the task ID.
- Legend labels name the method family ("Gradient Based", "Evolutionary Based"), not the solver. Name the solvers in the caption.
- Every plotted value comes from committed `trace.csv`/`meta.json` files. No hand-entered numbers.

## Layout

- One panel per task in a single row; `FIGSIZE_1x2` = 11 x 4.2 in for two panels.
- One legend for the whole figure, centered below the panels, no frame, one row.
- Titles centered above each panel.

## Axes

- x: "Budget" (objective evaluations), from 0 to the run's budget.
- y: "Regret", log scale. Label it on the leftmost panel only; each panel keeps its own tick labels.
- Regret 0 is drawn at the floor 1e-12.
- Left and bottom spines only, `#52514e`, 1.2 pt. No gridlines.

## Text

- All text black (`#000000`): titles, axis labels, tick labels, legend.
- Sizes: titles 20 pt, axis labels 13 pt, tick labels 12 pt, legend 12 pt.

## Series

- Solid lines only, 3 pt.
- Several seeds: geometric mean (mean of log10 regret) with a +/- 1 standard-error band computed in log space, alpha 0.18. State the seed count in the caption.
- One seed (deterministic methods: L-BFGS-B, BOBYQA, Nelder-Mead): plot that run, no band.
- A run that stops before the budget is held flat at its best value, still solid. Say in the caption that it stopped and after how many evaluations.

## Color

- Series colors in fixed order: series 1 `#2a78d6` (blue), series 2 `#e34948` (red), series 3 `#1baf7a` (aqua), series 4 `#eda100` (yellow), series 5 `#008300` (green). The same method keeps the same color across every panel of a figure.
- The first two pass every check of the dataviz palette validator (`validate_palette.py --mode light --surface #ffffff`). The full five pass with `--pairs all`, with warnings: aqua and yellow are below 3:1 contrast on white and aqua/red is in the deutan 6-8 band, so a figure using 3+ series must keep the legend. A figure with more series must extend `SERIES_COLORS` and re-run the validator on the full ordered list before use.
- Background pure white (`#ffffff`) everywhere: figure, axes and the saved file. No off-white or transparent backgrounds.

## Schematics

Illustrations of a task's shape (e.g. `results_linux/landscape_both.png`, from `scripts/make_landscape_plot.py`) are the one exception to the committed-data rule: they may draw a known formula. They follow every rule above that applies, plus:

- Say in the script docstring and the caption that the graphic is schematic, and what was simplified (dimension, grid spacing).
- No axes, ticks or colorbar: a schematic's coordinates carry no meaning. The caption states what color means (e.g. "lighter = lower loss").
- Color magnitude with one light-to-dark ramp of `SERIES_COLORS[0]`, on one scale shared by every panel.

## Output

- PNG at 200 dpi, `bbox_inches="tight"`, written next to the data it plots (e.g. `results_linux/`).
- One script per figure in `scripts/`, named `make_<figure>_plot.py`, with no required arguments. Its docstring lists the tasks and methods shown and what each panel demonstrates.
- Run it from `main/` with the project venv:

```bash
../.venv/bin/python scripts/make_crossover_plot.py
```

## Caption checklist

- What each panel's task is (task ID and dimension).
- Which solver each legend entry is.
- What the line and band are, and the seed count.
- Any run that stopped early, and when.
