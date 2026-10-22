"""Animated version of make_agent_switching_plot.py: both panels grow together,
evaluation by evaluation, so each switch between optimizer families can be seen
changing the path and the regret curve at the same moment.

Every frame is built by make_agent_switching_plot.build_figure, so it follows
the same style (FIGURES.md) and the same data. The background slice is
computed once. Frames step through the run at a fixed number of evaluations
each; the last frame (the full run) is held so the viewer can read it.

Usage (from main/):
    python scripts/make_agent_switching_gif.py    # writes results_linux/agent_switching_rosenbrock.gif
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_agent_switching_plot as sw  # noqa: E402  (also applies figure_style rcParams)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

OUT = sw.MAIN / "results_linux/agent_switching_rosenbrock.gif"
N_FRAMES = 120
FRAME_MS = 80
HOLD_MS = 3000
DPI = 130  # keeps the GIF small; the PNG stays at the standard's 200 dpi


def to_shared_palette(frames, n_colors=128):
    """Map every frame onto one palette built from the last (fullest) frame.

    Median cut averages white with nearby light grays, so pure white is pinned
    as entry 0 (FIGURES.md: white backgrounds). Pixels are matched to the exact
    nearest entry here because Pillow's own palette lookup rounds colors coarsely
    enough to send white to an off-white entry.
    """
    entries = frames[-1].quantize(colors=n_colors - 1, method=Image.Quantize.MEDIANCUT).getpalette()
    palette = np.vstack([[255, 255, 255], np.array(entries[: (n_colors - 1) * 3]).reshape(-1, 3)])
    out = []
    for frame in frames:
        rgb = np.asarray(frame).reshape(-1, 3)
        colors, inverse = np.unique(rgb, axis=0, return_inverse=True)
        nearest = ((colors[:, None, :].astype(int) - palette[None]) ** 2).sum(-1).argmin(1)
        img = Image.fromarray(nearest[inverse.ravel()].reshape(frame.size[::-1]).astype(np.uint8), mode="P")
        img.putpalette(palette.astype(np.uint8).ravel().tolist())
        out.append(img)
    return out


def main():
    X, regret, stage = sw.load_run()
    f_opt = json.loads(sw.F_OPT_META.read_text())["task_f_opt"]
    geo = sw.slice_geometry(X, regret, stage, f_opt)
    n = len(regret)
    uptos = sorted({max(1, round(n * (k + 1) / N_FRAMES)) for k in range(N_FRAMES)})
    # Crop every frame to the full run's tight bounding box: one fixed crop keeps
    # all frames the same size and leaves room for the legend below the panels.
    fig = sw.build_figure(geo, regret, stage, n, dpi=DPI)
    crop = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    plt.close(fig)
    frames = []
    for upto in uptos:
        fig = sw.build_figure(geo, regret, stage, upto, dpi=DPI)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches=crop)
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB"))
    frames = to_shared_palette(frames)
    durations = [FRAME_MS] * (len(frames) - 1) + [HOLD_MS]
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {OUT} ({len(frames)} frames, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
