"""Shared visual style for this standalone, already-audited OPV report."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
ROOT = OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'opv-random-start-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
INK = '#172334'
FLOOR = 1e-8
TITLES = {'opv_pool': 'OPV device pool'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'savefig.dpi': 220, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def read_json(path):
    return json.loads(path.read_text())

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def axes_style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=.18, linewidth=.7)
    ax.set_axisbelow(True)

def save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(directory / f'{stem}.{ext}', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def method_color(label):
    text = label.lower()
    if 'gp' in text or 'bayes' in text: return '#df8435'
    if 'turbo' in text: return '#8e5eb3'
    if 'cma' in text: return '#247bb2'
    if 'sobol' in text or 'design' in text or 'random' in text: return '#488e78'
    if 'probe' in text or 'repeat' in text: return '#9c7174'
    return ['#b29b36', '#ba5d86', '#557886'][int(hashlib.sha256(label.encode()).hexdigest()[:8],16)%3]

def shorten(label):
    return label.replace('_', ' ')
