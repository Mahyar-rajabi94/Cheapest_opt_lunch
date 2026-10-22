"""Score and render completed campaigns on the trusted supervisor side only."""
import argparse
import csv
import html
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import textwrap
import traceback

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

spec = importlib.util.spec_from_file_location('comparison', REPO / 'main/scripts/make_comparison_plots.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'axes.titlesize': 16, 'axes.labelsize': 11,
                     'savefig.dpi': 220, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
FLOOR = 1e-12
AGENT = '#172334'

def timeline_color(label):
    raw = label.lower()
    for prefix, canonical in [('cma', 'cma_es'), ('lbfgs', 'lbfgsb'), ('bobyqa', 'bobyqa'), ('sobol', 'sobol')]:
        if raw.startswith(prefix):
            return comparison.baseline_style[canonical]['color']
    if raw.startswith('slsqp'):
        return '#2a78d6'
    if 'surrogate' in raw:
        return '#6250d6'
    canonical = comparison.canonical_method(label)
    if canonical in comparison.baseline_style:
        return comparison.baseline_style[canonical]['color']
    palette = ['#bc5b24', '#287a88', '#95569b', '#596f38', '#64748b']
    return palette[int(hashlib.sha256(label.encode()).hexdigest()[:8],16) % len(palette)]

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def savefig(fig, directory, stem):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(directory / (stem + '.' + ext), bbox_inches='tight', facecolor='white')
    plt.close(fig)

def title(task):
    return task['task_name'].replace('_', ' ')

def clean_axes(ax):
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=.18, linewidth=.7)
    ax.set_axisbelow(True)

def plot_lines(ax, baselines, agent, legend=True):
    for method, b in baselines.items():
        style = comparison.baseline_style[method]
        ax.plot(b['x'], np.maximum(b['regret'], FLOOR), label=comparison.baseline_display_names[method],
                color=style['color'], linestyle=style.get('linestyle', '-'), alpha=.85, lw=1.5)
    ax.plot(agent['x'], np.maximum(agent['regret'], FLOOR), color=AGENT, lw=2.8,
            label=agent.get('run_label','Astra max agent (1 run)'), zorder=5)
    ax.set(xlim=(0, 2000), yscale='log', xlabel='Objective evaluations', ylabel='Incumbent regret (log scale)')
    clean_axes(ax)
    if legend:
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

def decisions(run, trace):
    with sqlite3.connect('file:' + str(run / 'private/evaluations.sqlite') + '?mode=ro', uri=True) as db:
        snapshots = db.execute('SELECT eval_idx,name,content FROM snapshots ORDER BY rowid').fetchall()
    n = len(trace)
    checkpoints = {0}
    chunks = {}
    previous = ''
    for idx, name, content in snapshots:
        if name == 'roadmap_initial.md':
            chunks[0] = content
        elif name == 'changelog.md':
            delta = content[len(previous):] if content.startswith(previous) else content
            previous = content
            if delta.strip() and 0 <= idx < n:
                checkpoints.add(idx)
                chunks[idx] = chunks.get(idx, '') + delta
    # A logged block can deliberately contain several labels (for example,
    # repeated noise probes followed by CMA-ES). Only a new prospective
    # document snapshot denotes a new decision checkpoint.
    boundaries = sorted(checkpoints | {n})
    result = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        segments=[]
        for i in range(start,end):
            label=trace[i]['optimizer']
            if segments and segments[-1]['method']==label:
                segments[-1]['last_eval']=i+1
            else:
                segments.append({'method':label,'first_eval':i+1,'last_eval':i+1})
        label=' → '.join(segment['method'] for segment in segments)
        result.append({'block': len(result)+1, 'method': label, 'first_eval': start+1, 'last_eval': end,
                       'calls': end-start, 'boundary_source': 'prospective document snapshot',
                       'method_segments':segments,
                       'decision_text': chunks.get(start, '')})
    return result

def final_seed_regrets(baseline_dir, method, f_opt):
    values = []
    for p in sorted((baseline_dir / method).glob('seed_*/trace.csv')):
        lines = p.read_text().splitlines()
        header = next(csv.reader([lines[0]]))
        last = next(csv.reader([lines[-1]]))
        value = last[header.index('y_best_so_far')]
        values.append(float(value)-f_opt if value else math.nan)
    return values

def render(task, force=False):
    key = task['task']
    annotation_path = ROOT / 'annotations.json'
    annotation = json.loads(annotation_path.read_text()).get(key,{}) if annotation_path.exists() else {}
    run = Path(task['run_dir'])
    manifest = json.loads((run / 'private/manifest.json').read_text())
    audit = json.loads((run / 'private/integrity_audit.json').read_text())
    assert manifest['model'] == 'gpt-6-astra' and manifest['reasoning_effort'] == 'max'
    assert manifest['budget'] == 2000
    assert audit['artifacts_match_journal'] and audit['unknown_evaluations'] == 0
    assert audit['only_workspace_execution_tools']
    assert task['returncode'] == 0
    directory = ROOT / 'figures' / task['task_name']
    done = directory / 'result.json'
    if done.exists() and not force:
        return json.loads(done.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    public = run / 'workspace/results/agent_run'
    trace = rows(public / 'trace.csv')
    scored = rows(run / 'private/scored_trace.csv')
    assert 0 < len(trace) <= 2000 and len(trace) == len(scored) == audit['evaluations_used']
    assert [int(x['eval_idx']) for x in trace] == list(range(1,len(trace)+1))
    analysis = ROOT / 'analysis' / task['task_name']
    analysis.mkdir(parents=True, exist_ok=True)
    for a, b in zip(trace, scored):
        assert a['eval_idx'] == b['eval_idx'] and float(a['y']) == float(b['y'])
        a['y_best_so_far'] = b['y_best_so_far']
        a['y_true'] = b['y_true']
    with (analysis / 'trace.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0])); w.writeheader(); w.writerows(trace)
    for name in ('meta.json', 'changelog.md', 'roadmap_initial.md'):
        shutil.copyfile(public / name, analysis / name)
    baseline_dir = REPO / 'main/results_full' / task['task_name']
    baselines, agents = comparison.load_task(baseline_dir, analysis, 2000)
    assert len(baselines) == 9 and len(agents) == 1
    agent = agents['Agent']
    if annotation.get('status') == 'execution_failure':
        agent['run_label'] = 'Astra max agent (execution failure)'
    blocks = decisions(run, trace)
    (analysis / 'decisions.json').write_text(json.dumps(blocks, indent=2)+'\n')
    noisy_note = 'Noisy task: score uses the noise-free value of the lowest-observed-y incumbent. ' if task['noisy'] else ''
    feasibility_note = 'Means include seeds with a feasible incumbent; feasibility counts are reported. ' if task['n_constraints'] else ''
    note = (f"Agent: one fresh run, {len(trace):,}/2,000 evaluations; GPT-6 Astra, max reasoning. "
            'Baselines: seed means (10 stochastic / 1 deterministic); 2,000-evaluation allowance.\n'
            + noisy_note + feasibility_note + 'Regret ≤ 10⁻¹² is shown at 10⁻¹²; early-stopped baseline incumbents are carried forward.')
    if annotation.get('note'):
        note += '\n' + annotation['note']

    # Convergence against every standalone method.
    fig, ax = plt.subplots(figsize=(12.5, 7))
    plot_lines(ax, baselines, agent)
    ax.set_title(title(task) + ' — optimization progress', loc='left', pad=16)
    fig.text(.08, -.01, '\n'.join(textwrap.wrap(note.replace('\n', ' '), 160)), fontsize=8, va='top', color='#526071')
    fig.tight_layout()
    savefig(fig, directory, '02_convergence')

    # Final regret with seed-level outcomes alongside the baseline means.
    values = {m: float(b['regret'][-1]) for m,b in baselines.items()}
    order = sorted(values, key=lambda m: values[m] if math.isfinite(values[m]) else math.inf)
    agent_bar_label = 'Astra max agent (execution failure)' if annotation.get('status') == 'execution_failure' else 'Astra max agent'
    entries = [(agent_bar_label, float(agent['regret'][-1]), AGENT, [])]
    for m in order:
        b = baselines[m]
        label = comparison.baseline_display_names[m] + f" (n={b['n_seeds']})"
        if task['n_constraints']:
            label += f"; feasible {b['n_feasible']}/{b['n_seeds']}"
        entries.append((label, values[m], comparison.baseline_style[m]['color'], final_seed_regrets(baseline_dir,m,task['f_opt'])))
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    finite = [max(FLOOR,v) for e in entries for v in [e[1],*e[3]] if math.isfinite(v)]
    low, high = min(finite or [FLOOR]), max(finite or [1.])
    ax.set_xscale('log'); ax.set_xlim(low/3, max(high*30,low*100))
    for index, (label,value,color,seeds) in enumerate(entries):
        y = len(entries)-1-index
        if math.isfinite(value):
            plotted = max(FLOOR,value)
            ax.barh(y, plotted, color=color, height=.55, alpha=.85)
            seed_values = [max(FLOOR,v) for v in seeds if math.isfinite(v)]
            offsets = np.linspace(-.15,.15,len(seed_values))
            ax.scatter(seed_values, y+offsets, s=19, facecolor='white', edgecolor='#324253', linewidth=.7, zorder=4)
            ax.text(1.015,y,f'{value:.3g}',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
        else:
            ax.text(low,y,'No feasible point',va='center',fontsize=9,color='#9c3a3a')
    ax.set_yticks(range(len(entries)), [e[0] for e in entries[::-1]])
    ax.set_xlabel('Final regret (log scale)'); clean_axes(ax); ax.grid(False,axis='y')
    ax.set_title(title(task) + ' — final regret',loc='left',pad=16)
    footer = 'Bars: agent result or baseline mean. White dots: individual baseline seeds. ' + note.replace('\n', ' ')
    fig.text(.08,-.01,'\n'.join(textwrap.wrap(footer,160)),fontsize=8,va='top',color='#526071')
    fig.tight_layout()
    savefig(fig,directory,'01_final_regret')

    # Combined convergence, decision boundaries and exact evaluation allocations.
    height = max(9.5, 2.6 + .49*len(blocks))
    fig = plt.figure(figsize=(18.5,height))
    gs = fig.add_gridspec(2,2,width_ratios=[3.2,1.65],height_ratios=[5,1],wspace=.21,hspace=.22)
    ax = fig.add_subplot(gs[0,0]); timeline = fig.add_subplot(gs[1,0],sharex=ax); listing = fig.add_subplot(gs[:,1]); listing.axis('off')
    plot_lines(ax,baselines,agent,legend=False)
    ax.set_xlabel(''); ax.tick_params(labelbottom=False)
    ax.legend(frameon=False,ncol=3,loc='lower left',bbox_to_anchor=(0,1.025),fontsize=8)
    listing.text(0,1,'Block decisions',fontsize=16,fontweight='bold',va='top')
    spacing = min(.085,.92/max(len(blocks),1))
    for i,block in enumerate(blocks):
        start,end = block['first_eval']-1,block['last_eval']
        color = timeline_color(block['method'])
        if start:
            ax.axvline(start,color='#8893a1',alpha=.4,linestyle='--',lw=.7,zorder=1)
            value = agent['regret'][start-1]
            if math.isfinite(value): ax.plot(start,max(value,FLOOR),'o',ms=4.5,color=AGENT,zorder=6)
        timeline.broken_barh([(start,end-start)],(.1,.8),facecolors=color,edgecolors='white',linewidth=1.2)
        if end-start >= 55:
            timeline.text((start+end)/2,.5,str(i+1),ha='center',va='center',color='white',fontweight='bold',fontsize=9)
        method = textwrap.fill(block['method'],width=40,break_long_words=True)
        listing.text(0,.94-i*spacing,f"{i+1}. {method}\n    {block['calls']} calls · evaluations {start+1}–{end}",
                     fontsize=10,va='top',color=AGENT,linespacing=1.5)
    timeline.set(ylim=(0,1),yticks=[],xlabel='Objective evaluations')
    for spine in ('top','right','left'): timeline.spines[spine].set_visible(False)
    fig.suptitle(title(task) + ' — what the agent chose, and when',x=.125,ha='left',fontsize=20,fontweight='bold',y=.98)
    timeline_note = 'Boundaries: evaluator snapshots of prospective decisions; method changes inside a block stay in that block.\n' + note
    timeline_note = '\n'.join('\n'.join(textwrap.wrap(line,200)) for line in timeline_note.splitlines())
    fig.text(.125,.025,timeline_note,
             fontsize=8,color='#526071',va='bottom')
    fig.subplots_adjust(top=.83,bottom=.13)
    savefig(fig,directory,'03_decision_timeline')

    final = float(agent['regret'][-1])
    result = {'task':key,'task_name':task['task_name'],'run_dir':str(run),'evaluations':len(trace),
              'status':annotation.get('status','complete' if len(trace)==2000 else 'completed_early'),
              'run_note':annotation.get('note',''),
              'agent_regret':final if math.isfinite(final) else None,
              'baseline_mean_regrets':{m:v if math.isfinite(v) else None for m,v in values.items()},
              'baseline_feasible_seed_counts':{m:[b['n_feasible'],b['n_seeds']] for m,b in baselines.items()},
              'decision_blocks':len(blocks),'noisy':task['noisy'],'n_constraints':task['n_constraints'],
              'source_commit':manifest['source_commit'],'model':manifest['model'],'effort':manifest['reasoning_effort']}
    temporary = done.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(done)
    print(key, result['status'], len(trace), 'evaluations, 3 figures exported as PNG/PDF/SVG',flush=True)
    return result

def index():
    state = json.loads((ROOT/'state.json').read_text())
    results = [json.loads(p.read_text()) for p in sorted((ROOT/'figures').glob('*/result.json'))]
    fields = ['task','status','evaluations','agent_regret','decision_blocks'] + list(comparison.baseline_style)
    with (ROOT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:
            w.writerow({**{k:r[k] for k in fields[:5]},**r['baseline_mean_regrets']})
    body=['<!doctype html><html><head><meta charset="utf-8"><title>Mathematical suite · Astra max · 2000 evaluations</title>',
          '<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;color:#172334}img{width:100%}section{border-top:1px solid #ccd3dc;margin:35px 0;padding-top:24px}table{border-collapse:collapse}td,th{padding:8px 16px;border-bottom:1px solid #ddd;text-align:left}a{color:#235cad}.muted{color:#657286}</style></head><body>',
          '<h1>Astra max · mathematical black-box suite</h1>',
          f'<p>{len(results)}/30 campaigns plotted. Budget: 2,000 evaluations per task. One agent run per task; all nine existing standalone baselines.</p>',
          '<p class="muted">Source 591e3de6. Noise-free scores stay private during optimization. Baselines use 10 stochastic seeds or 1 deterministic seed. Constrained agent receives objective and constraint values; baseline wrappers use a fixed penalty. These figures describe individual agent campaigns, not replicate-averaged agent performance.</p>',
          '<p><a href="summary.csv">Download numerical summary</a> · <a href="cohort.json">Protocol and provenance</a> · <a href="campaign_status.json">Live campaign status</a></p>',
          '<table><tr><th>Task</th><th>Status</th><th>Calls</th><th>Agent regret</th><th>Blocks</th></tr>']
    by_key={r['task']:r for r in results}
    normalized=[]
    for t in state['runs']:
        r=by_key.get(t['task'],{});v=r.get('agent_regret')
        normalized.append({'task':t['task'],'status':r.get('status',t['status']),
                           'evaluations':r.get('evaluations',t['evaluations']),'budget':2000,
                           'agent_regret':v,'figures_ready':bool(r)})
        value=f'{v:.5g}' if v is not None else '—'
        body.append(f"<tr><td><a href='#{html.escape(t['task'])}'>{html.escape(t['task'])}</a></td><td>{r.get('status',t['status'])}</td><td>{r.get('evaluations',t['evaluations'])}</td><td>{value}</td><td>{r.get('decision_blocks','—')}</td></tr>")
    body.append('</table>')
    for r in results:
        body.append(f"<section id='{r['task']}'><h2>{r['task']}</h2>")
        for stem in ['03_decision_timeline','02_convergence','01_final_regret']:
            base=f"figures/{r['task_name']}/{stem}"
            body.append(f"<p><a href='{base}.pdf'>PDF</a> · <a href='{base}.svg'>SVG</a> · <a href='{base}.png'>PNG</a></p><img loading='lazy' src='{base}.png' alt='{stem}'>")
        body.append('</section>')
    body.append('</body></html>')
    (ROOT/'index.html').write_text('\n'.join(body))
    (ROOT/'campaign_status.json').write_text(json.dumps({'updated_at':state['updated_at'],
        'done':state['done'],'paused':state['paused'],'campaigns':normalized},indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task');parser.add_argument('--force',action='store_true');args=parser.parse_args()
    state=json.loads((ROOT/'state.json').read_text())
    errors = []
    for task in state['runs']:
        if task.get('returncode')==0 and task.get('evaluations',0)>0 and (not args.task or args.task==task['task']):
            try:
                render(task,args.force)
            except Exception as exc:
                errors.append({'task': task['task'], 'error': repr(exc)})
                traceback.print_exc()
    if not args.task:
        (ROOT/'render_errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    index()
    if errors:
        raise SystemExit(1)
