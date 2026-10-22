"""Trusted post-run audits and figures. Never imported by a campaign agent."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import textwrap

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p / 'main/src/tasks').is_dir())
sys.path.insert(0, str(REPO / 'main'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.tasks.mof_pool import MOFPoolTask
from src.tasks.opv_pool import OPVPoolTask

OUT = ROOT if (ROOT / 'results.json').exists() else ROOT / 'report'
COLORS = ['#177e89', '#d66b32', '#7556ab']
INK = '#172334'
FLOOR = 1e-8
TITLES = {'opv_pool': 'OPV device pool', 'mof_pool': 'MOF candidate pool'}
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


def decisions(snapshots, trace):
    """Separate prospective decisions, including continuations of a method."""
    n = len(trace)
    chunks = {}
    previous = ''
    initial = None
    for s in snapshots:
        idx, name, content = s['eval_idx'], s['name'], s['content']
        if name == 'roadmap_initial.md':
            assert idx == 0 and (initial is None or initial == content)
            initial = content
            chunks[0] = content
        elif name == 'changelog.md':
            assert content.startswith(previous), 'Decision log was not append-only'
            delta = content[len(previous):]
            previous = content
            if delta.strip() and idx < n:
                chunks[idx] = chunks.get(idx, '') + delta
        if idx < n:
            assert s['wall_ns'] <= int(trace[idx]['wall_ns']), 'Retrospective decision snapshot'
    assert initial and initial.strip()
    boundaries = sorted(set(chunks) | {0, n})
    blocks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segments = []
        for i in range(start, end):
            method = trace[i]['optimizer']
            if segments and segments[-1]['method'] == method:
                segments[-1]['last_eval'] = i + 1
            else:
                segments.append({'method': method, 'first_eval': i + 1, 'last_eval': i + 1})
        blocks.append({'block': len(blocks) + 1, 'first_eval': start + 1, 'last_eval': end,
                       'calls': end - start, 'method': ' → '.join(s['method'] for s in segments),
                       'method_segments': segments, 'decision_text': chunks.get(start, ''),
                       'boundary_source': 'prospective evaluator document snapshot'})
    return blocks


def audit_and_export(row):
    run = Path(row['run_dir'])
    private = run / 'private'
    public = run / 'workspace/results/agent_run'
    manifest = read_json(private / 'manifest.json')
    integrity = read_json(private / 'integrity_audit.json')
    scoring = read_json(private / 'scoring.json')
    assert row['returncode'] == 0
    assert manifest['budget'] == 100 and manifest['seed'] == row['seed']
    assert manifest['model'] == 'gpt-6-astra' and manifest['reasoning_effort'] == 'max'
    assert manifest['source_commit'] == read_json(ROOT / 'cohort.json')['source_commit']
    assert integrity['artifacts_match_journal'] and integrity['only_workspace_execution_tools']
    assert integrity['unknown_evaluations'] == 0
    for group in ('source_sha256', 'private_data_sha256'):
        for name, expected in manifest[group].items():
            assert digest(REPO / name) == expected, f'Changed {name}'
    for name in ('AGENTS.md', 'CLAUDE.md', 'PUBLIC_EVALUATOR.md', 'src/agent_harness/harness.py'):
        assert digest(run / 'workspace' / name) == manifest['public_sha256'][name]
    checks = read_json(private / 'isolation_checks.json')
    assert checks['evaluations'] == 0 and checks['budget'] == 100 and len(checks['checks']) == 28
    trace = read_csv(public / 'trace.csv')
    scored = read_csv(private / 'scored_trace.csv')
    with sqlite3.connect('file:' + str(private / 'evaluations.sqlite') + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        journal = [dict(x) for x in db.execute('SELECT * FROM evaluations ORDER BY idx')]
        snapshots = [dict(x) for x in db.execute('SELECT * FROM snapshots ORDER BY rowid')]
    assert len(trace) == len(journal) == len(scored) == integrity['evaluations_used']
    assert 0 < len(trace) <= 100
    task = MOFPoolTask() if row['task'] == 'mof_pool' else OPVPoolTask()
    assert task.f_opt == scoring['f_opt']
    if task.indexed_pool:
        matrix = run / 'workspace/candidates.npy'
        assert digest(matrix) == manifest['public_sha256']['candidates.npy']
        assert np.array_equal(np.load(matrix, allow_pickle=False), task.candidates)
    seen = set()
    augmented = []
    best = float('inf')
    for k, (t, j, s) in enumerate(zip(trace, journal, scored), 1):
        assert int(t['eval_idx']) == j['idx'] == int(s['eval_idx']) == k
        assert json.loads(t['x']) == json.loads(j['x'])
        assert float(t['y']) == j['y'] == float(s['y'])
        assert t['optimizer'] == j['label'] and int(t['wall_ns']) == j['wall_ns']
        if task.indexed_pool:
            index = int(t['candidate_index'])
            assert index == j['candidate_index']
            assert np.array_equal(json.loads(t['x']), task.candidates[index])
            observed = task.offline_observation(index)
        else:
            # Replay the private nearest-untested mapping, without making any
            # new evaluation request or modifying the campaign's budget journal.
            index = task.nearest_untested(np.asarray(json.loads(t['x'])))
            raw = float(task._pce[index])
            observed = {'candidate_id': int(task._ids[index]), 'raw_y': raw,
                        'descending_rank': 1 + int(np.count_nonzero(task._pce > raw))}
            task._tested[index] = True
        assert index not in seen
        seen.add(index)
        assert -observed['raw_y'] == j['y']
        best = min(best, j['y'])
        assert float(t['best_y']) == best == float(s['y_best_so_far'])
        augmented.append({**t, 'offline_candidate_id': observed['candidate_id'],
                          'raw_score': observed['raw_y'], 'best_raw_score': -best,
                          'regret': max(0., best - task.f_opt),
                          'offline_descending_rank': observed['descending_rank']})
    blocks = decisions(snapshots, trace)
    dest = OUT / 'runs' / row['task'] / f"seed_{row['seed']}"
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / 'scored_trace.csv', augmented)
    write_json(dest / 'decisions.json', blocks)
    write_json(dest / 'decision_snapshots.json', snapshots)
    for name in ('trace.csv', 'meta.json', 'roadmap_initial.md', 'changelog.md'):
        shutil.copyfile(public / name, dest / name)
    for name in ('manifest.json', 'integrity_audit.json', 'isolation_checks.json', 'initial_prompt.txt',
                 'scoring.json', 'effective_overrides.json'):
        shutil.copyfile(private / name, dest / name)
    # Preserve the actual agent-created scripts and any modified wrappers.
    for part in ('work', 'src/optimizers'):
        for source in (run / 'workspace' / part).rglob('*.py'):
            target = dest / 'agent_code' / source.relative_to(run / 'workspace')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    events = [json.loads(line) for line in (private / 'cli_events.jsonl').read_text().splitlines() if line.strip()]
    messages = [e['item']['text'] for e in events if e.get('type') == 'item.completed'
                and e.get('item', {}).get('type') == 'agent_message']
    (dest / 'agent_messages.md').write_text('\n\n---\n\n'.join(messages) + '\n')
    # Full event stream is experiment-only: fresh context, no inherited user messages.
    shutil.copyfile(private / 'cli_events.jsonl', dest / 'cli_events.jsonl')
    # A portable, closed-journal evidence copy, without request/response cache noise.
    write_json(dest / 'evaluation_journal.json', journal)
    audit = {'status': 'passed', 'evaluations': len(trace), 'unique_candidates': len(seen),
             'private_label_replay_exact': True, 'trace_matches_journal': True,
             'scored_trace_verified': True, 'source_and_dataset_hashes_verified': True,
             'public_immutable_files_verified': True, 'prospective_decisions_verified': True,
             'isolation_checks': len(checks['checks'])}
    write_json(dest / 'offline_audit.json', audit)
    incumbent = min(augmented, key=lambda t: float(t['y']))
    summary = {'task': row['task'], 'seed': row['seed'], 'status': row['status'],
               'evaluations': len(trace), 'budget': 100, 'best_raw_score': -best,
               'pool_maximum': -task.f_opt, 'regret': max(0., best - task.f_opt),
               'best_candidate_id': incumbent['offline_candidate_id'],
               'best_descending_rank': incumbent['offline_descending_rank'],
               'first_best_evaluation': int(incumbent['eval_idx']), 'decision_blocks': len(blocks),
               'run_id': run.name, 'source_commit': manifest['source_commit']}
    write_json(dest / 'result.json', summary)
    return {'summary': summary, 'trace': augmented, 'blocks': blocks, 'dest': dest}


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


def display_method(block):
    # Alternate labels may repeat many times inside one prospective block.
    # Keep their exact order in JSON; name each distinct method once in the figure.
    return ' + '.join(dict.fromkeys(s['method'] for s in block['method_segments']))


def timeline_axes(fig, spec, data, color, all_runs=None):
    n = data['summary']['evaluations']
    grid = spec.subgridspec(2, 2, width_ratios=[3.3, 2], height_ratios=[4, .65], wspace=.25, hspace=.1)
    ax = fig.add_subplot(grid[0, 0])
    bar = fig.add_subplot(grid[1, 0], sharex=ax)
    listing = fig.add_subplot(grid[:, 1]); listing.axis('off')
    if all_runs:
        for other in all_runs:
            if other is data: continue
            ax.step(np.arange(1,len(other['trace'])+1), np.maximum([t['regret'] for t in other['trace']], FLOOR),
                    where='post', color='#aab2bd', lw=1, alpha=.55)
    regrets = np.array([t['regret'] for t in data['trace']])
    ax.step(np.arange(1,n+1), np.maximum(regrets,FLOOR), where='post', color=color, lw=2.5)
    ax.set(yscale='log', xlim=(0,100), ylabel='Incumbent regret',
           title=f"Seed {data['summary']['seed']} · {n}/100 evaluations · {len(data['blocks'])} decisions")
    ax.tick_params(labelbottom=False)
    axes_style(ax)
    paragraphs = []
    for b in data['blocks']:
        start, end = b['first_eval']-1, b['last_eval']
        if start:
            ax.axvline(start, color='#99a2af', lw=.7, ls='--', alpha=.6)
            ax.plot(start,max(FLOOR,regrets[start-1]),'o',color=color,ms=4)
        bar.broken_barh([(start,end-start)],(.05,.9),facecolors=method_color(b['method']),
                        edgecolors='white',linewidth=1.5)
        if end-start>=4:
            bar.text((start+end)/2,.5,str(b['block']),ha='center',va='center',color='white',fontsize=10,fontweight='bold')
        name = textwrap.fill(shorten(display_method(b)), width=47)
        paragraphs.append(f"{b['block']}. {name}\n    {b['calls']} calls · evaluations {start+1}–{end}")
    listing.text(0,1,'Block decisions',va='top',fontsize=13,fontweight='bold',color=INK)
    # Space is proportional to wrapped lines so longer method names do not overlap.
    line_counts = [p.count('\n')+1 for p in paragraphs]
    total = sum(line_counts) + .7*len(paragraphs)
    y = .94
    for para, count in zip(paragraphs,line_counts):
        listing.text(0,y,para,va='top',fontsize=9.3,color=INK,linespacing=1.35)
        y -= .90*(count+.7)/max(total,1)
    bar.set(ylim=(0,1),yticks=[],xlabel='Objective evaluations')
    for side in ('top','right','left'): bar.spines[side].set_visible(False)


def render(task, data):
    directory = OUT / 'figures' / task
    title = TITLES[task]
    note = ('GPT-6 Astra · max reasoning · 100 evaluations per seed, including initialization. '
            'Seeds 0, 1, 2 are independent campaigns; model generation is not deterministic.\n'
            'Agent seed traces; standalone baselines appear in the complete comparison figure. Regret = pool maximum − best measured score.')
    values = [d['summary']['regret'] for d in data]
    fig, ax = plt.subplots(figsize=(10,5.5))
    ax.bar(range(3), values, color=COLORS, width=.5)
    top = max(max(values)*1.25, .1)
    for i, d in enumerate(data):
        r=d['summary']
        ax.text(i, values[i]+top*.025, f"{values[i]:.6g}\nPool rank {r['best_descending_rank']}",
                ha='center',va='bottom',fontsize=12,color=INK)
    ax.set(xticks=range(3),xticklabels=[f'Seed {i}' for i in range(3)],ylim=(0,top),
           ylabel='Final regret (lower is better)',title=title+' — final outcomes')
    axes_style(ax);ax.grid(False,axis='x')
    fig.text(.1,.015,note,fontsize=8,color='#526071',va='bottom')
    fig.subplots_adjust(bottom=.2,top=.9)
    save(fig,directory,'01_final_regret')

    fig, axs = plt.subplots(1,2,figsize=(14,5.8))
    for d,color in zip(data,COLORS):
        x=np.arange(1,len(d['trace'])+1)
        axs[0].step(x,[t['best_raw_score'] for t in d['trace']],where='post',color=color,lw=2,label=f"Seed {d['summary']['seed']}")
        axs[1].step(x,np.maximum([t['regret'] for t in d['trace']],FLOOR),where='post',color=color,lw=2)
    maximum=data[0]['summary']['pool_maximum']
    axs[0].axhline(maximum,color=INK,lw=1,ls='--',label=f'Pool maximum ({maximum:.6g})')
    axs[0].set(ylabel='Best measured PCE (%)' if task=='opv_pool' else 'Best observed raw score',title='Best measured score')
    axs[0].legend(frameon=False,fontsize=9,loc='lower right')
    axs[1].set(yscale='log',ylabel='Incumbent regret (log scale)',title='Gap to the pool maximum')
    for ax in axs:
        ax.set(xlim=(0,100),xlabel='Objective evaluations');axes_style(ax)
    fig.suptitle(title+' — optimization progress',fontsize=19,fontweight='bold',y=.98)
    fig.text(.08,.015,note+f'\nZero regret is drawn at {FLOOR:g}; all three seed traces are shown without averaging.',fontsize=8,color='#526071',va='bottom')
    fig.subplots_adjust(top=.85,bottom=.23,wspace=.24)
    save(fig,directory,'02_convergence')

    heights=[max(5.2,1.3+.19*sum(len(textwrap.wrap(shorten(display_method(b)),width=47))+1.7
                                for b in d['blocks'])) for d in data]
    fig=plt.figure(figsize=(17,sum(heights)+1.2))
    grid=fig.add_gridspec(3,1,height_ratios=heights,hspace=.24)
    for i,(d,color) in enumerate(zip(data,COLORS)):
        timeline_axes(fig,grid[i],d,color)
        seedfig=plt.figure(figsize=(17,heights[i]+2))
        seedgrid=seedfig.add_gridspec(1,1)
        timeline_axes(seedfig,seedgrid[0],d,color,all_runs=data)
        seedfig.suptitle(f"{title} — seed {i}: decisions and progress",fontsize=20,fontweight='bold',y=.98)
        seedfig.text(.08,.015,note+f'\nBoundaries come from prospective decision snapshots; gray curves are the other two agent seeds. Zero regret is drawn at {FLOOR:g}.',fontsize=8,color='#526071',va='bottom')
        seedfig.subplots_adjust(top=.85,bottom=.21)
        save(seedfig,directory,f'seed_{i}_decision_timeline')
    fig.suptitle(title+' — what each agent chose, and when',fontsize=21,fontweight='bold',y=.99)
    fig.text(.08,.01,note+f'\nBoundaries come from prospective decision snapshots, including continuations. Zero regret is drawn at {FLOOR:g}.',fontsize=8,color='#526071',va='bottom')
    fig.subplots_adjust(top=.95,bottom=.08)
    save(fig,directory,'03_decision_timeline')


def write_readme(results):
    cohort=read_json(ROOT/'cohort.json')
    lines=[
        '# Materials pool campaigns: Astra max, 100 evaluations, three seeds',
        '',
        'Each campaign uses the current autonomous agent in a fresh isolated Codex CLI, '
        'with its own private evaluator and durable 100-call budget. Initialization is included. '
        'Seeds 0, 1, and 2 guide stochastic optimization choices; model generation itself is not deterministic.',
        '',
        f"Source commit: `{cohort['source_commit']}`. Model: `gpt-6-astra`; reasoning: `max`. "
        'No optimizer hints were supplied by the supervisor. One MOF seed-1 attempt was interrupted '
        'by a model-service capacity error at 58 calls. At the user’s request it was discarded from '
        'the comparison and restarted in a completely fresh CLI with a fresh 100-call budget; '
        'none of its observations were passed to the replacement. The completed replacement is the reported seed 1.',
        '',
        'Baselines are the colleague-supplied OPV pool results from branch commit '
        '`ef1be572e038487dc37a5bd098e5d5d25fab8fb3` and MOF `results_linux.zip`. '
        'All supplied baseline rows were checked against the registered private pools, including '
        'candidate identity, features (MOF), nearest-untested mapping (OPV), labels and incumbents. '
        'The old OPV emulator baselines and short integration smoke tests are not used.',
        '',
        '**Equal 100-call budget:** use calls 1–100 of every run. For MOF, discard the final ten '
        'calls of the supplied 110-call baseline traces. Their ten random initialization calls '
        'remain included in the first 100. The agent chooses its own initialization within the same total allowance.',
        '',
        'Curves are arithmetic means of seed-level incumbent regret. Bands and error bars are '
        'sample standard deviations (`ddof=1`), not confidence intervals. The agent has three '
        'seeds per task; stochastic baselines have ten. OPV BOBYQA, L-BFGS-B and Nelder–Mead '
        'have one supplied seed each, so no variability band is estimated for them. Negative '
        'mean-minus-SD endpoints are clipped at zero for display. OPV differential-evolution '
        'and uniform-box-random traces coincide throughout this 100-call window; both are '
        'retained and labelled as supplied.',
        '',
        'These detailed campaign reports follow the user’s requested arithmetic mean ± standard '
        'deviation and include individual optimizers plus decision timelines. The separate '
        '`main/FIGURES.md` paper style uses geometric means ± log-space standard errors; '
        'that shared helper and convention are unchanged. Do not confuse the two summaries.',
        '',
        '## Task contracts',
        '',
        '| Task | Pool | Public information | Private selection/scoring |',
        '|---|---|---|---|',
        '| OPV | 526 measured devices, 11 features | Anonymous [0,1]^11 box; disclosed history dependence | '
        'Each query chooses the nearest untested device; response is −PCE. Full features and labels stay private. |',
        '| MOF | 1,000 rows, 193 features (28 constant) | Anonymous feature rows; select an unused integer row index | '
        'Response is −raw_y. Unpurchased labels, source IDs and full-pool ranks stay private. |',
        '',
        'The MOF archive described 5,882 rows, but the supplied CSV contains 1,000. The registered CSV is '
        'byte-for-byte unchanged; no rows were removed or reordered. No physical meaning or units for raw_y '
        'were supplied. OPV and MOF have different public interfaces; compare each only with baselines using '
        'that task’s same interface, dataset hash, row order, objective sign and total budget.',
        '',
        'Reported regret is `full_pool_maximum − best_observed_raw_score`. The full-pool maximum, '
        'material/device IDs and ranks are added by trusted reporting after the agent exits. '
        'Rank 1 means no pool row has a strictly higher label; ties share rank. For OPV, the best result '
        'identifies the purchased device, not a promise that querying its original x again returns the same label.',
        '',
        '## Results',
        '',
        '| Task | Seed | Calls | Best raw score | Regret | Pool rank | First best call | Decision blocks |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in results:
        lines.append(f"| {r['task']} | {r['seed']} | {r['evaluations']} | {r['best_raw_score']:.10g} | "
                     f"{r['regret']:.10g} | {r['best_descending_rank']} | {r['first_best_evaluation']} | {r['decision_blocks']} |")
    lines+=['','Three seeds describe run-to-run variation; these results alone do not establish statistical '
            'superiority over standalone methods.','', '## Figures','']
    for task in TITLES:
        if not any(r['task']==task for r in results): continue
        lines += [f"### {TITLES[task]}", '',
                  f"![Complete comparison](figures/{task}/00_complete_comparison.png)", '',
                  f"[Log-scale comparison](figures/{task}/00_complete_comparison_log.png) · "
                  f"[Individual seed curves](figures/{task}/02_convergence.png)", '',
                  f"[Final regret](figures/{task}/01_final_regret.png) · "
                  f"[All decision timelines](figures/{task}/03_decision_timeline.png)", '',
                  'Individual timelines: '+ ' · '.join(f"[Seed {i}](figures/{task}/seed_{i}_decision_timeline.png)" for i in range(3)), '']
    lines += [
        'Every figure also has PDF and SVG versions. Zero regret is drawn at 1e-8 on log axes and is '
        'recorded as exactly zero in the tables. Decision boundaries come from evaluator snapshots '
        'captured before the subsequent evaluations, including continuations of a method. '
        'Different labels within a single planned block remain in that block.',
        '', '## Evidence and reproduction', '',
        '- `summary.csv` and `results.json`: seed-level outcomes.',
        '- `cohort.json`: model, seed, budget, task contracts, source commit and dataset hashes.',
        '- `runs/<task>/seed_<n>/`: original agent trace, scored trace, decisions and snapshots, '
        'agent messages, full fresh-context CLI event stream, generated Python code, run manifest and audits.',
        '- `evaluation_journal.json`: exported private evaluation rows; `offline_audit.json`: '
        'exact label replay, candidate uniqueness, trace consistency, prospective decision timing and hash checks.',
        '- All six runs must pass their own 28 native isolation checks before any model evaluation. '
        'Final audits check that only sandboxed workspace execution tools were used and that there '
        'are no evaluations with an unknown outcome. These are tested controls, not a formal guarantee against every OS or runtime vulnerability.',
        '- `sha256.json`: hashes of the exported evidence and figures.',
        '',
        'To launch a new campaign from the repository root after the documented environment setup:',
        '', '```bash',
        '.venv/bin/python main/scripts/run_agent.py --task mof_pool --budget 100 --seed 0 \\',
        '  --model gpt-6-astra --effort max',
        '```', '',
        'Use `opv_pool` for OPV and seeds 0, 1, 2 for three independent campaigns. '
        'Never reuse a prior campaign workspace or show an agent this reporting directory. '
        'The native runner enforces that separation.',
        '',
        'To regenerate the figures from this exported package, run `report.py --render-only` '
        'with the repository’s `.agent-venv/bin/python`, then run `comparison_report.py` for the '
        'mean/SD baseline comparison figures. `baselines/provenance.json` records supplied sources '
        'and the budget truncation rule. Full raw SQLite databases and the frozen '
        'source archive are retained in the local ignored cohort directory.',
        '',
    ]
    (OUT/'README.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task', choices=list(TITLES))
    parser.add_argument('--render-only', action='store_true', help='Render the exported, already audited traces')
    args=parser.parse_args()
    if args.render_only:
        for task in ([args.task] if args.task else TITLES):
            data=[]
            for seed in range(3):
                dest=OUT/'runs'/task/f'seed_{seed}'
                if not (dest/'result.json').exists(): break
                trace=read_csv(dest/'scored_trace.csv')
                for row in trace:
                    for key in ('raw_score','best_raw_score','regret'): row[key]=float(row[key])
                data.append({'summary':read_json(dest/'result.json'), 'trace':trace,
                             'blocks':read_json(dest/'decisions.json'), 'dest':dest})
            if len(data)==3:
                render(task,data)
                print(task+': rendered exported traces',flush=True)
        return
    state=read_json(ROOT/'state.json')
    replacement_path=ROOT/'replacement_state.json'
    if replacement_path.exists():
        replacement=read_json(replacement_path)
        state['runs']=[replacement['replacement'] if r['key']==replacement['replacement']['key'] else r
                       for r in state['runs']]
    OUT.mkdir(parents=True,exist_ok=True)
    complete=[]
    for task in ([args.task] if args.task else TITLES):
        selected=sorted([r for r in state['runs'] if r['task']==task],key=lambda r:r['seed'])
        if len(selected)!=3 or any(r['status'] not in ('complete','completed_early') for r in selected):
            print(task+': waiting for all three completed campaigns',flush=True)
            continue
        data=[audit_and_export(r) for r in selected]
        render(task,data)
        complete.extend(d['summary'] for d in data)
        print(task+': audited and rendered all three seeds',flush=True)
    if complete:
        all_results=[read_json(p) for p in sorted((OUT/'runs').glob('*/seed_*/result.json'))]
        write_csv(OUT/'summary.csv',all_results)
        write_json(OUT/'results.json',all_results)
        shutil.copyfile(ROOT/'cohort.json',OUT/'cohort.json')
        if replacement_path.exists():
            excluded=replacement['excluded_attempt']
            write_json(OUT/'excluded_attempt.json', {
                'task':excluded['task'], 'seed':excluded['seed'],
                'evaluations':excluded['evaluations'], 'run_id':Path(excluded['run_dir']).name,
                'reason':replacement['reason'], 'included_in_statistics':False,
                'replacement_run_id':Path(replacement['replacement']['run_dir']).name})
        shutil.copyfile(Path(__file__),OUT/'report.py')
        write_readme(all_results)
        write_json(OUT/'sha256.json',{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*'))
                                    if p.is_file() and p.name!='sha256.json'})


if __name__=='__main__': main()
