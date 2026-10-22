"""Mean ± sample SD with 10-seed random-start local baselines and unchanged stochastic/agent runs."""
import argparse
import json
from pathlib import Path
import shutil
import textwrap
import plot_support as r
import numpy as np
import matplotlib.pyplot as plt

NAMES={'agent':'Astra max agent','bobyqa':'BOBYQA','cma_es':'CMA-ES',
       'de':'Differential evolution','gp_bo':'GP Bayesian optimization','lbfgsb':'L-BFGS-B',
       'nelder_mead':'Nelder–Mead','pool_random':'Uniform pool sampling',
       'random_search':'Uniform box sampling','sobol':'Sobol','turbo':'TuRBO'}
STYLE={'agent':('#172334','-'),'bobyqa':('#af7a37','-'),'cma_es':('#247bb2','-'),
       'de':('#bd665e','--'),'gp_bo':('#da8332','-'),'lbfgsb':('#b56ca0',':'),
       'nelder_mead':('#716f68','--'),'pool_random':('#66974d',':'),
       'random_search':('#708596','-'),'sobol':('#339789','--'),'turbo':('#8e5eb3','-')}


for base, label in [('bobyqa', 'BOBYQA'), ('lbfgsb', 'L-BFGS-B'), ('nelder_mead', 'Nelder–Mead')]:
    NAMES[base + '_random_start'] = label + ' · random start'
    STYLE[base + '_random_start'] = STYLE[base]

def family(label):
    if ' → ' in label:
        return ' + '.join(dict.fromkeys(family(part) for part in label.split(' → ')))
    s=label.lower()
    if 'sobol' in s:return 'Sobol + probes' if any(x in s for x in ('anchor','reference','probe')) else 'Sobol'
    if 'cma' in s:return 'CMA-ES'
    if 'turbo' in s:return 'TuRBO'
    if 'gp' in s and any(x in s for x in ('tree','forest')):return 'GP + trees'
    if 'gp' in s or 'bayes' in s or 'matern' in s:return 'GP-BO'
    if 'cluster' in s or 'medoid' in s:return 'Cluster design'
    if 'repeat' in s or 'probe' in s or 'paired' in s or 'control' in s:return 'Probes'
    if 'revisit' in s:return 'Revisit'
    if 'tree' in s or 'forest' in s:return 'Tree model'
    if 'random' in s:return 'Random'
    if 'balanced' in s or 'coverage' in s or 'diverse' in s:return 'Diverse design'
    if 'neighbor' in s or 'nearest' in s or 'local' in s:return 'Local check'
    return r.shorten(label)[:24]


def load(task):
    data=[]
    for seed in range(3):
        d=r.OUT/'runs'/task/f'seed_{seed}'
        if not (d/'result.json').exists():return None
        result=r.read_json(d/'result.json')
        assert result['evaluations']==100 and result['status']=='complete'
        trace=r.read_csv(d/'scored_trace.csv')
        data.append({'summary':result,'trace':trace,'blocks':r.read_json(d/'decisions.json')})
    groups={'agent':np.array([[float(t['regret']) for t in d['trace']] for d in data])}
    folder=r.OUT/'baselines'/('opv/opv_pool_pce_d11' if task=='opv_pool' else 'mof/results_linux/task_061')
    maximum=data[0]['summary']['pool_maximum']
    for method in sorted(folder.iterdir()):
        if not method.is_dir():continue
        curves=[]
        for trace in sorted(method.glob('seed_*/trace.csv')):
            rows=r.read_csv(trace)
            # Count every paid call, including all initialization. Discard 101–110 for MOF.
            assert len(rows)>=100
            raw=np.array([float(t['y']) for t in rows[:100]])
            if task=='opv_pool':raw=-raw
            curves.append(np.maximum(0.,maximum-np.maximum.accumulate(raw)))
        groups[method.name]=np.array(curves)
    stats={key:{'n':len(a),'mean':a.mean(axis=0),'std':a.std(axis=0,ddof=1) if len(a)>1 else None}
           for key,a in groups.items()}
    return data,groups,stats


def timeline(ax,listing,data):
    listing.axis('off')
    paragraphs=[]
    for seed,d in enumerate(data):
        y=2-seed
        parts=[]
        for b in d['blocks']:
            start,end=b['first_eval']-1,b['last_eval']
            color='#617d88' if ' → ' in b['method'] else r.method_color(b['method'])
            ax.broken_barh([(start,end-start)],(y-.28,.56),facecolors=color,
                           edgecolors='white',linewidth=1.3)
            if end-start>=3:
                ax.text((start+end)/2,y,str(b['block']),ha='center',va='center',fontsize=9,fontweight='bold',color='white')
            parts.append(f"{b['block']}. {family(b['method'])} ({b['calls']})")
        paragraphs.append('Seed '+str(seed)+'\n'+textwrap.fill(' · '.join(parts),width=67))
    ax.set(xlim=(0,100),ylim=(-.6,2.6),yticks=[2,1,0],yticklabels=['Seed 0','Seed 1','Seed 2'],xlabel='Objective evaluations')
    ax.set_title('Agent decisions · block widths show calls used',loc='left',fontsize=13,pad=15)
    for side in ('top','right','left'):ax.spines[side].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=8)
    total=sum(p.count('\n')+1 for p in paragraphs)+2
    yy=1
    for p in paragraphs:
        listing.text(0,yy,p,va='top',fontsize=9.6,linespacing=1.4,color=r.INK)
        yy-=(p.count('\n')+2)/total


def draw(task,data,groups,stats,log=False):
    listing_lines=sum(1+len(textwrap.wrap(' · '.join(
        f"{b['block']}. {family(b['method'])} ({b['calls']})" for b in d['blocks']),width=67))
        for d in data)+2
    fig=plt.figure(figsize=(21,max(11.5,8+.20*listing_lines)))
    gs=fig.add_gridspec(2,2,width_ratios=[3.3,2],height_ratios=[4.6,max(2.25,.19*listing_lines)],wspace=.52,hspace=.38)
    ax=fig.add_subplot(gs[0,0]);final=fig.add_subplot(gs[0,1])
    x=np.arange(1,101)
    order=[k for k in stats if k!='agent']+['agent']
    for method in order:
        s=stats[method];color,style=STYLE[method]
        floor=r.FLOOR if log else 0.
        if s['std'] is not None:
            lo=np.maximum(floor,s['mean']-s['std']);hi=np.maximum(floor,s['mean']+s['std'])
            ax.fill_between(x,lo,hi,step='post',color=color,alpha=.16 if method=='agent' else .055,zorder=3 if method=='agent' else 1)
        ax.step(x,np.maximum(floor,s['mean']),where='post',color=color,ls=style,
                lw=3.1 if method=='agent' else 1.55,zorder=6 if method=='agent' else 2,
                label=f"{NAMES[method]} (n={s['n']})")
    ax.set(xlim=(0,100),xlabel='Objective evaluations (initialization included)',ylabel='Mean incumbent regret',
           title='Optimization progress · mean ± 1 standard deviation')
    if log:ax.set_yscale('log')
    else:ax.set_ylim(bottom=0)
    r.axes_style(ax)
    handles,labels=ax.get_legend_handles_labels()
    # Put the three-seed agent first without obscuring baseline lines.
    ax.legend([handles[-1],*handles[:-1]],[labels[-1],*labels[:-1]],frameon=False,
              fontsize=8.7,ncol=2 if len(stats)>4 else 1,loc='upper right')
    ranked=['agent']+sorted((k for k in stats if k!='agent'),key=lambda k:stats[k]['mean'][-1])
    maximum=0.
    for i,method in enumerate(ranked):
        s=stats[method];m=float(s['mean'][-1]);sd=float(s['std'][-1]) if s['std'] is not None else 0.
        y=len(ranked)-1-i;color,_=STYLE[method]
        if s['std'] is not None:
            final.errorbar(m,y,xerr=np.array([[min(m,sd)],[sd]]),fmt='o',color=color,
                           ms=8 if method=='agent' else 6,capsize=3,lw=1.5)
        else:final.plot(m,y,'o',color=color,ms=6)
        final.text(1.03,y,f'{m:.3g}'+(f' ± {sd:.3g}' if s['std'] is not None else ''),
                   transform=final.get_yaxis_transform(),ha='left',va='center',fontsize=9,
                   fontweight='bold' if method=='agent' else 'normal',color=r.INK)
        maximum=max(maximum,m+sd)
    final.set(yticks=range(len(ranked)),yticklabels=[f"{NAMES[k]} (n={stats[k]['n']})" for k in ranked[::-1]],
              xlabel='Regret at evaluation 100',title='Final performance · mean ± SD',xlim=(-.04*max(maximum,.1),max(.1,maximum)*1.15))
    final.tick_params(axis='y',labelsize=9)
    r.axes_style(final);final.grid(False,axis='y')
    timeline(fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),data)
    fig.suptitle(r.TITLES[task]+' — autonomous agent versus standalone optimizers',fontsize=22,fontweight='bold',x=.065,ha='left',y=.985)
    pool_info='526 devices · 11 features' if task=='opv_pool' else '1,000 candidates · 193 features'
    fig.text(.065,.946,pool_info+' · GPT-6 Astra, max reasoning · 3 agent seeds · 100 total evaluations per run',fontsize=12,color='#526071')
    notes='Curves and markers: arithmetic means. Shading/error bars: sample standard deviation (ddof=1), not confidence intervals; lower limits clipped at zero.\n'
    notes+='Agent n=3; every baseline n=10. BOBYQA, L-BFGS-B and Nelder–Mead use shared seeded uniform starts; original midpoint runs are excluded. ' if task=='opv_pool' else 'Agent n=3; each baseline n=10. MOF baselines use calls 1–100 of the supplied 110-call traces, including all ten initialization calls. '
    notes+='Regret = full-pool maximum − best observed score.\n'
    notes+='Decision boundaries are prospective evaluator snapshots; parentheses give block call counts. Exact labels, settings and rationale are preserved in the per-seed reports.'
    if log:notes+=f'\nLog display only: zero values and clipped band endpoints are drawn at {r.FLOOR:g}.'
    fig.text(.065,.015,notes,fontsize=8.5,color='#526071',va='bottom')
    fig.subplots_adjust(top=.875,bottom=.15,left=.065,right=.89)
    r.save(fig,r.OUT/'figures'/task,'00_complete_comparison'+('_log' if log else ''))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=list(r.TITLES));args=parser.parse_args()
    if r.ROOT!=r.OUT:
        shutil.copytree(r.ROOT/'baselines',r.OUT/'baselines',dirs_exist_ok=True)
        shutil.copyfile(__file__,r.OUT/'comparison_report.py')
        shutil.copyfile(r.ROOT/'baseline_audit.py',r.OUT/'baseline_audit.py')
        for name in ('agent-requirements.txt','evaluator-requirements.txt'):
            shutil.copyfile(r.ROOT/name,r.OUT/name)
    summaries=[]
    for task in ([args.task] if args.task else r.TITLES):
        loaded=load(task)
        if loaded is None:
            print(task+': waiting for three audited agent results',flush=True);continue
        data,groups,stats=loaded
        for log in (False,True):draw(task,data,groups,stats,log=log)
        rows=[]
        for method,s in stats.items():
            for i in range(100):
                rows.append({'task':task,'method':method,'eval_idx':i+1,'seeds':s['n'],
                             'mean_regret':float(s['mean'][i]),'sample_std':float(s['std'][i]) if s['std'] is not None else ''})
            summaries.append({'task':task,'method':method,'seeds':s['n'],
                              'mean_regret':float(s['mean'][-1]),'sample_std':float(s['std'][-1]) if s['std'] is not None else None})
        r.write_csv(r.OUT/'figures'/task/'comparison_curves.csv',rows)
        r.write_json(r.OUT/'figures'/task/'comparison_statistics.json',summaries[-len(stats):])
        print(task+': complete comparison exported',flush=True)
    if summaries:
        all_summaries=[]
        for path in sorted((r.OUT/'figures').glob('*/comparison_statistics.json')):
            all_summaries.extend(r.read_json(path))
        r.write_json(r.OUT/'comparison_summary.json',all_summaries)
        r.write_json(r.OUT/'sha256.json',{str(p.relative_to(r.OUT)):r.digest(p)
                                        for p in sorted(r.OUT.rglob('*'))
                                        if p.is_file() and p.name!='sha256.json'})


if __name__=='__main__':main()
