#!/bin/bash
# Run run_sweep.py for several tasks at once on one whole CPU node.
#
# Submit from main/ in the $SCRATCH clone (jobs can only write to $SCRATCH):
#     sbatch -p debug scripts/slurm/sweep_all.sh                  # every task in TASKS
#     sbatch -p debug scripts/slurm/sweep_all.sh sphere10 rastrigin10
# Optional env: OUTDIR (default results), BUDGET, SEEDS (run_sweep defaults).
#
# The node's cores are split evenly across tasks. Each task logs to
# sweep_<task>_<jobid>.log; reruns skip completed campaigns.
#SBATCH --job-name=sweep_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

# One thread per worker process; the sweep's parallelism is across processes.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore

PY=../.venv/bin/python
if (( $# )); then
    tasks=("$@")
else
    read -ra tasks <<<"$("$PY" -c 'from scripts.run_campaign import TASKS; print(*sorted(TASKS))')"
fi
workers=$(( SLURM_CPUS_PER_TASK / ${#tasks[@]} ))
extra=()
[[ -n "${BUDGET:-}" ]] && extra+=(--budget "$BUDGET")
[[ -n "${SEEDS:-}" ]] && extra+=(--seeds "$SEEDS")

echo "tasks: ${tasks[*]}  workers per task: $workers  outdir: ${OUTDIR:-results}"
pids=()
for t in "${tasks[@]}"; do
    "$PY" scripts/run_sweep.py --task "$t" --workers "$workers" \
        --outdir "${OUTDIR:-results}" "${extra[@]}" > "sweep_${t}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "${tasks[$i]}: $(tail -1 "sweep_${tasks[$i]}_${SLURM_JOB_ID}.log")"
    else
        echo "${tasks[$i]}: FAILED (see sweep_${tasks[$i]}_${SLURM_JOB_ID}.log)"
        status=1
    fi
done
exit $status
