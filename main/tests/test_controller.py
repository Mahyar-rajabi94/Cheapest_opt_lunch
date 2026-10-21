import json

import numpy as np
import pytest

from agent.controller import Controller
from agent.dev_tasks import get_task
from agent.intake import Intake
from agent.policies.scripted import ScriptedPolicy
from agent.schema import Action


def make(task="dev_f1_d5", budget=300, policy=None, **kw):
    t = get_task(task)
    intake = Intake(dim=t.dim, bounds=t.bounds.tolist(), budget_evals=budget,
                    eval_cost="cheap", noise="deterministic")
    return t, Controller(t, intake, policy or ScriptedPolicy(), seed=0, **kw)


class Scripted:
    """Replays a fixed list of actions, then 'continue'."""
    name = "replay"

    def __init__(self, actions):
        self.actions = list(actions)
        self.reports = []

    def decide(self, r):
        self.reports.append(r)
        return self.actions.pop(0) if self.actions else {"kind": "continue"}


@pytest.mark.parametrize("task", ["dev_f1_d5", "dev_f15_d5"])
def test_scripted_run_respects_budget(task, tmp_path):
    t, ctl = make(task, budget=300)
    res = ctl.run()
    assert ctl.ev.used <= 300
    assert res["best_y"] == pytest.approx(float(ctl.archive.y.min()))
    ctl.save(tmp_path)
    for f in ["trace.csv", "meta.json", "decisions.jsonl", "intake.yaml"]:
        assert (tmp_path / f).exists()
    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert len(lines) == ctl.round
    json.loads(lines[0])


def test_invalid_actions_fall_back_and_are_logged():
    pol = Scripted([{"kind": "switch"},                       # schema error: no method
                    {"kind": "switch", "method": "nope"},     # unknown method
                    {"kind": "continue"}])
    _, ctl = make(budget=200, policy=pol)
    ctl.run()
    srcs = [e["source"] for e in ctl.log.entries[:2]]
    assert all(s.startswith("fallback") for s in srcs)
    assert ctl.ev.used <= 200


def test_report_never_contains_task_identity():
    pol = Scripted([{"kind": "probe"}, {"kind": "switch", "method": "cma_es"}])
    t, ctl = make("dev_f8_d5", budget=200, policy=pol)
    ctl.run()
    blob = json.dumps(pol.reports, default=str).lower()
    for leak in ["f_opt", "dev_f", "bbob", "rosenbrock", "task001", str(round(t.f_opt, 2))]:
        assert leak not in blob


def test_probes_stay_within_budget_and_return_features():
    pol = Scripted([{"kind": "probe", "budget": 120}])
    _, ctl = make("dev_f8_d5", budget=400, policy=pol)
    ctl.max_rounds = 1
    ctl.run()
    f = ctl.probe_features
    assert f["probe_budget_used"] <= 120
    assert {"P1", "P2", "P3", "P4", "P5"} <= set(f)
    assert -1 <= f["P2"]["fdc_spearman"] <= 1


def test_smooth_problem_is_classified_single_basin():
    from agent.policies.scripted import classify
    pol = Scripted([{"kind": "probe"}])
    _, ctl = make("dev_f1_d5", budget=400, policy=pol)
    ctl.max_rounds = 1
    ctl.run()
    assert classify(ctl.probe_features) == "smooth_single_basin"


def test_race_activates_winner_within_budget():
    pol = Scripted([{"kind": "switch", "method": "sobol", "budget": 40},
                    {"kind": "race", "candidates": ["cma_es", "lbfgsb", "random_search"],
                     "budget": 150}])
    _, ctl = make("dev_f1_d5", budget=400, policy=pol)
    ctl.max_rounds = 2
    ctl.run()
    race = ctl.last_race
    assert race["winner"] in ("cma_es", "lbfgsb")
    assert ctl.active["method"] == race["winner"]
    raced = sum(1 for t in ctl.archive.tags if t.startswith("race:"))
    assert raced <= 150


def test_polish_active_best_starts_from_active_methods_best():
    pol = Scripted([{"kind": "switch", "method": "cma_es", "start": "random", "budget": 60},
                    {"kind": "polish", "method": "lbfgsb", "start": "active_best", "budget": 20}])
    _, ctl = make("dev_f15_d5", budget=300, policy=pol)
    ctl.max_rounds = 1
    ctl.run()
    xb = ctl.active["opt"].best()[0]
    ctl.max_rounds = 2
    ctl.run()
    first_polish = [i for i, t in enumerate(ctl.archive.tags) if t == "lbfgsb"][0]
    assert np.allclose(ctl.archive.X[first_polish], xb)
