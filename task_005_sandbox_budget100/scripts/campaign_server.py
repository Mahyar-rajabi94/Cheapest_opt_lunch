"""Persistent single-process campaign server: one harness instance for the whole run."""
import contextlib
import io
import json
import os
import sys
import traceback
from multiprocessing.connection import Listener

REPO = "/home/kevzhu14/Documents/task_005_sandbox_budget100"
SCR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCR)
sys.path.insert(0, REPO)
sys.path.insert(0, SCR)

import numpy as np
import bo_lib as L
from src.agent_harness.task_005 import AgentTaskHarness, BudgetExhausted

harness = AgentTaskHarness(budget=100)
LOG = []
_cur = [None]


def ev(x, method):
    x = L.snap(x)
    if not L.allowed(x):
        raise ValueError("refusing to evaluate a disallowed point")
    if len(LOG) >= 100:
        raise RuntimeError("budget already used")
    if _cur[0] != method:
        harness.set_active_optimizer(method)
        _cur[0] = method
    y = float(harness.evaluate([float(v) for v in x]))
    LOG.append({"n": len(LOG) + 1, "x": x.tolist(), "y": y,
                "allowed": bool(L.allowed(x)), "method": method})
    with open(os.path.join(SCR, "eval_log.json"), "w") as fh:
        json.dump(LOG, fh)
    return y


def data():
    X = np.array([r["x"] for r in LOG])
    y = np.array([r["y"] for r in LOG])
    ok = np.array([r["allowed"] for r in LOG])
    return X, y, ok


ns = globals()
if os.path.exists("srv.sock"):
    os.remove("srv.sock")
listener = Listener("srv.sock", family="AF_UNIX", authkey=b"task005")
print("server ready", harness.dim, harness.direction, flush=True)
while True:
    conn = listener.accept()
    code = conn.recv()
    if code.strip() == "__SHUTDOWN__":
        conn.send("bye")
        conn.close()
        break
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(code, ns)
        except Exception:
            traceback.print_exc()
    conn.send(buf.getvalue())
    conn.close()
listener.close()
