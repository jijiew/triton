"""Rules behind every number on the /ci/ dashboard.

Pure functions over stored run records (see ci_health.py for the record shape).
page_data() builds the data.json the page renders, one section per tab:

  main     did a push to main break something new? Per test step (GPU, job,
           step), each failing test is new (it passed on the previous push
           that ran the step) or known (it failed there too, and keeps the push
           it first failed at). A skipped or setup-failed step has no result
           and is invisible to the comparison.
  nightly  the nightly release and the bench dashboard's perf events.
  pr       the latest run of each PR in the last 24 h, tagged by whose it is.
  ops      GPU hours, job durations and setup failures.
No I/O.
"""
import collections
import datetime as dt
import re

#: The reusable workflows gpu-tests.yml calls, one per GPU. Their step names define the test steps.
GPU_WORKFLOWS = ("mi350.yml", "b200.yml", "h100.yml")
#: The summary job of gpu-tests.yml: fails whenever any suite does, so it is ignored everywhere.
SUMMARY_JOB = "GPU tests"

#: Step name -> test group. A step with a group is a test step; any other step is setup.
GROUPS = [
    (r"^Run TLX (core|unit) tests$", "TLX core"),  # MI350 called it "unit" until its rename
    (r"^Run TLX tutorial correctness tests$", "Tutorial correctness"),
    (r"^Run device-agnostic TLX codegen tests$", "TLX codegen"),
    (r"^Run TritonBench", "TritonBench"),
    (r"^Run Gluon tests", "Gluon"),
    (r"^Run runtime unit tests$", "Runtime"),
    (r"^Run host-default TLX L2 perf$", "TLX perf"),
    (r"^Run TorchTLX", "TorchTLX"),
]
#: Job name -> job group, for runner time. *-meta-triton-test was renamed *-tritonbench-test in #3943.
JOB_GROUPS = [
    (r"-build", "Build"),
    (r"-tlx-test$", "TLX"),
    (r"-(tritonbench|meta-triton)-test$", "TritonBench"),
    (r"-gluon-test$", "Gluon"),
    (r"-runtime-test$", "Runtime"),
    (r"-torchtlx-test$", "TorchTLX"),
    (r"^(report|reconcile)-", "Reporting"),
]
#: A failed step matching this is an infrastructure failure, not a test result.
INFRA_STEP = re.compile(
    r"set up job|checkout|download prebuilt|install triton|upload wheel|build wheel|setup tritonbench|"
    r"authenticate|azure|aws|restore ccache|tune nvidia|fail if the build job|select kernel-affected|"
    r"set up test environment|install prebuilt triton|resolve tritonbench ref|install tritonbench dependencies", re.I)


def group_of(step):
    for pat, g in GROUPS:
        if re.search(pat, step):
            return g
    return None


def job_group(job):
    for pat, g in JOB_GROUPS:
        if re.search(pat, job):
            return g
    return "Other"


def t(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def mins(a, b):
    return (t(b) - t(a)).total_seconds() / 60 if a and b else 0.0


def iso(d):
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_job(j):
    return bool(j.get("labels")) and not any("ubuntu" in label for label in j["labels"])


GPU_PREFIX = {"h100": "H100", "b200": "B200", "mi350": "MI350"}
GPUS = ("H100", "B200", "MI350")
#: Shared jobs of gpu-tests.yml that every GPU job needs: their failure is a build failure, not a setup failure.
BUILD_JOBS = ("build", "select")
#: How many pushes the Main tab's grid and setup counts look back.
PUSH_GRID = 12
SETUP_WINDOW = 10
RAN = ("success", "failure", "cancelled")


def gpu_of(run, job):
    """The GPU a gpu-tests.yml job ran on, from its "h100 / " name prefix."""
    if run.get("wf") != "gpu-tests.yml":
        return None
    prefix, sep, _ = job["name"].partition(" / ")
    return GPU_PREFIX.get(prefix) if sep else None


def job_key(name):
    return name.split(" / ", 1)[-1]


def _gpu_jobs(run):
    return [(gpu_of(run, j), j) for j in run.get("jobs") or [] if gpu_of(run, j) and gpu_job(j)]


def step_result(job, step):
    """"pass" or "fail" for a test step that ran to a verdict; None when it has no result (skipped, cancelled,
    not a test step, or a failed step whose log gave no failures)."""
    if group_of(step["name"]) is None:
        return None
    if step["conclusion"] == "success":
        return "pass"
    if step["conclusion"] == "failure" and "failures" in step:
        return "fail"
    return None


def setup_step(job):
    """The step a setup-failed job died in, or None if the job has a test failure or did not fail."""
    if job["conclusion"] != "failure":
        return None
    failed = [s for s in job["steps"] if s["conclusion"] == "failure"]
    if any(group_of(s["name"]) and "failures" in s for s in failed):
        return None
    return failed[0]["name"] if failed else "runner lost (no failed step)"


def _build_failure(run):
    for j in run.get("jobs") or []:
        if j["name"] in BUILD_JOBS and j["conclusion"] == "failure":
            failed = [s["name"] for s in j["steps"] if s["conclusion"] == "failure"]
            return failed[0] if failed else j["name"]
    return None


def run_outcome(run):
    """passed / test / setup / build / cancelled, or None while the run is not finished."""
    if run.get("status") != "completed":
        return None
    if run["conclusion"] == "cancelled":
        return "cancelled"
    if run["conclusion"] == "success":
        return "passed"
    if _build_failure(run):
        return "build"
    for _, j in _gpu_jobs(run):
        if any(group_of(s["name"]) and s["conclusion"] == "failure" for s in j["steps"]):
            return "test"
    return "setup"


def _pushes(runs):
    return sorted((r for r in runs if r.get("wf") == "gpu-tests.yml" and r["event"] == "push"
                   and r.get("jobs") is not None and r.get("status") == "completed"), key=lambda r: r["created_at"])


def _commit(run):
    return {"sha": run.get("head_sha") or "", "pr": run.get("pr"), "author": run.get("author"), "t": run["created_at"]}


def _history(pushes):
    """Walk pushes oldest first. A unit is (gpu, job, test group), so a renamed step keeps its history. Returns, per
    unit: its current failing tests, the run index of its last result and its latest step name; and per run index:
    {gpu: {"new", "known"}} statuses seen there."""
    state, last, names, seen = {}, {}, {}, collections.defaultdict(dict)
    for i, run in enumerate(pushes):
        for gpu, j in _gpu_jobs(run):
            for s in j["steps"]:
                res = step_result(j, s)
                if res is None:
                    continue
                unit = (gpu, job_key(j["name"]), group_of(s["name"]))
                names[unit] = s["name"]
                prev, cur = state.get(unit, {}), {}
                for f in s.get("failures") or [] if res == "fail" else []:
                    if f["id"] in prev:
                        cur[f["id"]] = {**prev[f["id"]], "streak": prev[f["id"]]["streak"] + 1}
                    else:
                        base = pushes[last[unit]] if unit in last else None
                        cur[f["id"]] = {
                            "id": f["id"], "msg": f["msg"], "first": i, "prev_sha": (base or {}).get("head_sha"),
                            "streak": 1
                        }
                    seen[i].setdefault(gpu, set()).add("new" if cur[f["id"]]["streak"] == 1 else "known")
                state[unit], last[unit] = cur, i
    return state, last, names, seen


def _sched_result(runs, sha, unit, test_id):
    gpu, key, group = unit
    found = "none"
    for run in runs:
        if run.get("wf") != "gpu-tests.yml" or run["event"] != "schedule" or run.get("head_sha") != sha:
            continue
        for g, j in _gpu_jobs(run):
            if g != gpu or job_key(j["name"]) != key:
                continue
            for s in j["steps"]:
                if group_of(s["name"]) == group and step_result(j, s):
                    if any(f["id"] == test_id for f in s.get("failures") or []):
                        return "reproduced"
                    found = "not reproduced"
    return found


def main_view(runs, asof):
    pushes = [r for r in _pushes(runs) if r["created_at"] <= asof]
    state, last, names, seen = _history(pushes)
    # A unit counts only while its test group still exists in the latest push that ran its job.
    latest_steps = {}
    for run in pushes:
        for gpu, j in _gpu_jobs(run):
            if j["conclusion"] in RAN:
                latest_steps[(gpu, job_key(j["name"]))] = {group_of(s["name"]) for s in j["steps"]} - {None}
    gpus = {}
    for gpu in GPUS:
        ran = [r for r in pushes if any(g == gpu and j["conclusion"] in RAN for g, j in _gpu_jobs(r))]
        tail = pushes[-SETUP_WINDOW:]
        out = {
            "verdict":
            "none", "new": [], "known": [], **(_commit(ran[-1]) if ran else {}), "setup_last10":
            sum(any(g == gpu and setup_step(j) for g, j in _gpu_jobs(r)) for r in tail)
        }
        for unit, fails in state.items():
            if unit[0] != gpu or unit[2] not in latest_steps.get(unit[:2], ()):
                continue
            for f in fails.values():
                first = pushes[f["first"]]
                item = {
                    "gpu": gpu, "job": unit[1], "step": names[unit], "id": f["id"], "msg": f["msg"], "first_sha":
                    first.get("head_sha") or "", "first_pr": first.get("pr"), "first_author": first.get("author"),
                    "first_t": first["created_at"], "first_url": first.get("url"), "prev_sha": f["prev_sha"], "streak":
                    f["streak"]
                }
                if f["streak"] == 1:
                    item["sched"] = _sched_result(runs, item["first_sha"], unit, f["id"])
                    out["new"].append(item)
                else:
                    out["known"].append(item)
        if ran:
            out["verdict"] = "new" if out["new"] else "known" if out["known"] else "passing"
        for k in ("new", "known"):
            out[k].sort(key=lambda x: (x["first_t"], x["job"], x["id"]))
        gpus[gpu] = out

    grid = []
    for i in range(max(0, len(pushes) - PUSH_GRID), len(pushes)):
        run, cells = pushes[i], {}
        broken = _build_failure(run)
        for gpu in GPUS:
            jobs = [j for g, j in _gpu_jobs(run) if g == gpu and j["conclusion"] in RAN]
            st = seen[i].get(gpu, set())
            if broken or not jobs:
                cells[gpu] = "na"
            elif "new" in st:
                cells[gpu] = "new"
            elif "known" in st:
                cells[gpu] = "known"
            elif any(setup_step(j) for j in jobs):
                cells[gpu] = "setup"
            else:
                cells[gpu] = "pass"
        grid.append({**_commit(run), "title": run.get("title", ""), "url": run.get("url"), "cells": cells})

    setup = [{
        "sha": r.get("head_sha") or "", "t": r["created_at"], "gpu": g, "job": job_key(j["name"]), "step": setup_step(j)
    } for r in pushes[-SETUP_WINDOW:] for g, j in _gpu_jobs(r) if setup_step(j)]
    build = None
    if pushes and _build_failure(pushes[-1]):
        build = {**_commit(pushes[-1]), "step": _build_failure(pushes[-1])}
    return {"gpus": gpus, "build": build, "pushes": grid, "setup": setup}


def pr_view(runs, asof, main):
    since = iso(t(asof) - dt.timedelta(hours=24))
    on_main = {(i["gpu"], i["job"], i["id"]) for g in main["gpus"].values() for i in g["new"] + g["known"]}
    latest = {}
    for run in runs:
        if run.get("wf") != "gpu-tests.yml" or run["event"] != "pull_request" or not (since <= run["created_at"] <=
                                                                                      asof):
            continue
        key = run.get("pr") or ("branch", run.get("branch"))
        if key not in latest or run["created_at"] > latest[key]["created_at"]:
            latest[key] = run
    out = []
    for run in sorted(latest.values(), key=lambda r: r["created_at"], reverse=True):
        jobs = _gpu_jobs(run)
        fails = {(g, job_key(j["name"]), f["id"]) for g, j in jobs for s in j["steps"] for f in s.get("failures") or []}
        outcome = run_outcome(run)
        row = {
            "pr": run.get("pr"), "title": run.get("title", ""), "author": run.get("author"), "url": run.get("url"),
            "branch": run.get("branch"), "t": run["created_at"], "gpus":
            sorted({g
                    for g, j in jobs
                    if j["conclusion"] in RAN}, key=GPUS.index), "n_failed": len(fails), "whose": None, "note": ""
        }
        if outcome is None:
            row["result"] = "running"
        elif fails:
            row["result"] = "tests"
            if fails <= on_main:
                row["whose"], row["note"] = "main", f"all {len(fails)} also fail on main"
            else:
                row["whose"] = "author"
        elif outcome in ("setup", "build"):
            row["result"] = outcome if outcome == "build" else "setup"
            row["whose"] = "author" if outcome == "build" else "ci"
            row["note"] = _build_failure(run) or next((setup_step(j) for _, j in jobs if setup_step(j)), "")
        else:
            row["result"] = outcome
        out.append(row)
    return out


def nightly_view(runs, asof, bench):
    nights = sorted(
        (r for r in runs
         if r.get("wf") == "nightly-fbtriton.yml" and r.get("status") == "completed" and r["created_at"] <= asof),
        key=lambda r: r["created_at"])
    week_ago = iso(t(asof) - dt.timedelta(days=7))

    def job_result(run, name):
        for j in run.get("jobs") or []:
            if j["name"] == name or j["name"].startswith(name + " / "):
                return j["conclusion"]
        return None

    release = None
    if nights:
        n = nights[-1]
        release = {
            "conclusion": n["conclusion"], "build": job_result(n, "build"), "publish": job_result(n, "publish"), "t":
            n["created_at"], "url": n.get("url"), "published_7":
            sum(job_result(r, "publish") == "success" for r in nights if r["created_at"] >= week_ago), "nights_7":
            sum(r["created_at"] >= week_ago for r in nights)
        }
    perf = []
    for platform, events in sorted((bench or {}).items()):
        for night in events[-2:]:  # the latest night is unconfirmed by construction, so keep the one before too
            for op, e in sorted(night.get("perf", {}).items()):
                perf.append({
                    "platform": platform, "date": night["date"], "op": op, "n_slower": e["n_slower"], "n_faster":
                    e["n_faster"], "signature": e.get("signature"), "unconfirmed": e.get("unconfirmed"), "range":
                    night.get("range"), "compare": night.get("compare"), "suspects": e.get("suspects", [])[:3]
                })
    return {"release": release, "perf": perf}


def _pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None


def ops_view(runs, asof):
    now = t(asof)
    week_ago, two_weeks = iso(now - dt.timedelta(days=7)), iso(now - dt.timedelta(days=14))
    gpu_runs = [r for r in runs if r.get("wf") == "gpu-tests.yml" and r["created_at"] <= asof and r.get("jobs")]
    days = [str((now - dt.timedelta(days=d)).date()) for d in range(13, -1, -1)]
    hours = {d: {"day": d, "passed": 0.0, "test": 0.0, "setup": 0.0, "cancelled": 0.0} for d in days}
    for r in gpu_runs:
        k = {"build": "test"}.get(run_outcome(r), run_outcome(r))
        if k in ("passed", "test", "setup", "cancelled") and r["created_at"][:10] in hours:
            hours[r["created_at"][:10]][k] += sum(mins(j["started_at"], j["completed_at"])
                                                  for _, j in _gpu_jobs(r)) / 60

    def gpu_h(lo, hi):
        return round(
            sum(
                mins(j["started_at"], j["completed_at"]) for r in gpu_runs if lo <= r["created_at"] < hi
                for _, j in _gpu_jobs(r)) / 60, 1)

    recent = [r for r in gpu_runs if r["created_at"] >= week_ago]
    finished = [r for r in recent if run_outcome(r) in ("passed", "test", "setup", "build")]
    setups = [r for r in finished if run_outcome(r) == "setup"]
    queue = [mins(j["created_at"], j["started_at"]) for r in recent for _, j in _gpu_jobs(r) if j.get("created_at")]
    prs = [r for r in recent if r["event"] == "pull_request"]

    durations, prev, step_t = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(
        lambda: collections.defaultdict(list))
    for r in gpu_runs:
        for gpu, j in _gpu_jobs(r):
            if j["conclusion"] not in ("success", "failure") or r["created_at"] < two_weeks:
                continue
            key = job_key(j["name"]).split("-", 1)[-1] if job_key(j["name"]).split(
                "-", 1)[0] in GPU_PREFIX else job_key(j["name"])
            m = mins(j["started_at"], j["completed_at"])
            if r["created_at"] >= week_ago:
                durations[key].append(m)
                for s in j["steps"]:
                    if s.get("started_at") and s.get("completed_at"):
                        step_t[key][s["name"]].append(mins(s["started_at"], s["completed_at"]))
            else:
                prev[key].append(m)
    jobs = []
    for key in sorted(durations):
        slow = max(step_t[key].items(), key=lambda kv: _pct(kv[1], 0.5), default=(None, []))
        jobs.append({
            "job": key, "n": len(durations[key]), "p50": _pct(durations[key], 0.5), "p90": _pct(durations[key], 0.9),
            "p90_prev": _pct(prev[key], 0.9), "slowest_step": slow[0], "slowest_p50": _pct(slow[1], 0.5), "slowest_p90":
            _pct(slow[1], 0.9)
        })
    jobs.sort(key=lambda x: -(x["p90"] or 0))

    steps = collections.defaultdict(lambda: {"n": 0, "triggers": collections.Counter()})
    for r in gpu_runs:
        if r["created_at"] < week_ago:
            continue
        for gpu, j in _gpu_jobs(r):
            name = setup_step(j)
            if name:
                e = steps[(name, gpu)]
                e["n"] += 1
                e["triggers"][r["event"]] += 1
    setup_steps = sorted(({"step": k[0], "gpu": k[1], "n": v["n"], "triggers": dict(v["triggers"])}
                          for k, v in steps.items()), key=lambda x: -x["n"])
    return {
        "tiles": {
            "gpu_h_7d":
            gpu_h(week_ago, asof + "~"), "gpu_h_prev_7d":
            gpu_h(two_weeks, week_ago), "setup_n":
            len(setups), "runs_n":
            len(finished), "setup_rate":
            round(len(setups) / len(finished), 3) if finished else None, "setup_mi350":
            sum(any(g == "MI350" and setup_step(j) for g, j in _gpu_jobs(r)) for r in setups), "queue_p50":
            _pct(queue, 0.5), "queue_p90":
            _pct(queue, 0.9), "pr_n":
            len(prs), "cancelled_pr":
            sum(r["conclusion"] == "cancelled" for r in prs), "cancelled_gpu_h":
            round(
                sum(
                    mins(j["started_at"], j["completed_at"]) for r in prs if r["conclusion"] == "cancelled"
                    for _, j in _gpu_jobs(r)) / 60, 1)
        }, "gpu_hours": [{**v, **{k: round(v[k], 1)
                                  for k in ("passed", "test", "setup", "cancelled")}}
                         for v in hours.values()], "jobs":
        jobs, "setup_steps":
        setup_steps
    }


def page_data(runs, asof, bench):
    """The data.json the /ci/ page renders, computed from stored run records and bench perf events."""
    main = main_view(runs, asof)
    return {
        "asof": asof, "main": main, "nightly": nightly_view(runs, asof, bench), "pr": pr_view(runs, asof, main), "ops":
        ops_view(runs, asof)
    }
