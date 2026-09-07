# Copyright (c) 2026 Makoto Yamada and contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers that make experiment output directories self-describing.

Each run gets its own subdirectory under the user-specified --output_dir,
named from the job id (or a timestamp), the git commit the code was run
from, and a short caller-supplied tag of the hyperparameters that matter
for that script. A run_meta.json with the full args + git info is written
alongside it, so a checkpoint or log directory found later can always be
traced back to the exact code version and configuration that produced it.
"""
import datetime
import json
import os
import subprocess
from pathlib import Path


def get_git_info(repo_dir):
    def run(cmd):
        try:
            return subprocess.check_output(
                cmd, cwd=repo_dir, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return None

    commit = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "short_commit": commit[:7] if commit else "nogit",
        "branch": branch,
        "dirty": bool(status),
    }


def job_id():
    return os.environ.get("PJM_JOBID") or os.environ.get("SLURM_JOB_ID")


def make_run_dir(base_output_dir, tag, git_info, is_main_process, extra=None):
    """Return base_output_dir/<run_name>.

    On the main process only: create the directory and write run_meta.json
    (git info + `extra`, typically the parsed args). All distributed ranks
    must call this with identical `base_output_dir`/`tag`/`git_info` so they
    agree on the resulting path without needing to communicate it.
    """
    jid = job_id() or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = git_info["short_commit"] + ("-dirty" if git_info["dirty"] else "")
    run_name = f"run{jid}_{commit}_{tag}"
    run_dir = os.path.join(base_output_dir, run_name)

    if is_main_process:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        meta = {"git": git_info}
        if extra:
            meta.update(extra)
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    return run_dir
