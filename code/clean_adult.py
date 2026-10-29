"""Clean the adult resting datasets (Dortmund, LEMON, SRM) with ICA + ICLabel
in EEGLAB (clean_eeglab.m) and re-extract their spectra with the datasets'
own extractors, streaming as hbn_clean_all.py does for HBN:
  - a feeder thread stages up to --ahead sessions: downloads (Dortmund,
    LEMON) or copies (SRM, with the EDF start time set to 00.00.00, which
    one file gets wrong) the session's files into WORK/jobs/<job>/, then
    writes WORK/jobs/<job>.json, which the MATLAB processes pick up;
  - --workers MATLAB processes (one thread each) run clean_eeglab.m;
  - every minute the cleaned sessions go through the extractor's own one()
    with <DATASET>_CLEANED set, into --out (default <data>/<dataset>_psd_iclabel),
    one after the other in a fresh process (a process pool broke with
    WinError 6 on the loaded machine and took the run down, 2026-10-28), and
    their staged and cleaned files are deleted (diag files and logs stay).
A session is finished when its spectrum file exists, so running the same
command again resumes; one whose staging or extraction fails 3 times is
listed in WORK/failed.txt. A staging failure while the internet is down is
not counted.

Usage: python clean_adult.py dortmund|lemon|srm [--data DIR] [--work DIR]
           [--out DIR] [--workers 8] [--ahead 16] [--limit N]
"""
import argparse
import csv
import glob
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MATLAB = "C:/Program Files/MATLAB/R2026a/bin/matlab.exe"
TRIES = 3
ROI = ["O1", "Oz", "O2", "PO3", "POz", "PO4"]
SPEC = {"dortmund": dict(module="dortmund_extract_psd", env="DORTMUND_CLEANED", ref="FCz", eog=[]),
        "lemon": dict(module="lemon_extract_psd", env="LEMON_CLEANED", ref="FCz", eog=["VEOG"]),
        "srm": dict(module="srm_extract_psd", env="SRM_CLEANED", ref="", eog=[])}


def jobs_for(dataset, data):
    """[(job name, extractor one() argument, staging function)]."""
    X = importlib.import_module(SPEC[dataset]["module"])
    out = []
    if dataset == "dortmund":
        for p in X.participants():
            sub = p["participant_id"]
            for s in (1, 2):
                if p.get(f"session{s}") != "yes":
                    continue
                files = [X.file_name(sub, s, task, acq) for acq in ("pre", "post") for _, task in X.RECS]

                def stage(dest, sub=sub, s=s, files=files):
                    got = [f for f in files if X.fetch(f"{X.S3}/{sub}/ses-{s}/eeg/{f}", os.path.join(dest, f))]
                    return got or "no recording"
                out.append((f"{sub}_ses-{s}", (sub, s, p), stage))
    elif dataset == "lemon":
        meta = X.participants()
        for sub in X.subjects():
            def stage(dest, sub=sub):
                paths = X.stage(sub, dest)
                return [os.path.basename(paths["vhdr"])] if isinstance(paths, dict) else paths
            out.append((sub, (sub, meta.get(sub, {})), stage))
    else:
        root = os.path.join(data, "ds003775")
        with open(os.path.join(root, "participants.tsv")) as fh:
            meta = {r["participant_id"]: r for r in csv.DictReader(fh, delimiter="\t")}
        for path in sorted(glob.glob(os.path.join(root, "sub-*", "ses-*", "eeg", "*_eeg.edf"))):
            base = os.path.basename(path)
            sub, ses = base.split("_")[:2]

            def stage(dest, path=path, base=base):
                with open(path, "rb") as src, open(os.path.join(dest, base), "wb") as dst:
                    head = bytearray(src.read(256))
                    head[176:184] = b"00.00.00"
                    dst.write(head)
                    shutil.copyfileobj(src, dst)
                return [base]
            out.append((f"{sub}_{ses}", (path, meta.get(sub, {})), stage))
    return out


def online(host="s3.amazonaws.com"):
    """False while DNS or the connection fails: a failed download then says
    nothing about the recording (a night's outage marked 452 HBN recordings
    failed, 2026-10-28)."""
    try:
        socket.create_connection((host, 443), timeout=10).close()
        return True
    except OSError:
        return False


def extract_batch(dataset, path):
    """Run one() on each job of PATH in this process; started by Run.extract
    in a fresh process because the extractors read <DATASET>_CLEANED at import."""
    X = importlib.import_module(SPEC[dataset]["module"])
    for name, args in json.load(open(path)):
        try:
            s = X.one(args)[-1]
        except Exception as e:
            s = f"error: {type(e).__name__}: {e}"
        print(f"{name}\t{s}", flush=True)


class Run:
    def __init__(self, a):
        self.a, self.spec = a, SPEC[a.dataset]
        self.work = a.work or os.path.join(a.data, f"clean_{a.dataset}")
        self.out = a.out or os.path.join(a.data, f"{a.dataset}_psd_iclabel")
        self.jobs_dir, self.clean = os.path.join(self.work, "jobs"), os.path.join(self.work, "clean")
        self.logs = os.path.join(self.work, "logs")
        for d in (self.jobs_dir, self.clean, self.logs, self.out):
            os.makedirs(d, exist_ok=True)
        self.jobs = jobs_for(a.dataset, a.data)
        if a.limit:
            self.jobs = self.jobs[:a.limit]
        self.fails, self.bad = {}, set()
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def finished(self, name):
        return os.path.exists(os.path.join(self.out, f"{name}.npz"))

    def cleaned(self, name):
        return os.path.exists(os.path.join(self.clean, f"{name}_diag.mat"))

    def staged(self, name):
        return os.path.exists(os.path.join(self.jobs_dir, f"{name}.json"))

    def fail(self, name, why):
        with self.lock:
            self.fails[name] = self.fails.get(name, 0) + 1
            if self.fails[name] >= TRIES and name not in self.bad:
                self.bad.add(name)
                with open(os.path.join(self.work, "failed.txt"), "a") as fh:
                    fh.write(f"{name}\t{why}\n")

    def clean_errors(self):
        out = set()
        for f in glob.glob(os.path.join(self.clean, "clean_log_*.csv")):
            for line in open(f):
                parts = line.split(",", 2)
                if len(parts) > 1 and parts[1].startswith("error"):
                    out.add(parts[0])
        return out & {os.path.basename(d) for d in glob.glob(os.path.join(self.clean, "claims", "*"))}

    def stage_one(self, job):
        name, _, stage = job
        dest = os.path.join(self.jobs_dir, name)
        os.makedirs(dest, exist_ok=True)
        try:
            files = stage(dest)
        except Exception as e:
            files = f"{type(e).__name__}: {e}"
        if isinstance(files, str):
            shutil.rmtree(dest, ignore_errors=True)
            up = online()
            print(f"stage {name}: {files}{'' if up else ' (offline, not counted)'}", flush=True)
            if up:
                self.fail(name, f"stage: {files}")
            return
        spec = dict(files=files, ref=self.spec["ref"], mains=50,
                    eog=self.spec["eog"] or [""], roi=ROI)
        tmp = os.path.join(self.jobs_dir, f"{name}.json.part")
        with open(tmp, "w") as fh:
            json.dump(spec, fh)
        os.replace(tmp, os.path.join(self.jobs_dir, f"{name}.json"))

    def feeder(self):
        while not self.stop.is_set():
            waiting = sum(self.staged(n) and not self.cleaned(n) for n, _, _ in self.jobs)
            todo = [j for j in self.jobs if j[0] not in self.bad and not self.staged(j[0])
                    and not self.finished(j[0])][:max(self.a.ahead - waiting, 0)]
            if todo:
                with ThreadPoolExecutor(4) as ex:
                    list(ex.map(self.stage_one, todo))
            self.stop.wait(20)

    def worker(self, k):
        cmd = (f"maxNumCompThreads(1); cd('{HERE}'); "
               f"clean_eeglab('{self.jobs_dir}', '{self.clean}', true)").replace("\\", "/")
        log = open(os.path.join(self.logs, f"matlab_{k}.log"), "a")
        return subprocess.Popen([MATLAB, "-batch", cmd], stdout=log, stderr=subprocess.STDOUT, cwd=HERE)

    def extract(self):
        ready = [j for j in self.jobs if j[0] not in self.bad and self.cleaned(j[0])
                 and self.staged(j[0]) and not self.finished(j[0])]
        if not ready:
            return
        env = dict(os.environ, **{self.spec["env"]: self.clean, "DORTMUND_OUT": self.out,
                                  "LEMON_OUT": self.out, "SRM_OUT": self.out})
        batch = os.path.join(self.logs, "extract_batch.json")
        with open(batch, "w") as fh:
            json.dump([(j[0], j[1]) for j in ready], fh)
        with open(os.path.join(self.logs, "extract.log"), "a") as log:
            subprocess.run([sys.executable, os.path.abspath(__file__), self.a.dataset,
                            "--extract-batch", batch], env=env, stdout=log, stderr=subprocess.STDOUT)
        for name, _, _ in ready:
            if not self.finished(name):
                self.fail(name, "extraction")
                continue
            files = json.load(open(os.path.join(self.jobs_dir, f"{name}.json")))["files"]
            for f in files:
                p = os.path.join(self.clean, os.path.splitext(f)[0] + ".set")
                if os.path.exists(p):
                    os.remove(p)
            shutil.rmtree(os.path.join(self.jobs_dir, name), ignore_errors=True)
            os.remove(os.path.join(self.jobs_dir, f"{name}.json"))

    def prepare(self):
        clean = self.clean.replace("\\", "/")
        for p in psutil.process_iter(["name", "cmdline"]):
            cl = " ".join(p.info["cmdline"] or []).replace("\\", "/")
            if "matlab" in (p.info["name"] or "").lower() and clean in cl:
                sys.exit(f"MATLAB process {p.pid} still cleans into {clean}; stop it first")
        for f in [os.path.join(self.clean, "STOP")] + glob.glob(os.path.join(self.clean, "*_part.set")):
            if os.path.exists(f):
                os.remove(f)
        for d in glob.glob(os.path.join(self.clean, "claims", "*")):
            os.rmdir(d)   # no process holds them now: unfinished or failed sessions, retried

    def run(self):
        self.prepare()
        names = [j[0] for j in self.jobs]
        n0 = sum(map(self.finished, names))
        print(f"{self.a.dataset}: {len(names)} sessions, {n0} finished, {self.a.workers} MATLAB processes",
              flush=True)
        threading.Thread(target=self.feeder, daemon=True).start()
        procs = [self.worker(k) for k in range(self.a.workers)]
        t0, restarts = time.time(), 0
        while True:
            self.extract()
            done = sum(map(self.finished, names))
            errs = self.clean_errors()
            left = [n for n in names if not self.finished(n) and n not in self.bad and n not in errs]
            rate = (done - n0) / max(time.time() - t0, 1) * 3600
            eta = f"{len(left) / rate:.1f} h" if rate > 0 else "?"
            print(f"{time.strftime('%m-%d %H:%M')} finished {done}/{len(names)}, {len(self.bad)} failed, "
                  f"{len(errs)} cleaning errors, {rate:.1f}/h, ETA {eta}", flush=True)
            if not left:
                break
            for k, p in enumerate(procs):
                if p.poll() is not None and restarts < 20:
                    print(f"MATLAB process {k} exited ({p.returncode}); restarting", flush=True)
                    procs[k] = self.worker(k)
                    restarts += 1
            time.sleep(60)
        open(os.path.join(self.clean, "STOP"), "w").close()
        self.stop.set()
        for p in procs:
            p.wait()
        self.extract()
        print(f"DONE {sum(map(self.finished, names))}/{len(names)} finished", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(SPEC))
    ap.add_argument("--data", default=os.environ.get("EEG_DATA", "C:/Users/ccann/eeg_data"))
    ap.add_argument("--work", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ahead", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extract-batch", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.extract_batch:
        extract_batch(a.dataset, a.extract_batch)
    else:
        Run(a).run()


if __name__ == "__main__":
    main()
