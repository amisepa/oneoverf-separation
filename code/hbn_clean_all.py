"""Clean every HBN resting recording in EEGLAB and re-extract its spectra.

The raw sets of all ~2,800 recordings (~100 MB each) do not fit on disk, so
this streams:
  - a feeder thread keeps --ahead raw sets downloaded ahead of the cleaning;
  - --workers MATLAB processes (hbn_clean_eeglab.m, one thread each, each
    claiming one recording at a time) clean them with --method; separate
    processes rather than a parallel pool, so clean_rawdata's RANSAC gives
    the same bad channels whichever process runs it;
  - every minute the recordings cleaned since are extracted twice with
    hbn_extract_psd.py, the cleaned set into WORK/psd_<method> and the raw
    set into WORK/psd_primary_seg (the uncleaned spectra with the
    per-segment fields hbn_clean_check.py segmatch needs), and their raw and
    cleaned sets are deleted (the diag .mat files and logs stay;
    --keep-clean keeps the cleaned sets).

A recording is finished when both spectra exist, so running the same command
again resumes. A recording whose cleaning fails keeps its claim
(WORK/clean_<method>/claims) and is retried at the next start; one whose
download or extraction fails 3 times is listed in WORK/failed.txt.

Subjects: every <ds>_<sub>.npz in --psd-dir (the primary spectra), or the
"ds sub" lines of --subjects. --seed-raw DIR hard-links raw sets already on
disk (e.g. the cleaning check's raw/) instead of downloading them.

--irasa: a recording is finished only when both spectra files hold the IRASA
spectra (hbn_extract_psd.py added them on 2026-10-28, after ~530 recordings
were done). The others are downloaded, cleaned and extracted again, and the
extraction keeps the replaced files as <name>.npz.old, so the Welch spectra
of the two runs can be compared (cleaning is deterministic, so they should
match).

Usage: python hbn_clean_all.py [--work DIR] [--psd-dir DIR] [--method iclabel]
           [--workers 8] [--ahead 24] [--subjects FILE] [--seed-raw DIR]
           [--keep-clean] [--irasa]
"""
import argparse
import glob
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hbn_clean_check import fetch_raw

MATLAB = "C:/Program Files/MATLAB/R2026a/bin/matlab.exe"
TRIES = 3


class Run:
    def __init__(self, a):
        self.a = a
        w = a.work
        self.raw, self.links = os.path.join(w, "raw"), os.path.join(w, "raw_as_clean")
        self.clean = os.path.join(w, f"clean_{a.method}")
        self.psd, self.prim = os.path.join(w, f"psd_{a.method}"), os.path.join(w, "psd_primary_seg")
        self.logs = os.path.join(w, "logs")
        for d in (self.raw, self.links, self.clean, self.psd, self.prim, self.logs):
            os.makedirs(d, exist_ok=True)
        if a.subjects:
            self.names = ["_".join(l.split()) for l in open(a.subjects) if l.strip()]
        else:
            self.names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.psd_dir, "*.npz")))
        self.fails, self.bad = {}, set()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.irasa_seen = {}

    def clean_set(self, n):
        return os.path.join(self.clean, f"{n}_clean.set")

    def has_irasa(self, path):
        """Cached by modification time: every recording is asked about every minute."""
        t = os.path.getmtime(path)
        if self.irasa_seen.get(path, (None,))[0] != t:
            with np.load(path) as d:
                self.irasa_seen[path] = (t, "ec_irasa" in d.files)
        return self.irasa_seen[path][1]

    def finished(self, n):
        files = (os.path.join(self.psd, f"{n}.npz"), os.path.join(self.prim, f"{n}.npz"))
        if not all(map(os.path.exists, files)):
            return False
        return not self.a.irasa or all(map(self.has_irasa, files))

    def fail(self, n, why):
        with self.lock:
            self.fails[n] = self.fails.get(n, 0) + 1
            if self.fails[n] >= TRIES and n not in self.bad:
                self.bad.add(n)
                with open(os.path.join(self.a.work, "failed.txt"), "a") as fh:
                    fh.write(f"{n}\t{why}\n")

    def clean_errors(self):
        """Recordings whose cleaning failed in this run (they keep their claim)."""
        out = set()
        for f in glob.glob(os.path.join(self.clean, "clean_log_*.csv")):
            for line in open(f):
                parts = line.split(",", 2)
                if len(parts) > 1 and parts[1].startswith("error"):
                    out.add(parts[0][:-4])
        claimed = {os.path.basename(d)[:-4] for d in glob.glob(os.path.join(self.clean, "claims", "*"))}
        return out & claimed

    def feeder(self):
        while not self.stop.is_set():
            on_disk = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(self.raw, "*.set"))}
            waiting = [n for n in on_disk if not os.path.exists(self.clean_set(n))]
            need = self.a.ahead - len(waiting)
            todo = [n for n in self.names
                    if n not in on_disk and n not in self.bad and not self.finished(n)][:max(need, 0)]
            offline = False
            if todo:
                with ThreadPoolExecutor(4) as ex:
                    st = list(ex.map(lambda n: fetch_raw(*n.split("_", 1), self.raw), todo))
                for n, s in zip(todo, st):
                    if s not in ("ok", "cached"):
                        print(f"download {n}: {s}", flush=True)
                        # no connection at all (URLError, not an HTTP error) is not the
                        # recording's fault: a DNS outage on 2026-10-29 marked 452 failed
                        if "URLError" in s:
                            offline = True
                        else:
                            self.fail(n, f"download: {s}")
            if offline:
                print(f"{time.strftime('%m-%d %H:%M')} network down? downloads paused 5 min", flush=True)
            self.stop.wait(300 if offline else 20)

    def worker(self, k):
        cmd = (f"maxNumCompThreads(1); cd('{HERE}'); "
               f"hbn_clean_eeglab('{self.raw}', '{self.clean}', '{self.a.method}', true)").replace("\\", "/")
        log = open(os.path.join(self.logs, f"matlab_{k}.log"), "a")
        return subprocess.Popen([MATLAB, "-batch", cmd], stdout=log, stderr=subprocess.STDOUT, cwd=HERE)

    def extract(self):
        """Extract the recordings cleaned but not yet extracted, then delete
        the raw and cleaned sets of the finished ones."""
        ready = [n for n in self.names if n not in self.bad and os.path.exists(self.clean_set(n))
                 and os.path.exists(os.path.join(self.raw, f"{n}.set")) and not self.finished(n)]
        if not ready:
            return
        for n in ready:
            link = os.path.join(self.links, f"{n}_clean.set")
            if not os.path.exists(link):
                os.link(os.path.join(self.raw, f"{n}.set"), link)
        subs = os.path.join(self.logs, "extract_batch.txt")
        with open(subs, "w") as fh:
            fh.writelines(n.replace("_", " ", 1) + "\n" for n in ready)
        dss = sorted({n.split("_")[0] for n in ready})
        with open(os.path.join(self.logs, "extract.log"), "a") as log:
            for src, out in ((self.clean, self.psd), (self.links, self.prim)):
                env = dict(os.environ, HBN_LOCAL_SET=src, HBN_OUT=out,
                           HBN_REDO_IRASA="1" if self.a.irasa else "")
                subprocess.run([sys.executable, os.path.join(HERE, "hbn_extract_psd.py"), *dss,
                                "--workers", "2", "--passes", "0", "--subjects", subs],
                               env=env, stdout=log, stderr=subprocess.STDOUT)
        for n in ready:
            if not self.finished(n):
                self.fail(n, "extraction")
                continue
            drop = [os.path.join(self.raw, f"{n}.set"), os.path.join(self.links, f"{n}_clean.set")]
            if not self.a.keep_clean:
                drop.append(self.clean_set(n))
            for p in drop:
                try:
                    os.remove(p)
                except OSError:
                    pass

    def prepare(self):
        clean = self.clean.replace("\\", "/")
        for p in psutil.process_iter(["name", "cmdline"]):
            cl = " ".join(p.info["cmdline"] or []).replace("\\", "/")
            if "matlab" in (p.info["name"] or "").lower() and clean in cl:
                sys.exit(f"MATLAB process {p.pid} still cleans into {clean}; stop it first")
        for f in [os.path.join(self.clean, "STOP")] + glob.glob(os.path.join(self.clean, "*_clean_part.set")):
            if os.path.exists(f):
                os.remove(f)
        for d in glob.glob(os.path.join(self.clean, "claims", "*")):
            os.rmdir(d)   # no process holds them now: unfinished or failed recordings, retried
        if self.a.seed_raw:
            for n in self.names:
                src, dest = os.path.join(self.a.seed_raw, f"{n}.set"), os.path.join(self.raw, f"{n}.set")
                if os.path.exists(src) and not os.path.exists(dest) and not self.finished(n):
                    os.link(src, dest)

    def run(self):
        self.prepare()
        n0 = sum(map(self.finished, self.names))
        print(f"{len(self.names)} recordings, {n0} finished, method {self.a.method}, "
              f"{self.a.workers} MATLAB processes", flush=True)
        threading.Thread(target=self.feeder, daemon=True).start()
        procs = [self.worker(k) for k in range(self.a.workers)]
        t0, restarts = time.time(), 0
        while True:
            self.extract()
            done = sum(map(self.finished, self.names))
            left = [n for n in self.names if not self.finished(n) and n not in self.bad]
            errs = self.clean_errors()
            left = [n for n in left if n not in errs]
            rate = (done - n0) / max(time.time() - t0, 1) * 3600
            eta = f"{len(left) / rate:.1f} h" if rate > 0 else "?"
            print(f"{time.strftime('%m-%d %H:%M')} finished {done}/{len(self.names)}, "
                  f"{len(self.bad)} failed, {len(errs)} cleaning errors, {rate:.1f}/h, ETA {eta}",
                  flush=True)
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
        print(f"DONE {sum(map(self.finished, self.names))}/{len(self.names)} finished", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="C:/Users/ccann/hbn_eeg/clean_all")
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--method", choices=["iclabel", "gedai"], default="iclabel")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ahead", type=int, default=24)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--seed-raw", default="")
    ap.add_argument("--keep-clean", action="store_true")
    ap.add_argument("--irasa", action="store_true")
    Run(ap.parse_args()).run()


if __name__ == "__main__":
    main()
