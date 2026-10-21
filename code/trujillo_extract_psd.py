"""Per-channel spectra of the Trujillo et al. (2019) cortical organoid LFP.

Data: Zenodo record 4751759 (CC-BY-4.0), one MATLAB 7.3 file per weekly
recording, LFP_Sp_YYMMDD.mat, holding a 12-well cell array of [64 channels x
time] LFP at 1 kHz (Axion Maestro, 0.1 Hz-5 kHz hardware band). Wells 1-4 are
empty; the paper analysed wells 5-12. Files from 2017 (and three from 2016)
are MATLAB 7.3 (HDF5), so only the LFP of the requested wells is read, over
HTTP range requests; the other 2016 files are MATLAB 5, whose LFP variable is
streamed and decompressed on the fly (sampling rate 1 kHz, as documented).
Nothing is saved but the spectra.

Per well: Welch spectra of every channel (4-s Hann windows, 50% overlap,
mean removed), 0-200 Hz, and each channel's standard deviation for QC.

Writes OUT/YYMMDD.npz with f, fs, w<k> (64 x freqs, V^2/Hz) and rms<k>
(64,) for each 0-indexed well k.

Usage: python trujillo_extract_psd.py LFP_Sp_170113.mat [--out DIR]
       [--wells 4,5,...,11]
"""
import argparse
import os
import struct
import time
import urllib.request
import zlib

import fsspec
import h5py
import numpy as np
from scipy.signal import welch

URL = "https://zenodo.org/api/records/4751759/files/{}/content"


def _wells_v73(fh, wells):
    """MATLAB 7.3 (HDF5): read only the requested cells."""
    with h5py.File(fh, "r") as f:
        fs = float(np.ravel(f["fs_ds"][()])[0])
        refs = f["LFP"][()].ravel()
        for w in wells:
            yield w, fs, f[refs[w]][()]                           # 64 x time


MI_DTYPES = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4",
             9: "<f8", 12: "<i8", 13: "<u8"}


class _ZStream:
    """Bytes of one compressed MAT element, decompressed as they arrive over HTTP."""

    def __init__(self, url, start, length):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        self.resp = urllib.request.urlopen(req, timeout=120)
        self.z = zlib.decompressobj()
        self.buf = bytearray()

    def read(self, n):
        while len(self.buf) < n:
            chunk = self.resp.read(2 ** 20)
            if not chunk:
                raise EOFError("compressed element ended early")
            self.buf += self.z.decompress(chunk)
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def close(self):
        self.resp.close()


def _subelement(read):
    """Type and data of one element. A small element (type and size packed
    into the first word) keeps up to 4 bytes of data in the second word."""
    raw = read(8)
    t, n = struct.unpack("<II", raw)
    if t >> 16:
        return t & 0xFFFF, raw[4:4 + (t >> 16)]
    data = read(n)
    read((8 - n % 8) % 8)
    return t, data


def _matrix_header(read):
    """Flags, dims and name of a miMATRIX element."""
    _, flags = _subelement(read)
    _, dims = _subelement(read)
    _, name = _subelement(read)
    return struct.unpack("<II", flags)[0] & 0xFF, np.frombuffer(dims, "<i4"), name.decode()


def cells_v5(url, wanted):
    """Stream cells of top-level cell-array variables of a MATLAB 5 file.

    wanted maps a variable name to the set of cell indices to return
    (column-major, as MATLAB stores them). The top-level variables are
    located by their tags, so variables not wanted are skipped without being
    transferred; a wanted one is decompressed as it streams. Yields
    (name, index, array) with each cell reshaped to its MATLAB dimensions
    reversed (a [time x 64] cell comes out 64 x time)."""
    size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                                      timeout=60).headers["Content-Length"])
    off, todo = 128, set(wanted)
    while off < size and todo:
        req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + 7}"})
        t, n = struct.unpack("<II", urllib.request.urlopen(req, timeout=60).read())
        if t == 15:
            zs = _ZStream(url, off + 8, n)
            zs.read(8)
            cls, dims, name = _matrix_header(zs.read)
            if name in todo:
                todo.discard(name)
                keep = wanted[name]
                for k in range(int(np.prod(dims))):
                    ct, cn = struct.unpack("<II", zs.read(8))
                    body = zs.read(cn)
                    if k not in keep:
                        continue
                    if cn == 0:                            # an empty cell has no header
                        yield name, k, np.zeros(0)
                        continue
                    pos = [0]

                    def rd(m, body=body, pos=pos):
                        out = body[pos[0]:pos[0] + m]
                        pos[0] += m
                        return out
                    _, cdims, _ = _matrix_header(rd)
                    if np.prod(cdims) == 0:
                        yield name, k, np.zeros(0)
                        continue
                    dt_, data = _subelement(rd)
                    yield name, k, np.frombuffer(data, MI_DTYPES[dt_]).reshape(cdims[::-1]).astype(float)
            zs.close()
        off += 8 + n
    if todo:
        raise ValueError(f"variables not found: {sorted(todo)}")


def _wells_v5(url, wells):
    """MATLAB 5 (most 2016 recordings): only the LFP variable is transferred,
    one well at a time (sampling rate 1 kHz, as documented)."""
    for _, k, x in cells_v5(url, {"LFP": set(wells)}):
        yield k, 1000.0, x                                        # 64 x time


def extract(fn, out, wells):
    res = {}
    t0 = time.time()
    url = URL.format(fn)
    head = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-127"}),
                                  timeout=60).read()
    with fsspec.filesystem("http").open(url, "rb", block_size=2 ** 23,
                                        cache_type="readahead") as fh:
        source = _wells_v73(fh, wells) if b"MATLAB 7.3" in head else _wells_v5(url, wells)
        for w, fs, x in source:
            if x.shape[0] != 64:
                raise ValueError(f"{fn} well {w + 1}: unexpected shape {x.shape}")
            fr, p = welch(x - x.mean(1, keepdims=True), fs, window="hann",
                          nperseg=int(4 * fs), noverlap=int(2 * fs), axis=1)
            res[f"w{w}"] = p[:, fr <= 200].astype(np.float32)
            res[f"rms{w}"] = x.std(1)
            res[f"n{w}"] = x.shape[1]
            print(fn, "well", w + 1, x.shape, f"{time.time() - t0:.0f} s", flush=True)
    np.savez(out, f=fr[fr <= 200], fs=fs, **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results",
                                                  "trujillo_psd"))
    ap.add_argument("--wells", default="4,5,6,7,8,9,10,11")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    date = a.file.replace("LFP_Sp_", "").replace(".mat", "")
    extract(a.file, os.path.join(a.out, date), [int(w) for w in a.wells.split(",")])


if __name__ == "__main__":
    main()
