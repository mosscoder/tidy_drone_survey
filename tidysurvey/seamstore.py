"""The seam measurements handed between machines (0.5.0).

A stitch measures each pair of flights with eFOLKI in a box around their faultline (merge.seam_merge). With a work
folder the stitch splits that out: it exports every box's two grey images to the folder, any number of workers
measure them, and the stitch renders with the measured offsets. The method and its numbers do not change: a worker
runs the same eFOLKI on the same two arrays the stitch would have, and the render uses its offsets as they come. Only
the device can differ (CUDA, Apple MPS or the CPU), which moves an offset by about 1e-3 px; each flow records it.

The folder is a local directory or a gs:// prefix, nothing else is needed. Layout:
    boxes/<key>.npz    the two grey images (float32) and where the box sits
    claims/<key>       which worker is measuring it, and since when
    flows/<key>.npz    the offsets on the part of the box the stitch keeps, the device and the time
    index.json         every key, written when the export is complete
A worker claims a box by creating its claim, which only one worker can do (O_EXCL on a disk, ifGenerationMatch=0 in a
bucket). A claim older than the lease with no flow is taken over, so a worker that dies costs one lease.

    python -m tidysurvey.seamstore measure <folder> [--device cpu|cuda|mps] [--max-minutes M] [--idle-exit S]
"""
from __future__ import annotations
import io
import json
import os
import random
import socket
import sys
import time
import uuid
from pathlib import Path

import numpy as np

LEASE_S = 600                                   # a claim this old with no flow is taken over


def _now() -> float:
    return time.time()


# ---- backends ----------------------------------------------------------------------------------------
class _Local:
    def __init__(self, root: str):
        self.root = Path(root)

    def _p(self, rel: str) -> Path:
        return self.root / rel

    def list(self, prefix: str) -> list[str]:
        d = self._p(prefix)
        return sorted(p.name for p in d.iterdir() if p.is_file() and not p.name.startswith(".")) if d.is_dir() else []

    def read(self, rel: str) -> bytes | None:
        try:
            return self._p(rel).read_bytes()
        except FileNotFoundError:
            return None

    def write(self, rel: str, data: bytes) -> None:
        p = self._p(rel); p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}")
        tmp.write_bytes(data); os.replace(tmp, p)

    def create(self, rel: str, data: bytes) -> bool:
        p = self._p(rel); p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return True

    def take_over(self, rel: str, data: bytes) -> bool:
        """Replace a stale claim: rename it away (atomic, one winner), then create afresh."""
        p = self._p(rel)
        try:
            os.rename(p, p.with_name(f".{p.name}.stale-{uuid.uuid4().hex}"))
        except FileNotFoundError:
            pass
        return self.create(rel, data)

    def delete(self, rel: str) -> None:
        self._p(rel).unlink(missing_ok=True)

    def clear(self) -> None:
        import shutil
        for sub in ("boxes", "claims", "flows"):
            shutil.rmtree(self._p(sub), ignore_errors=True)
        self._p("index.json").unlink(missing_ok=True)


class _GCS:
    def __init__(self, root: str):
        from google.cloud import storage
        bucket, _, prefix = root[5:].partition("/")
        self.bucket = storage.Client().bucket(bucket); self.prefix = prefix.rstrip("/")

    def _n(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def list(self, prefix: str) -> list[str]:
        base = self._n(prefix).rstrip("/") + "/"
        return sorted(b.name[len(base):] for b in self.bucket.client.list_blobs(self.bucket, prefix=base) if "/" not in b.name[len(base):])

    def read(self, rel: str) -> bytes | None:
        from google.api_core.exceptions import NotFound
        try:
            return self.bucket.blob(self._n(rel)).download_as_bytes()
        except NotFound:
            return None

    def write(self, rel: str, data: bytes) -> None:
        b = self.bucket.blob(self._n(rel)); b.cache_control = "no-cache, max-age=0"
        b.upload_from_string(data, content_type="application/octet-stream")

    def create(self, rel: str, data: bytes) -> bool:
        from google.api_core.exceptions import PreconditionFailed
        b = self.bucket.blob(self._n(rel)); b.cache_control = "no-cache, max-age=0"
        try:
            b.upload_from_string(data, content_type="application/json", if_generation_match=0)
            return True
        except PreconditionFailed:
            return False

    def take_over(self, rel: str, data: bytes) -> bool:
        from google.api_core.exceptions import NotFound, PreconditionFailed
        b = self.bucket.get_blob(self._n(rel))
        if b is None:
            return self.create(rel, data)
        nb = self.bucket.blob(self._n(rel)); nb.cache_control = "no-cache, max-age=0"
        try:
            nb.upload_from_string(data, content_type="application/json", if_generation_match=b.generation)
            return True
        except (PreconditionFailed, NotFound):
            return False

    def delete(self, rel: str) -> None:
        from google.api_core.exceptions import NotFound
        try:
            self.bucket.blob(self._n(rel)).delete()
        except NotFound:
            pass

    def clear(self) -> None:
        for sub in ("boxes", "claims", "flows"):
            base = self._n(sub) + "/"
            for b in self.bucket.client.list_blobs(self.bucket, prefix=base):
                try:
                    b.delete()
                except Exception:
                    pass
        self.delete("index.json")


# ---- the store -----------------------------------------------------------------------------------------
def _pack(arrays: dict, meta: dict) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, meta=np.array(json.dumps(meta)), **arrays)
    return buf.getvalue()


def _unpack(data: bytes) -> tuple[dict, dict]:
    z = np.load(io.BytesIO(data), allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return {k: z[k] for k in z.files if k != "meta"}, meta


class SeamStore:
    """The work folder of one stitch: boxes out, flows back."""

    def __init__(self, root: str):
        self.root = str(root).rstrip("/")
        self.b = _GCS(self.root) if self.root.startswith("gs://") else _Local(self.root)

    # -- the stitch's side
    def put_box(self, key: str, ga: np.ndarray, gb: np.ndarray, meta: dict) -> None:
        self.b.write(f"boxes/{key}.npz", _pack(dict(ga=ga, gb=gb), meta))

    def finish_export(self, keys: list[str], info: dict) -> None:
        self.b.write("index.json", json.dumps(dict(keys=sorted(keys), **info)).encode())

    def index(self) -> dict | None:
        data = self.b.read("index.json")
        return json.loads(data) if data else None

    def box_keys(self) -> set[str]:
        return {n[:-4] for n in self.b.list("boxes") if n.endswith(".npz")}

    def flow_keys(self) -> set[str]:
        return {n[:-4] for n in self.b.list("flows") if n.endswith(".npz")}

    def get_flow(self, key: str) -> tuple[np.ndarray, np.ndarray, dict] | None:
        data = self.b.read(f"flows/{key}.npz")
        if data is None:
            return None
        arr, meta = _unpack(data)
        return arr["u"], arr["v"], meta

    def clear(self) -> None:
        self.b.clear()

    # -- a worker's side
    def get_box(self, key: str):
        data = self.b.read(f"boxes/{key}.npz")
        if data is None:
            return None
        arr, meta = _unpack(data)
        return arr["ga"], arr["gb"], meta

    def put_flow(self, key: str, u: np.ndarray, v: np.ndarray, meta: dict) -> None:
        self.b.write(f"flows/{key}.npz", _pack(dict(u=u, v=v), meta))

    def claim(self, key: str, worker: str, lease_s: float = LEASE_S) -> bool:
        data = json.dumps(dict(worker=worker, t=_now())).encode()
        if self.b.create(f"claims/{key}", data):
            return True
        old = self.b.read(f"claims/{key}")
        try:
            t = json.loads(old)["t"] if old else 0.0
        except Exception:
            t = 0.0
        if _now() - t > lease_s:                                  # its worker died: take it over
            return self.b.take_over(f"claims/{key}", data)
        return False

    def release(self, key: str) -> None:
        self.b.delete(f"claims/{key}")

    def claimed(self) -> set[str]:
        return set(self.b.list("claims"))


def worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def measure(root: str, device: str | None = None, max_seconds: float | None = None, idle_exit_s: float | None = None,
            lease_s: float = LEASE_S, poll_s: float = 5.0, log=print) -> dict:
    """Measure boxes until none is left: claim one, run eFOLKI on its two grey images exactly as the stitch would,
    keep the part of the flow the stitch uses, write it, take the next. Waits for boxes while the export runs.
    Stops when the export is complete and every box has a flow, after `max_seconds`, or after `idle_exit_s` with
    nothing to claim once the export is complete (a remote worker leaves the last boxes to whoever holds them)."""
    from . import efolki as E
    store = SeamStore(root); wid = worker_id()
    dev = E.device(device)
    t0 = _now(); n = 0; spent = 0.0; idle_since = None
    log(f"seams worker {wid} on {dev} measuring {root}")
    while True:
        if max_seconds and _now() - t0 > max_seconds:
            log(f"seams worker {wid}: time is up after {n} boxes"); break
        idx = store.index()
        flows = store.flow_keys()
        todo = sorted(store.box_keys() - flows)
        random.shuffle(todo)
        did = False
        for key in todo:
            if max_seconds and _now() - t0 > max_seconds:
                break
            if not store.claim(key, wid, lease_s):
                continue
            if key in store.flow_keys():                          # finished while we looked
                store.release(key); continue
            got = store.get_box(key)
            if got is None:
                store.release(key); continue
            ga, gb, meta = got
            t = _now()
            u, v = E.flow(ga, gb, dev=dev)
            r0, r1, c0, c1 = meta["keep"]
            sec = round(_now() - t, 3); spent += sec
            store.put_flow(key, np.ascontiguousarray(u[r0:r1, c0:c1]), np.ascontiguousarray(v[r0:r1, c0:c1]),
                           dict(meta, device=str(dev), seconds=sec, worker=wid))
            store.release(key)
            n += 1; did = True
        if idx is not None and set(idx["keys"]) <= store.flow_keys():
            log(f"seams worker {wid}: all {len(idx['keys'])} boxes measured; {n} by this worker"); break
        if did:
            idle_since = None
            continue
        if idx is not None and idle_exit_s is not None:
            idle_since = idle_since or _now()
            if _now() - idle_since > idle_exit_s:
                log(f"seams worker {wid}: nothing left to claim; leaving after {n} boxes"); break
        time.sleep(poll_s)
    return dict(worker=wid, device=str(dev), boxes=n, flow_seconds=round(spent, 1), seconds=round(_now() - t0, 1))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m tidysurvey.seamstore")
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("measure", help="measure seam boxes from a work folder until none is left")
    m.add_argument("folder"); m.add_argument("--device"); m.add_argument("--max-minutes", type=float)
    m.add_argument("--idle-exit", type=float, help="seconds with nothing to claim, once the export is complete, before leaving")
    a = ap.parse_args(argv)
    res = measure(a.folder, device=a.device, max_seconds=a.max_minutes * 60 if a.max_minutes else None,
                  idle_exit_s=a.idle_exit, log=lambda s: print(s, flush=True))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
