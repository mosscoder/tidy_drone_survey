"""Publish harden: a re-publish DELETES the prior run's work/ archive and
replaces it — one current archive per survey, never a superseded/stale backup
accumulating on the NAS.

Born from the 2024 summer re-publish: it first *crashed* on a pre-existing
work/ (the old `raise FileExistsError`), then a first fix wrongly kept a
`.superseded` backup (61 GB of stale, regenerable intermediates). This guards
the final behavior so neither regression returns.
"""
import os
import tempfile
import shutil
from pathlib import Path

import tidysurvey.cli as C


def _mk(p, txt):
    Path(p).mkdir(parents=True, exist_ok=True)
    (Path(p) / "f").write_text(txt)


def test_move_tree_deletes_stale_no_backup():
    T = tempfile.mkdtemp()
    try:
        # a prior run's archive already sits at dst; a re-publish must replace it
        _mk(f"{T}/src", "NEW"); _mk(f"{T}/dst", "STALE")
        C._move_tree(f"{T}/src", f"{T}/dst", log=lambda m: None)
        assert (Path(T) / "dst" / "f").read_text() == "NEW", "new archive not in place"
        assert not (Path(T) / "dst.superseded").exists(), "kept a backup — must delete"
        assert not (Path(T) / "src").exists(), "src not consumed"
        assert os.listdir(T) == ["dst"], f"stale leftovers: {os.listdir(T)}"

        # a second re-publish still leaves exactly one archive — no accumulation
        _mk(f"{T}/src", "NEWER")
        C._move_tree(f"{T}/src", f"{T}/dst", log=lambda m: None)
        assert (Path(T) / "dst" / "f").read_text() == "NEWER"
        assert os.listdir(T) == ["dst"], "archives accumulated across re-publishes"
        print("PASS  delete-on-supersede: no backup, no accumulation")
    finally:
        shutil.rmtree(T, ignore_errors=True)


def test_move_tree_clean_dst_normal_move():
    T = tempfile.mkdtemp()
    try:
        _mk(f"{T}/src", "X")
        r = C._move_tree(f"{T}/src", f"{T}/dst", log=lambda m: None)
        assert (Path(T) / "dst" / "f").read_text() == "X"
        assert not (Path(T) / "dst.superseded").exists()
        assert not (Path(T) / "src").exists()
        print(f"PASS  clean dst: normal move ({r})")
    finally:
        shutil.rmtree(T, ignore_errors=True)


def test_move_tree_missing_src_is_idempotent():
    # resume re-entry: src already moved on a prior attempt — must not error
    T = tempfile.mkdtemp()
    try:
        _mk(f"{T}/dst", "ALREADY")
        assert C._move_tree(f"{T}/src", f"{T}/dst", log=lambda m: None) == "already"
        _mk_missing = C._move_tree(f"{T}/nope", f"{T}/nope_dst", log=lambda m: None)
        assert _mk_missing == "missing"
        print("PASS  missing src: idempotent resume (already / missing)")
    finally:
        shutil.rmtree(T, ignore_errors=True)


if __name__ == "__main__":
    test_move_tree_deletes_stale_no_backup()
    test_move_tree_clean_dst_normal_move()
    test_move_tree_missing_src_is_idempotent()
    print("\nALL PUBLISH-HARDEN CHECKS PASSED")
