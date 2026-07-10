"""Subprocess entry for one dense-match chunk of register_survey_dense.

Run as:  python -m tidysurvey._regchunk <work_dir> <chunk_idx>

A fresh process per chunk resets the MPS allocator/kernel-cache decay that
grows with varied-shape LoFTR calls in a long-lived process (audit
02_registration §7). The parent orchestrates; this does one chunk and exits.
"""
import sys

from .registration import _run_reg_chunk

if __name__ == "__main__":
    _run_reg_chunk(sys.argv[1], sys.argv[2])
