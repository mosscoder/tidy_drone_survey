"""eFOLKI's window sums by cumulative sums on a CPU give the convolution's numbers (0.4.2).

On the season VM the 65-tap convolutions took 47.7 s per seam box and the cumulative sums 24.5 s, so _box uses
them on a CPU. The two must agree on window means of rank values, and a whole flow on the CPU must give the
convolution's flow to well under a hundredth of a pixel."""
import unittest

import numpy as np
import torch

import tidysurvey.efolki as E
from test_register_efolki import texture


def _conv_box(I, rad):
    return E._sep(I, torch.full((2 * rad + 1,), 1.0 / (2 * rad + 1)))


class CumulativeSums(unittest.TestCase):
    def test_window_means_agree(self):
        x = torch.rand(1, 1, 700, 900) * 80                       # rank values are 0 to 80
        for rad in (8, 16, 24, 32):
            d = (E._box(x, rad) - _conv_box(x, rad)).abs().max().item()
            self.assertLess(d, 1e-4, f"radius {rad}: {d}")

    def test_cpu_flow_matches_the_convolution(self):
        g = (texture(600, 700, seed=3) * 200).astype(np.float32)
        yy, xx = np.mgrid[0:600, 0:700].astype(np.float32)
        import cv2
        m = cv2.remap(g, xx - 2.5, yy + 1.5, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        u, v = E.flow(g, m, dev="cpu")
        box = E._box
        E._box = _conv_box
        try:
            u2, v2 = E.flow(g, m, dev="cpu")
        finally:
            E._box = box
        d = max(np.abs(u - u2).max(), np.abs(v - v2).max())
        self.assertLess(d, 1e-3, f"flows differ by {d} px")
        core = np.s_[100:-100, 100:-100]
        self.assertAlmostEqual(float(np.median(u[core])), 2.5, delta=0.1)
        self.assertAlmostEqual(float(np.median(v[core])), -1.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
