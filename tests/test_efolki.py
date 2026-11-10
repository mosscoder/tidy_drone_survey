"""0.4.0: eFOLKI (tidysurvey/efolki.py), the flow registration and the seam merge measure with. flows() and
EFOLKI_VERSION were added on the odm-fork branch (2026-11-10).

The port itself was checked against the authors' code in the audit (02_registration/matcher_ablation/poc/
check_port.py, cuda_check_synth.py); these tests pin what the engine relies on: a known shift is recovered to
a small fraction of a pixel, with the sign convention moving(x + u, y + v) = reference(x, y), the settings are
the adopted ones, and no-data is filled from the nearest valid pixel. Runs on whatever device efolki.device()
picks (CUDA, MPS or the CPU)."""
import unittest

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from tidysurvey import efolki as E


def texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    g = sum(gaussian_filter(rng.normal(size=(h, w)), s) / gaussian_filter(rng.normal(size=(h, w)), s).std()
            for s in (1.5, 4.0, 12.0))
    return ((g - g.min()) / (g.max() - g.min()) * 200 + 20).astype(np.float32)


class Flow(unittest.TestCase):
    def test_settings_are_the_adopted_ones(self):
        self.assertEqual(E.SETTINGS, dict(iteration=8, radius=(32, 24, 16, 8), rank=4, levels=3))
        self.assertEqual(E.REACH_PX, 256)

    def test_known_shift_is_recovered(self):
        """The moving image is the reference with its content moved 3.4 px east and 1.7 px north; eFOLKI
        reads u = +3.4, v = -1.7: moving(x + u, y + v) lines up with reference(x, y)."""
        ref = texture(768, 896)
        yy, xx = np.mgrid[0:768, 0:896].astype(np.float32)
        mov = cv2.remap(ref, xx - 3.4, yy + 1.7, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        u, v = E.flow(ref, mov)
        inner = np.s_[64:-64, 64:-64]
        self.assertAlmostEqual(float(np.median(u[inner])), 3.4, delta=0.1)
        self.assertAlmostEqual(float(np.median(v[inner])), -1.7, delta=0.1)
        self.assertLess(float(np.percentile(np.hypot(u[inner] - 3.4, v[inner] + 1.7), 95)), 0.25)

    def test_brightness_change_does_not_move_it(self):
        """Rank transforms: a monotonic change of brightness between dates reads as no shift."""
        ref = texture(512, 512, seed=2)
        u, v = E.flow(ref, np.sqrt(ref) * 9.0 + 5.0)
        self.assertLess(float(np.percentile(np.hypot(u, v)[32:-32, 32:-32], 99)), 0.05)

    def test_version_is_named(self):
        """Raise EFOLKI_VERSION when a change can alter a flow (see efolki.py); callers write it beside their results."""
        self.assertEqual(E.EFOLKI_VERSION, 1)

    def test_several_moving_images_give_flows_numbers(self):
        """flows() shares the reference's side between several moving images; each field is flow()'s, bit for bit on
        the CPU, whether the images go through one at a time or together."""
        import torch
        cpu = torch.device("cpu")
        ref = texture(224, 256, seed=3)
        yy, xx = np.mgrid[0:224, 0:256].astype(np.float32)
        movs = [cv2.remap(ref, xx - dx, yy - dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) * gain
                for dx, dy, gain in ((1.6, -0.8, 1.0), (-2.2, 1.1, 0.7), (0.4, 2.5, 1.3))]
        stock = [E.flow(ref, m, cpu) for m in movs]
        self.assertGreater(float(np.abs(stock[0][0]).max()), 0.5)               # real fields, not zeros
        for batch in (1, 2, 3):
            ours = E.flows(ref, movs, cpu, batch)
            self.assertEqual(len(ours), 3)
            for (u, v), (u0, v0) in zip(ours, stock):
                self.assertTrue(np.array_equal(u, u0) and np.array_equal(v, v0), f"batch {batch}")
        self.assertEqual(E.flows(ref, [], cpu), [])

    def test_many_at_once_helpers_are_the_one_image_ones(self):
        import torch
        g = torch.Generator().manual_seed(0)
        I = torch.rand((3, 1, 20, 24), generator=g)
        u = torch.rand((3, 1, 20, 24), generator=g) * 4 - 2
        v = torch.rand((3, 1, 20, 24), generator=g) * 4 - 2
        yy, xx = torch.meshgrid(torch.arange(20, dtype=I.dtype), torch.arange(24, dtype=I.dtype), indexing="ij")
        base = (xx[None, None], yy[None, None])
        warped = E._warp_many(I, u, v, base)
        for shape in ((40, 48), (39, 47), (41, 50)):
            down = E._pyr_down_many(u, shape)
            for k in range(3):
                self.assertTrue(torch.equal(down[k:k + 1], E._pyr_down(u[k:k + 1], shape)))
        for k in range(3):
            self.assertTrue(torch.equal(warped[k:k + 1], E._warp(I[k:k + 1], u[k:k + 1], v[k:k + 1], base)))

    def test_fill_nearest(self):
        g = np.arange(12, dtype=np.float32).reshape(3, 4)
        valid = np.zeros((3, 4), bool); valid[1, 1] = valid[1, 2] = True
        f = E.fill_nearest(g, valid)
        self.assertTrue(set(np.unique(f)) <= {5.0, 6.0})
        self.assertEqual(f[1, 1], 5.0); self.assertEqual(f[1, 2], 6.0)
        self.assertIs(E.fill_nearest(g, np.ones((3, 4), bool)), g)


if __name__ == "__main__":
    unittest.main()
