"""Scene-selection core: the pure chooser both Sentinel-2 selectors share.

Regression guard for the fall-2025 miscalibration. `pick_scene` read 0.0% AOI
cloud from one mission's clear corner and locked 2025-10-01 — a scene 46.5%
clouded over the whole survey — and the downloader then ranked purely by date
proximity and kept it, discarding the clean 2025-09-24. The shared chooser must
prefer clean-then-nearest and fall back to least-cloudy, so neither stage can
ever again select a scene it has itself measured as over the limit.
"""
from tidysurvey.sentinel import _choose_scene


def test_prefers_clean_over_nearer_cloudy():
    # nearest scene is cloudy; two clean scenes sit further out -> nearest CLEAN
    cands = [
        dict(date="2025-10-01", days_from_target=0, aoi_cloud_pct=46.5),
        dict(date="2025-09-24", days_from_target=-7, aoi_cloud_pct=0.0),
        dict(date="2025-09-29", days_from_target=-2, aoi_cloud_pct=10.8),
    ]
    assert _choose_scene(cands, 20.0)["date"] == "2025-09-29"


def test_nearest_among_clean():
    cands = [
        dict(date="2025-09-24", days_from_target=-7, aoi_cloud_pct=0.0),
        dict(date="2025-10-08", days_from_target=0, aoi_cloud_pct=5.0),
    ]
    assert _choose_scene(cands, 20.0)["date"] == "2025-10-08"


def test_all_cloudy_falls_back_to_least_cloudy():
    # none at/under the limit -> the single least-cloudy, regardless of distance
    cands = [
        dict(date="2025-10-01", days_from_target=0, aoi_cloud_pct=46.5),
        dict(date="2025-10-04", days_from_target=3, aoi_cloud_pct=94.4),
        dict(date="2025-10-14", days_from_target=13, aoi_cloud_pct=30.0),
    ]
    assert _choose_scene(cands, 20.0)["date"] == "2025-10-14"


def test_limit_is_inclusive():
    cands = [
        dict(date="2025-10-08", days_from_target=0, aoi_cloud_pct=20.0),
        dict(date="2025-10-06", days_from_target=-2, aoi_cloud_pct=21.0),
    ]
    assert _choose_scene(cands, 20.0)["date"] == "2025-10-08"


def test_fall_2025_regression():
    # the fall-2025 trap: 2025-10-01 is NEAREST to target but 46.5% clouded,
    # while earlier scenes are clean. The chooser must reject the cloudy-nearest
    # and return the nearest CLEAN scene — never 2025-10-01. (aoi_cloud_pct here
    # are illustrative selection inputs; the real values come from GEE at run
    # time, which is why the exact clean date is confirmed from the scene menu.)
    cands = [
        dict(date="2025-09-24", days_from_target=-14, aoi_cloud_pct=0.0),
        dict(date="2025-09-29", days_from_target=-9, aoi_cloud_pct=10.8),
        dict(date="2025-10-01", days_from_target=-7, aoi_cloud_pct=46.5),
        dict(date="2025-10-04", days_from_target=-4, aoi_cloud_pct=94.4),
    ]
    pick = _choose_scene(cands, 20.0)
    assert pick["date"] != "2025-10-01"          # the bug: cloudy-nearest
    assert pick["aoi_cloud_pct"] <= 20.0         # always returns a clean scene
    assert pick["date"] == "2025-09-29"          # nearest clean to target
