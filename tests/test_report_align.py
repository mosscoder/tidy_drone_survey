"""The quality report's align table: a registered mission's row says what the registration did; a mission that
arrives with its own offset check (drawn from one camera solve, never moved) has a row of what was measured."""
from tidysurvey.report import _align_section


def test_registered_missions_keep_their_table():
    html = _align_section("align_ms", dict(kind="align", missions=[
        dict(name="01", r_before=0.71, r_after=0.89, move_median_cm=12.0, moved_over_50cm_ha=0.4, look=False)]))
    assert "r before → after" in html and "0.71 → 0.89" in html and "<td class=num>12.0</td>" in html
    assert "offset" not in html


def test_missions_with_an_offset_check_get_their_own_table():
    fork = dict(name="06", check="offset", offset_median_cm=0.6, offset_p90_cm=1.2, tiles=12288, look=False)
    html = _align_section("align_ms", dict(kind="align", missions=[fork]))
    assert "Not moved." in html and "offset, median (cm)" in html
    assert "<td class=num>0.6</td>" in html and "<td class=num>1.2</td>" in html and "12,288" in html
    assert "r before" not in html and "median move" not in html          # no empty table of moves beside it
    both = _align_section("align_ms", dict(kind="align", missions=[
        dict(name="01", r_before=0.71, r_after=0.89, move_median_cm=12.0, moved_over_50cm_ha=0.4), fork]))
    assert both.count("<table>") == 2 and both.index("r before") < both.index("Not moved.")


def test_a_check_that_did_not_run_is_marked_for_a_look():
    html = _align_section("align_ms", dict(kind="align", missions=[dict(name="07", check="offset", look=True, tiles=0)]))
    assert "⚠ look" in html and "not measured" in html


def test_an_empty_stage_still_has_its_table():
    assert "r before → after" in _align_section("align_ms", dict(kind="align", missions=[]))
