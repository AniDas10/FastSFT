"""Tier-1 unit tests for fastsft.eval.results -- the noise-floor margin, the
good/warn/info Finding bands, JSON serialization, and save/load round-trip."""

import json
import math

import pytest

from fastsft.eval.results import (
    SIM_MARGIN,
    WIN_MARGIN_SIGMAS,
    _caveat,
    _parent_likeness,
    _similarity,
    _tuned_vs_parent,
    _tuned_vs_untuned,
    _win_margin,
    interpret,
    load_results,
    results_as_json,
    save_results,
)

# --- _win_margin ---------------------------------------------------------


@pytest.mark.parametrize("num_prompts", [1, 10, 50, 100, 1000])
def test_win_margin_matches_formula(num_prompts):
    assert _win_margin(num_prompts) == pytest.approx(
        WIN_MARGIN_SIGMAS * math.sqrt(0.25 / num_prompts)
    )


def test_win_margin_shrinks_as_sample_grows():
    margins = [_win_margin(n) for n in (10, 100, 1000, 10000)]
    assert margins == sorted(margins, reverse=True)
    assert len(set(margins)) == len(margins)  # strictly decreasing


def test_win_margin_at_100_is_seven_and_a_half_percent():
    # 1.5 * sqrt(0.25/100) = 1.5 * 0.05 = 0.075
    assert _win_margin(100) == pytest.approx(0.075)


# --- _tuned_vs_untuned / _parent_likeness (same 3-band shape) -------------

# Both take (comparison, num_prompts) and split into good / warn(lost) /
# warn(indistinguishable) around 0.5 +/- margin.
THREE_BAND = [_tuned_vs_untuned, _parent_likeness]


@pytest.mark.parametrize("fn", THREE_BAND)
def test_three_band_good_above_upper_floor(fn):
    margin = _win_margin(100)
    finding = fn({"win_rate": 0.5 + margin + 0.05}, 100)
    assert finding.status == "good"


@pytest.mark.parametrize("fn", THREE_BAND)
def test_three_band_warn_below_lower_floor(fn):
    margin = _win_margin(100)
    finding = fn({"win_rate": 0.5 - margin - 0.05}, 100)
    assert finding.status == "warn"
    assert "LESS" in finding.message or "lost" in finding.message


@pytest.mark.parametrize("fn", THREE_BAND)
def test_three_band_warn_within_floor_is_indistinguishable(fn):
    finding = fn({"win_rate": 0.5}, 100)
    assert finding.status == "warn"
    # the within-floor branch is worded as no reliable signal, distinct from loss
    assert "lost" not in finding.message


@pytest.mark.parametrize("fn", THREE_BAND)
def test_three_band_upper_boundary_is_exclusive(fn):
    # rate exactly at 0.5 + margin is NOT > the floor -> within-floor warn.
    margin = _win_margin(100)
    finding = fn({"win_rate": 0.5 + margin}, 100)
    assert finding.status == "warn"


@pytest.mark.parametrize("fn", THREE_BAND)
@pytest.mark.parametrize("comparison,num_prompts", [(None, 100), ({"win_rate": 0.9}, 0)])
def test_three_band_none_when_missing_inputs(fn, comparison, num_prompts):
    assert fn(comparison, num_prompts) is None


# --- _tuned_vs_parent (good if within/above parity floor, else info) ------


def test_tuned_vs_parent_good_at_or_above_lower_floor():
    margin = _win_margin(100)
    # >= 0.5 - margin counts as competitive
    assert _tuned_vs_parent({"win_rate": 0.5 - margin}, 100).status == "good"
    assert _tuned_vs_parent({"win_rate": 0.5}, 100).status == "good"


def test_tuned_vs_parent_info_below_floor():
    margin = _win_margin(100)
    finding = _tuned_vs_parent({"win_rate": 0.5 - margin - 0.05}, 100)
    assert finding.status == "info"


@pytest.mark.parametrize("comparison,num_prompts", [(None, 100), ({"win_rate": 0.4}, 0)])
def test_tuned_vs_parent_none_when_missing_inputs(comparison, num_prompts):
    assert _tuned_vs_parent(comparison, num_prompts) is None


# --- _similarity (good / warn / info around SIM_MARGIN) -------------------


def test_similarity_good_when_tuned_clearly_closer():
    finding = _similarity({"tuned_vs_parent": 0.80, "untuned_vs_parent": 0.70})
    assert finding.status == "good"


def test_similarity_warn_when_tuned_clearly_farther():
    finding = _similarity({"tuned_vs_parent": 0.70, "untuned_vs_parent": 0.80})
    assert finding.status == "warn"


def test_similarity_info_within_margin():
    finding = _similarity(
        {"tuned_vs_parent": 0.70 + SIM_MARGIN / 2, "untuned_vs_parent": 0.70}
    )
    assert finding.status == "info"


@pytest.mark.parametrize(
    "similarity",
    [None, {}, {"tuned_vs_parent": 0.7}, {"untuned_vs_parent": 0.7},
     {"tuned_vs_parent": None, "untuned_vs_parent": 0.7}],
)
def test_similarity_none_when_incomplete(similarity):
    assert _similarity(similarity) is None


# --- _caveat / interpret / results_as_json -------------------------------


def test_caveat_is_always_info():
    caveat = _caveat()
    assert caveat.status == "info"
    assert caveat.message


def _full_results():
    return {
        "num_prompts": 100,
        "comparisons": {
            "tuned_vs_untuned": {"win_rate": 0.9},
            "parent_likeness": {"win_rate": 0.9},
            "tuned_vs_parent": {"win_rate": 0.2},
        },
        "similarity_to_parent": {"tuned_vs_parent": 0.8, "untuned_vs_parent": 0.7},
    }


def test_interpret_full_results_order_and_count():
    findings = interpret(_full_results())
    # three comparisons + similarity + caveat, in reading order.
    assert len(findings) == 5
    assert [f.status for f in findings] == ["good", "good", "info", "good", "info"]
    assert findings[-1] == _caveat()


def test_interpret_empty_results_is_only_the_caveat():
    findings = interpret({})
    assert len(findings) == 1
    assert findings[0].status == "info"
    assert findings[0] == _caveat()


def test_interpret_drops_missing_sections_but_keeps_caveat():
    # num_prompts present but no comparisons/similarity -> only the caveat.
    findings = interpret({"num_prompts": 100})
    assert findings == [_caveat()]


def test_results_as_json_roundtrips_and_embeds_findings():
    results = _full_results()
    parsed = json.loads(results_as_json(results))
    # original keys preserved
    assert parsed["num_prompts"] == 100
    assert parsed["comparisons"]["tuned_vs_untuned"]["win_rate"] == 0.9
    # findings appended, matching interpret()
    expected = [{"status": f.status, "message": f.message} for f in interpret(results)]
    assert parsed["findings"] == expected


def test_results_as_json_does_not_mutate_input():
    results = _full_results()
    results_as_json(results)
    assert "findings" not in results


# --- save_results / load_results round-trip ------------------------------


def test_save_load_roundtrip(tmp_path):
    results = _full_results()
    path = save_results(results, str(tmp_path))
    assert path.endswith(".json")
    assert load_results(str(tmp_path)) == results


def test_load_results_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(str(tmp_path))
