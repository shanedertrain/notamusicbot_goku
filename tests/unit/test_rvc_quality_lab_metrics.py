"""Pure-metric, ranking and report tests for tools/rvc_quality_lab.py (no GPU, no torch)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

LAB_PATH = Path(__file__).resolve().parents[2] / "tools" / "rvc_quality_lab.py"
_spec = importlib.util.spec_from_file_location("rvc_quality_lab", LAB_PATH)
lab = importlib.util.module_from_spec(_spec)
sys.modules["rvc_quality_lab"] = lab
_spec.loader.exec_module(lab)

PLAN_FIELDS = {
    "f0_method",
    "pitch_shift",
    "use_index",
    "index_rate",
    "index_nprobe",
    "protect",
    "rms_mix_rate",
    "f0_post",
    "auto_pitch",
}


def _track(n: int = 300, hz: float = 110.0) -> np.ndarray:
    return np.full(n, hz, dtype=np.float64)


# -- f0 metrics ---------------------------------------------------------------------------


def test_octave_error_detected_on_injected_jump() -> None:
    ref = _track()
    out = ref.copy()
    out[100:130] *= 2.0  # +12 st for 30 of 300 frames
    assert lab.f0_octave_err_rate(out, ref) == pytest.approx(30 / 300)
    assert lab.f0_octave_err_rate(ref, ref) == 0.0


def test_octave_error_ignores_genuine_pitch_difference() -> None:
    ref = _track()
    out = ref * 2 ** (9 / 12)  # 9 st off: not near an octave
    assert lab.f0_octave_err_rate(out, ref) == 0.0
    out2 = ref * 2 ** (6 / 12)  # below the 8 st floor
    assert lab.f0_octave_err_rate(out2, ref) == 0.0


def test_octave_error_down_and_double_octave() -> None:
    ref = _track()
    out = ref.copy()
    out[:50] /= 2.0
    out[50:80] *= 4.0
    assert lab.f0_octave_err_rate(out, ref) == pytest.approx(80 / 300)


def test_octave_error_only_counts_commonly_voiced() -> None:
    ref = _track(100)
    out = ref.copy()
    out[:50] = 0.0
    out[50:60] *= 2
    assert lab.f0_octave_err_rate(out, ref) == pytest.approx(10 / 50)
    assert math.isnan(lab.f0_octave_err_rate(np.zeros(10), ref[:10]))


def test_jump_rate_flat_vs_glitchy() -> None:
    flat = _track()
    assert lab.f0_jump_rate(flat) == 0.0
    glitchy = flat.copy()
    glitchy[100] *= 2.0  # one-frame spike: two jumps
    rate = lab.f0_jump_rate(glitchy)
    assert rate == pytest.approx(2 / (300 * 0.01))
    assert math.isnan(lab.f0_jump_rate(np.zeros(50)))


def test_jump_rate_ignores_gaps() -> None:
    f = _track(100)
    f[40:50] = 0.0
    f[50:] *= 2.0  # big step across an unvoiced gap: not a frame-to-frame jump
    assert lab.f0_jump_rate(f) == 0.0


def test_smoothness_constant_vs_noisy() -> None:
    rng = np.random.default_rng(0)
    assert lab.f0_smooth_cents(_track()) == pytest.approx(0.0, abs=1e-6)
    ramp = 110.0 * 2 ** (np.arange(300) / 300.0)  # linear in log f0: second diff 0
    assert lab.f0_smooth_cents(ramp) == pytest.approx(0.0, abs=1e-6)
    noisy = _track() * 2 ** (rng.normal(0, 0.5, 300) / 12)
    assert lab.f0_smooth_cents(noisy) > 20.0
    assert math.isnan(lab.f0_smooth_cents(np.zeros(50)))


def test_smoothness_respects_unvoiced_breaks() -> None:
    f = np.concatenate([_track(20, 100), np.zeros(5), _track(20, 200)])
    assert lab.f0_smooth_cents(f) == pytest.approx(0.0, abs=1e-6)


def test_track_corr_and_median_register() -> None:
    t = np.linspace(0, 6, 300)
    ref = 110.0 * 2 ** (np.sin(t) / 4)
    assert lab.f0_track_corr(ref * 2, ref) == pytest.approx(1.0)
    assert lab.f0_track_corr(ref[::-1], ref) < 0.5
    assert math.isnan(lab.f0_track_corr(_track(), _track()))  # zero variance
    assert lab.f0_median_hz(np.array([0, 100, 0, 120, 140.0])) == pytest.approx(120.0)
    assert math.isnan(lab.f0_median_hz(np.zeros(5)))
    assert lab.register_err_st(216.0, 108.0) == pytest.approx(12.0)
    assert lab.register_err_st(54.0, 108.0) == pytest.approx(12.0)
    assert math.isnan(lab.register_err_st(float("nan"), 108.0))


def test_voiced_frac_ratio_and_reference_shift() -> None:
    src = _track(100)
    out = src.copy()
    out[:30] = 0.0
    assert lab.voiced_frac_ratio(out, src) == pytest.approx(0.7)
    shifted = lab.source_f0_ref(src, -12)
    assert shifted[0] == pytest.approx(55.0)
    m = lab.f0_metrics(src * 0.5, src, -12, 55.0)
    assert m["f0_octave_err_rate"] == 0.0
    assert m["register_err_st"] == pytest.approx(0.0, abs=1e-9)
    assert m["f0_track_corr"] != m["f0_track_corr"] or m["f0_track_corr"] > 0.99  # nan or 1


# -- audio metrics --------------------------------------------------------------------------


def test_hiss_ratio_noise_vs_tone() -> None:
    sr = 48000
    n = sr
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.1, n).astype(np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)
    unvoiced = np.zeros(n // 480 + 2, dtype=bool)  # every frame unvoiced
    hn = lab.hf_hiss_metrics(noise, sr, unvoiced)
    ht = lab.hf_hiss_metrics(tone, sr, unvoiced)
    assert hn["hf_hiss_ratio"] > 0.3  # white noise: ~ (24-12)/24 of the energy
    assert ht["hf_hiss_ratio"] < 0.001
    assert hn["unvoiced_flatness"] > ht["unvoiced_flatness"]


def test_hiss_ratio_uses_only_unvoiced_frames() -> None:
    sr = 48000
    n = sr
    rng = np.random.default_rng(2)
    y = (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)
    y[n // 2 :] = rng.normal(0, 0.1, n - n // 2)
    frames = n // 480 + 2
    voiced = np.zeros(frames, dtype=bool)
    voiced[: frames // 2 - 2] = True  # first half voiced (tone), second half noise unvoiced
    assert lab.hf_hiss_metrics(y, sr, voiced)["hf_hiss_ratio"] > 0.3
    assert lab.hf_hiss_metrics(y, sr, ~voiced)["hf_hiss_ratio"] < 0.3
    all_voiced = np.ones(frames, dtype=bool)
    assert math.isnan(lab.hf_hiss_metrics(y, sr, all_voiced)["hf_hiss_ratio"])


def test_sanity_flags() -> None:
    y = np.array([0.0, 1.0, -1.0, 0.5, np.nan], dtype=np.float32)
    s = lab.sanity(y)
    assert s["nan"] == 1.0 and s["peak"] == 1.0 and s["clip_frac"] == pytest.approx(2 / 5)


def test_mel_dist_zero_for_identical_and_lag_invariant() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(16000 * 2) / 16000
    x = (
        np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)
        + 0.01 * rng.normal(size=t.size)
    ).astype(np.float32)
    assert lab.mel_dist_db(x, x) == pytest.approx(0.0, abs=1e-6)
    delayed = np.concatenate([np.zeros(800, np.float32), x])[: len(x)]  # 5 frames
    assert lab.mel_dist_db(x, delayed) < lab.mel_dist_db(x, np.roll(x, 8000))
    assert lab.mel_dist_db(x, 0.01 * rng.normal(size=x.size).astype(np.float32)) > 5.0


def test_join_ratio_detects_a_click() -> None:
    sr = 48000
    y = (0.2 * np.sin(2 * np.pi * 150 * np.arange(sr) / sr)).astype(np.float64)
    clean = lab.join_ratio(y, [24000])["join_max"]
    y2 = y.copy()
    y2[24000:] += 0.3  # step discontinuity
    assert lab.join_ratio(y2, [24000])["join_max"] > 10 * clean
    assert lab.join_ratio(y, []) == {}


def test_stream_join_positions() -> None:
    j = lab.stream_join_positions(48000 * 2, 100, 320)
    assert j[0] == 2 * 100 * 24
    assert j[1] - j[0] == 2 * 320 * 24


def test_knn_mean_l2_takes_root_of_squared_distances() -> None:
    assert lab.knn_mean_l2(np.array([[4.0, 9.0]])) == pytest.approx(2.5)
    assert lab.knn_mean_l2(np.array([[-1e-9]])) == 0.0


# -- ranking ----------------------------------------------------------------------------------


def _row(name, oe, jr, rd, vr=1.0, hiss=0.01, mode="whole", status="ok", **extra):
    mean = {
        "f0_octave_err_rate": oe,
        "f0_jump_rate": jr,
        "retrieval_dist": rd,
        "voiced_frac_ratio": vr,
        "hf_hiss_ratio": hiss,
        "f0_smooth_cents": 10.0,
        "f0_track_corr": 0.9,
        "register_err_st": 1.0,
        "mel_dist_ref_db": 3.0,
        "rtf": 0.2,
    }
    mean.update(extra)
    return {"model": "Obama", "variant": name, "mode": mode, "status": status, "mean": mean}


def test_best_overall_is_lowest_rank_average() -> None:
    rows = [
        _row("a", 0.05, 5.0, 10.0),
        _row("b", 0.01, 2.0, 8.0),  # best on all three
        _row("c", 0.02, 3.0, 9.0),
        _row("d", 0.03, 4.0, 9.5),
        _row("e", 0.04, 4.5, 9.7),
    ]
    flags = lab.compute_flags(rows)
    assert "BEST_OVERALL" in flags["b"]
    assert all("BEST_OVERALL" not in flags[v] for v in "acde")
    assert "BEST_f0_octave_err_rate" in flags["b"]


def test_guards_exclude_dropout_and_hiss_and_skipped() -> None:
    rows = [
        _row("dropout", 0.0, 0.0, 1.0, vr=0.5),  # best numbers but dropouts
        _row("hissy", 0.0, 0.0, 1.0, hiss=0.5),
        _row("skipped", 0.0, 0.0, 1.0, status="skipped"),
        _row("ok1", 0.02, 2.0, 9.0, hiss=0.010),
        _row("ok2", 0.03, 3.0, 8.0, hiss=0.011),
        _row("ok3", 0.04, 4.0, 7.0, hiss=0.012),
    ]
    flags = lab.compute_flags(rows)
    for bad in ("dropout", "hissy", "skipped"):
        assert "BEST_OVERALL" not in flags[bad]
    winners = [v for v, f in flags.items() if "BEST_OVERALL" in f]
    assert len(winners) == 1 and winners[0] in {"ok1", "ok2", "ok3"}
    elig = {r["variant"] for r in lab.eligible_rows(rows)}
    assert elig == {"ok1", "ok2", "ok3"}


def test_stream_rows_ranked_separately() -> None:
    rows = [
        _row("w1", 0.01, 1.0, 5.0),
        _row("w2", 0.02, 2.0, 6.0),
        _row("s1", 0.5, 9.0, 9.0, mode="stream"),
        _row("s2", 0.6, 9.5, 9.5, mode="stream"),
    ]
    flags = lab.compute_flags(rows)
    assert "BEST_OVERALL" in flags["w1"]
    assert "BEST_STREAM" in flags["s1"]
    assert "BEST_OVERALL" not in flags["s1"]


def test_higher_is_better_metric_flag() -> None:
    rows = [_row("a", 0.1, 1, 1, f0_track_corr=0.7), _row("b", 0.1, 1, 1, f0_track_corr=0.95)]
    flags = lab.compute_flags(rows)
    assert "BEST_f0_track_corr" in flags["b"] and "BEST_f0_track_corr" not in flags["a"]


def test_mel_ref_best_excludes_reference_itself() -> None:
    ref = _row(lab.REF_VARIANT, 0.1, 1, 1, mel_dist_ref_db=0.0)
    other = _row("x", 0.1, 1, 1, mel_dist_ref_db=4.0)
    flags = lab.compute_flags([ref, other])
    assert "BEST_mel_dist_ref_db" in flags["x"]
    assert "BEST_mel_dist_ref_db" not in flags[lab.REF_VARIANT]


# -- variant table ----------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["Obama", "Goku"])
def test_variant_table_well_formed(model: str) -> None:
    table = lab.build_variants(model, base_shift=-3)
    ids = list(table)
    assert len(ids) == len(set(ids))
    assert "v00_baseline" in table and lab.REF_VARIANT in table
    for vid, v in table.items():
        assert v.id == vid
        assert set(v.fields) <= PLAN_FIELDS, (vid, set(v.fields) - PLAN_FIELDS)
        assert set(v.stream) <= lab.STREAM_FIELDS, (vid, set(v.stream) - lab.STREAM_FIELDS)
        assert v.mode in ("whole", "stream")
        assert v.source in lab.SPEAKERS
        assert (v.mode == "stream") == bool(v.stream)
        post = v.fields.get("f0_post") or v.stream.get("f0_post")
        if post:
            assert set(post) <= lab.F0_POST_KEYS
        if "auto_pitch" in v.fields:
            assert set(v.fields["auto_pitch"]) == {"target_median_hz", "max_abs", "strength"}
            assert v.fields["auto_pitch"]["target_median_hz"] == lab.TARGET_HZ[model]


def test_plan_stage_defaults_obama() -> None:
    t = lab.build_variants("Obama", base_shift=-4)
    assert [v for v in t if t[v].stage == "0"] == ["v00_baseline", lab.REF_VARIANT]
    assert t["v00_baseline"].fields == {"f0_method": "pm"}
    a4 = t["a4_rmvpe_post_idx05"].fields
    assert a4["f0_method"] == "rmvpe" and a4["index_rate"] == 0.5 and a4["use_index"] is True
    assert a4["f0_post"]["median_width"] == 5 and a4["f0_post"]["fill_gap_frames"] == 2
    assert t["b1_shift_fixed_m3"].fields["pitch_shift"] == -3
    assert t["b3_shift_fixed_m5"].fields["pitch_shift"] == -5
    assert t["b4_autoshift"].fields["auto_pitch"]["strength"] == 1.0
    assert t["b5_autoshift_half"].fields["auto_pitch"]["strength"] == 0.5
    assert t["c1_idx03"].fields["pitch_shift"] == -4  # carried
    assert t["c1_idx03"].fields["index_rate"] == 0.3
    assert t["c9_fp32"].env == {"RVC_WORKER_FP32": "1"}
    assert t["d1_aiden"].source == "aiden" and "pitch_shift" not in t["d1_aiden"].fields
    e = t["e4_stream_rmvpe_ctx2000"]
    assert e.mode == "stream" and e.stream["f0_algo"] == "rmvpe" and e.stream["context_ms"] == 2000
    assert "rms_mix_rate" not in e.stream and e.stream["pitch_shift"] == -4
    assert "f0_post" not in t["e2_stream_pm_ctx2000"].stream
    # stage A base must not leak the carried shift into B
    assert "pitch_shift" not in t["a4_rmvpe_post_idx05"].fields


def test_goku_reduced_matrix() -> None:
    t = lab.build_variants("Goku")
    assert set(t) >= {"v00_baseline", "a1_pm_idx05", "a4_rmvpe_post_idx05", "a5_harvest_idx05"}
    assert "a2_rmvpe" not in t and "a3_rmvpe_post" not in t
    assert t["b1_shift_fixed_0"].fields["pitch_shift"] == 0
    assert t["b2_shift_fixed_p2"].fields["pitch_shift"] == 2
    assert [v for v in t if t[v].stage == "E"] == ["e4_stream_rmvpe_ctx2000"]
    assert not any(v.stage == "D" for v in t.values())
    assert t["b4_autoshift"].fields["auto_pitch"]["target_median_hz"] == 145.0


def test_stage_ids_and_base_variant_fallback() -> None:
    t = lab.build_variants("Obama", base_id="a3_rmvpe_post", base_shift=-3)
    assert t["c1_idx03"].fields["f0_post"]["median_width"] == 5
    assert lab.stage_ids(t, "E") == [v for v in t if t[v].stage == "E"]
    assert lab.stage_ids(t, "all") == list(t)
    t2 = lab.build_variants("Obama", base_id="nonsense")
    assert "c1_idx03" in t2


def test_parse_clips() -> None:
    assert lab.parse_clips("0..5") == [0, 1, 2, 3, 4, 5]
    assert lab.parse_clips("0,2,3") == [0, 2, 3]
    with pytest.raises(ValueError):
        lab.parse_clips("0..6")


# -- results / index.md ----------------------------------------------------------------------


def test_merge_results_replaces_by_model_variant() -> None:
    old = [{"model": "Obama", "variant": "a", "x": 1}, {"model": "Goku", "variant": "a", "x": 2}]
    new = [{"model": "Obama", "variant": "a", "x": 3}, {"model": "Obama", "variant": "b", "x": 4}]
    merged = lab.merge_results(old, new)
    assert [(r["model"], r["variant"], r["x"]) for r in merged] == [
        ("Obama", "a", 3),
        ("Goku", "a", 2),
        ("Obama", "b", 4),
    ]


def test_write_outputs_and_index_md(tmp_path: Path) -> None:
    good = _row("a4_x", 0.01, 1.0, 5.0, retrieval_dist_src=12.0)
    good.update(
        stage="A",
        source="ryan",
        settings={
            "fields": {"f0_method": "rmvpe", "f0_post": {"median_width": 5}},
            "stream": {},
            "env": {},
        },
        wavs=[f"/abs/s{i}.wav" for i in range(6)],
    )
    skipped = {
        "model": "Obama",
        "variant": "a2",
        "stage": "A",
        "mode": "whole",
        "status": "skipped",
        "reason": "worker.convert has no 'use_index' yet",
        "settings": {"fields": {}, "stream": {}, "env": {}},
    }
    rows = lab.write_outputs(tmp_path, [good, skipped])
    # idempotent re-run: replacing keeps one row per variant
    rows = lab.write_outputs(tmp_path, [good])
    assert len(rows) == 2
    data = json.loads((tmp_path / "results.json").read_text())
    assert {r["variant"] for r in data} == {"a4_x", "a2"}
    md = (tmp_path / "index.md").read_text()
    assert "## Obama" in md and "a4_x" in md and "BEST_OVERALL" in md
    assert "[s0](/abs/s0.wav)" in md and "[s5](/abs/s5.wav)" in md
    assert "skipped" in md and "src control" in md
    assert "nan" not in md.lower().replace("nan_", "")  # nan -> '-'
