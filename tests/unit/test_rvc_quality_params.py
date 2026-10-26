"""Quality parameters of the RVC engine: stream params, protect, change_rms, index, auto pitch.

Pure-numpy tests (StreamParams) run in the bot venv; the torch / faiss ones need the legacy venv
and skip elsewhere. Nothing here touches a GPU or a real model.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # tts.py, as in the worker
sys.path.insert(0, str(REPO / "legacy"))
sys.path.insert(0, str(REPO / "rvc_cli"))

import rvc_stream  # noqa: E402
from rvc_stream import StreamParams  # noqa: E402

# -- StreamParams ---------------------------------------------------------------------------------


def test_default_stream_params_unchanged() -> None:
    p = StreamParams()
    assert (p.first_block_ms, p.block_ms, p.context_ms, p.lookahead_ms) == (100, 320, 1000, 100)
    assert (p.crossfade_ms, p.search_ms, p.pitch_shift, p.f0_method) == (40, 20, 0, "pm")
    assert (p.use_index, p.index_rate) == (False, None)
    assert (p.f0_algo, p.protect, p.f0_post, p.index_nprobe) == ("pm", None, None, None)
    p.validate()


def test_stream_params_new_fields_valid() -> None:
    p = StreamParams.from_fields(
        {
            "f0_algo": "rmvpe",
            "use_index": True,
            "index_rate": 0.5,
            "index_nprobe": 16,
            "protect": 0.33,
            "f0_post": {"median_width": 5, "octave_repair": True},
            "unknown": 1,
            "protect_none": None,
        }
    )
    assert p.f0_algo == "rmvpe" and p.index_nprobe == 16 and p.protect == 0.33
    assert p.f0_post == {"median_width": 5, "octave_repair": True}


@pytest.mark.parametrize(
    "fields",
    [
        {"f0_algo": "harvest"},
        {"protect": 0.6},
        {"protect": -0.1},
        {"protect": True},
        {"index_nprobe": 0},
        {"index_nprobe": 257},
        {"index_nprobe": True},
        {"index_nprobe": 1.5},
        {"f0_post": [1]},
        {"f0_post": {"bogus": 1}},
        {"index_rate": 1.5},
    ],
)
def test_stream_params_rejects(fields) -> None:
    with pytest.raises(ValueError):
        StreamParams.from_fields(fields)


def test_stream_f0_method_stays_pm_only() -> None:
    with pytest.raises(ValueError, match="streaming supports only pm"):
        StreamParams(f0_method="rmvpe").validate()


# -- index path resolution ------------------------------------------------------------------------


def test_resolve_index_path_prefers_absolute_and_ignores_cwd(tmp_path, monkeypatch) -> None:
    idx = tmp_path / "m.index"
    idx.write_bytes(b"x")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    vc = types.SimpleNamespace(
        index_abs_path=str(idx), index_path_stripped="models/Nope/nope.index"
    )
    assert rvc_stream._resolve_index_path(vc) == str(idx)


def test_resolve_index_path_legacy_relative_still_works() -> None:
    real = next((REPO / "rvc_cli" / "models").glob("*/*.index"), None)
    if real is None:
        pytest.skip("no model index in rvc_cli/models")
    rel = str(real.relative_to(REPO / "rvc_cli"))
    got = rvc_stream._resolve_index_path(types.SimpleNamespace(index_path_stripped=rel))
    assert got is not None and os.path.isfile(got)


# -- torch / faiss parts (legacy venv) ------------------------------------------------------------

torch = pytest.importorskip("torch")
faiss = pytest.importorskip("faiss")
vc_infer_pipeline = pytest.importorskip("vc_infer_pipeline")


class _FakeHubert:
    """Constant HuBERT features of dimension 4: one frame per 320 samples."""

    def __init__(self, value: float = 1.0, dim: int = 4) -> None:
        self.value, self.dim = value, dim

    def extract_features(self, source, padding_mask, output_layer):  # noqa: ANN001
        frames = source.shape[1] // 320
        return (torch.full((1, frames, self.dim), self.value),)


class _FakeNet:
    def __init__(self) -> None:
        self.feats = None

    def infer(self, feats, p_len, *rest):  # noqa: ANN001, ANN002
        self.feats = feats.clone()
        return (torch.zeros(1, 1, 320 * int(p_len.item())),)


def _pipe() -> vc_infer_pipeline.VC:
    cfg = types.SimpleNamespace(
        x_pad=1, x_query=1, x_center=1, x_max=1, is_half=False, device="cpu"
    )
    return vc_infer_pipeline.VC(40000, cfg)


def _index_with(value: float = 10.0, dim: int = 4, n: int = 8):
    rows = np.full((n, dim), value, dtype="float32")
    rows += np.arange(n, dtype="float32")[:, None] * 1e-3
    idx = faiss.IndexFlatL2(dim)
    idx.add(rows)
    return idx, rows


def _run_vc(protect, pitchf):
    pipe, net = _pipe(), _FakeNet()
    idx, rows = _index_with()
    audio = np.zeros(320 * 6, dtype=np.float32)
    pitch = torch.ones(1, pitchf.shape[1]).long()
    pipe.vc(
        _FakeHubert(), net, torch.zeros(1).long(), audio, pitch, pitchf, [0, 0, 0],
        idx, rows, 1.0, "v2", protect=protect,
    )  # fmt: skip
    return net.feats[0, :, 0].numpy()


def test_protect_unvoiced_frames_keep_raw_features() -> None:
    # 12 f0 frames (10 ms) for 6 hubert frames x2 interpolation: first half voiced, rest unvoiced
    pitchf = torch.cat([torch.full((1, 6), 120.0), torch.zeros(1, 6)], dim=1)
    off = _run_vc(None, pitchf)
    assert np.allclose(off, 10.0, atol=0.1)  # index_rate 1: pure index features everywhere
    on = _run_vc(0.33, pitchf)
    assert np.allclose(on[:6], 10.0, atol=0.1)  # voiced: index
    expected = 0.33 * 10.0 + 0.67 * 1.0
    assert np.allclose(on[6:], expected, atol=0.1)  # unvoiced: protect weight on index


def test_protect_half_or_more_is_off() -> None:
    pitchf = torch.zeros(1, 12)
    assert np.allclose(_run_vc(0.5, pitchf), 10.0, atol=0.1)
    assert np.allclose(_run_vc(None, pitchf), 10.0, atol=0.1)


def test_index_score_clamp_exact_hit_is_finite() -> None:
    pipe, net = _pipe(), _FakeNet()
    idx, rows = _index_with(value=1.0)
    pipe.vc(
        _FakeHubert(1.0), net, torch.zeros(1).long(), np.zeros(320 * 4, np.float32), None, None,
        [0, 0, 0], idx, rows, 1.0, "v2",
    )  # fmt: skip
    assert np.isfinite(net.feats.numpy()).all()


def test_change_rms_rate_one_is_noop() -> None:
    sr = 8000
    y = (np.sin(np.arange(sr * 3) * 0.05) * np.linspace(0.1, 0.9, sr * 3)).astype(np.float32)
    out = vc_infer_pipeline.change_rms(y, sr, y * 0.5, sr, 1.0)
    assert np.allclose(out, y * 0.5, atol=1e-6)


def test_change_rms_rate_zero_follows_input_envelope() -> None:
    sr = 8000
    n = sr * 6
    env = np.linspace(0.1, 0.8, n)
    src = (np.sin(np.arange(n) * 0.07) * env).astype(np.float32)
    flat = (np.sin(np.arange(n) * 0.11) * 0.3).astype(np.float32)  # constant-level "output"
    out = vc_infer_pipeline.change_rms(src, sr, flat, sr, 0.0)
    w = sr // 2
    a = np.sqrt(np.mean(src[2 * sr : 2 * sr + w] ** 2))
    b = np.sqrt(np.mean(out[2 * sr : 2 * sr + w] ** 2))
    assert abs(b / a - 1) < 0.15
    c = np.sqrt(np.mean(src[5 * sr : 5 * sr + w] ** 2))
    d = np.sqrt(np.mean(out[5 * sr : 5 * sr + w] ** 2))
    assert abs(d / c - 1) < 0.15
    assert d > b * 1.5  # the level change of the input is reproduced


# -- VoiceConverter: index cache, cwd independence, auto pitch ------------------------------------


@pytest.fixture()
def voice_converter_mod(monkeypatch):
    monkeypatch.setenv("ESTIMATE_MULTIPLIER", os.environ.get("ESTIMATE_MULTIPLIER", "1.0"))
    monkeypatch.setattr(sys, "argv", ["rvc"])  # rvc_cli/config.py parses sys.argv at import
    try:
        import voice_converter
    except Exception as exc:  # noqa: BLE001 - rvc_cli stack not importable here
        pytest.skip(f"voice_converter not importable: {exc}")
    return voice_converter


def _bare_converter(mod, index_path):
    vc = mod.VoiceConverter.__new__(mod.VoiceConverter)
    vc.index_abs_path = str(index_path)
    vc.index_path_stripped = "models/x/x.index"
    vc._index = vc._big_npy = vc._index_nprobe_default = None
    vc._index_loaded = False
    vc.last_info = {}
    return vc


def test_get_index_loads_once_and_ignores_cwd(voice_converter_mod, tmp_path, monkeypatch) -> None:
    rows = np.random.default_rng(0).normal(size=(300, 8)).astype("float32")
    quant = faiss.IndexFlatL2(8)
    ivf = faiss.IndexIVFFlat(quant, 8, 4)
    ivf.train(rows)
    ivf.add(rows)
    ivf.nprobe = 2
    path = tmp_path / "t.index"
    faiss.write_index(ivf, str(path))
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    calls = []
    real = faiss.read_index
    monkeypatch.setattr(faiss, "read_index", lambda p: (calls.append(p), real(p))[1])
    vc = _bare_converter(voice_converter_mod, path)
    idx, big = vc.get_index()
    assert idx is not None and big.shape == (300, 8) and idx.nprobe == 2
    idx2, _ = vc.get_index(nprobe=4)
    assert idx2 is idx and idx.nprobe == 4 and len(calls) == 1
    vc.get_index()  # nprobe None restores the index's own value
    assert idx.nprobe == 2 and len(calls) == 1


def test_get_index_missing_file_returns_none(voice_converter_mod, tmp_path) -> None:
    vc = _bare_converter(voice_converter_mod, tmp_path / "missing.index")
    assert vc.get_index() == (None, None)


def test_convert_voice_auto_pitch_and_default_call(
    voice_converter_mod, tmp_path, monkeypatch
) -> None:
    mod = voice_converter_mod
    seen = {}

    class _Pipe:
        def pipeline(self, *args, **kwargs):  # noqa: ANN002, ANN003
            seen["args"], seen["kwargs"] = args, kwargs
            return np.zeros(4000, dtype=np.float32)

    fake_f0 = types.ModuleType("rvc_f0")

    class _Post:
        def __init__(self, **kw) -> None:  # noqa: ANN003
            self.kw = kw

    fake_f0.F0PostConfig = _Post
    fake_f0.extract_f0 = lambda x, p_len, method, **kw: np.full(p_len, 200.0, np.float32)
    fake_f0.median_f0 = lambda f0: float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 0.0
    fake_f0.auto_pitch_shift = lambda src, tgt, max_abs=12, strength=1.0: int(
        max(-max_abs, min(max_abs, round(strength * 12 * np.log2(tgt / src))))
    )
    monkeypatch.setitem(sys.modules, "rvc_f0", fake_f0)
    monkeypatch.setattr(mod, "load_audio", lambda path, sr: np.zeros(16000, dtype=np.float32))

    vc = _bare_converter(mod, tmp_path / "x.index")
    vc.cpt = {"f0": 1}
    vc.vc = _Pipe()
    vc.hubert_model = vc.net_g = None
    vc.version, vc.tgt_sr = "v2", 40000
    args = types.SimpleNamespace(
        sid=0, f0_pitch=2, f0_method="pm", index_rate=0.4, crepe_hop_length=128, f0_file=None
    )
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")

    # default call: exactly today's arguments (relative index string, no index object, shift 2)
    out = vc.convert_voice(wav, args)
    assert out is not None
    assert seen["args"][5] == 2 and seen["args"][7] == "models/x/x.index" and seen["args"][8] == 0.4
    assert seen["kwargs"]["index_obj"] is None and seen["kwargs"]["protect"] is None
    assert vc.last_info["index_used"] is False and vc.last_info["pitch_shift_applied"] == 2

    # auto pitch: source median 200 Hz -> target 100 Hz is -12 st on top of the base shift 2
    vc.convert_voice(
        wav, args, auto_pitch={"target_median_hz": 100.0, "max_abs": 12, "strength": 1.0}
    )
    assert seen["args"][5] == -10
    assert vc.last_info["pitch_shift_applied"] == -10
    assert vc.last_info["source_f0_median_hz"] == pytest.approx(200.0)
    vc.convert_voice(
        wav, args, auto_pitch={"target_median_hz": 100.0, "max_abs": 5, "strength": 1.0}
    )
    assert seen["args"][5] == 2 - 5
    vc.convert_voice(wav, args, auto_pitch={"target_median_hz": 100.0, "strength": 0.5})
    assert seen["args"][5] == 2 - 6
