"""Parity gate for the fairseq-free HuBERT loader (rvc_cli/hubert_loader.py).

The oracle (tests/fixtures/hubert_oracle.npz) was produced by the original
fairseq 0.12.2 loader in the legacy py3.10 venv (CPU, fp32) for two clips:
layer-9 features through final_proj ("v1", 256-d) and layer-12 features ("v2",
768-d), with the exact call vc_infer_pipeline makes. The new loader must
reproduce them without fairseq, from the plain tensor checkpoint
hubert_base.state_dict.pt (a fairseq-free conversion of hubert_base.pt).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "tests" / "fixtures" / "hubert_oracle.npz"
STATE_DICT = Path(os.environ.get("HUBERT_STATE_DICT", ROOT / "hubert_base.state_dict.pt"))

pytestmark = pytest.mark.skipif(
    not STATE_DICT.exists(), reason="hubert_base.state_dict.pt not present"
)


@pytest.fixture(scope="module")
def model():
    sys.path.insert(0, str(ROOT / "rvc_cli"))
    from hubert_loader import load_hubert

    m = load_hubert(STATE_DICT, device="cpu", is_half=False)
    assert "fairseq" not in sys.modules, "the loader must not import fairseq"
    return m


@pytest.fixture(scope="module")
def oracle():
    return np.load(ORACLE)


def _features(model, wav, version):
    feats = torch.from_numpy(wav).float().view(1, -1)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    with torch.no_grad():
        logits = model.extract_features(
            source=feats, padding_mask=padding_mask, output_layer=9 if version == "v1" else 12
        )
        out = model.final_proj(logits[0]) if version == "v1" else logits[0]
    return out[0].numpy()


@pytest.mark.parametrize("clip", ["sine", "goku"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_matches_fairseq_reference(model, oracle, clip, version):
    expected = oracle[f"{clip}_{version}"]
    got = _features(model, oracle[f"{clip}_wav16k"], version)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-3)


def test_eval_mode_and_device_move(model):
    assert not model.training
    assert hasattr(model, "to") and hasattr(model, "half")


def test_half_precision_load():
    sys.path.insert(0, str(ROOT / "rvc_cli"))
    from hubert_loader import load_hubert

    m = load_hubert(STATE_DICT, device="cpu", is_half=True)
    assert next(m.parameters()).dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 inference needs CUDA")
def test_half_precision_inference_matches_reference(oracle):
    # The worker runs HuBERT in fp16 on the GPU; loading in half is not enough,
    # the forward pass must work (mixed-dtype norms) and stay near the fp32 oracle.
    sys.path.insert(0, str(ROOT / "rvc_cli"))
    from hubert_loader import load_hubert

    m = load_hubert(STATE_DICT, device="cuda:0", is_half=True)
    wav = oracle["goku_wav16k"]
    feats = torch.from_numpy(wav).half().view(1, -1).to("cuda:0")
    padding_mask = torch.BoolTensor(feats.shape).fill_(False).to("cuda:0")
    with torch.no_grad():
        out = m.final_proj(
            m.extract_features(source=feats, padding_mask=padding_mask, output_layer=9)[0]
        )
    got = out[0].float().cpu().numpy()
    expected = oracle["goku_v1"]
    assert got.shape == expected.shape
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=0, atol=5e-2)


def test_checkpoint_is_loaded_weights_only(monkeypatch):
    sys.path.insert(0, str(ROOT / "rvc_cli"))
    import hubert_loader

    seen = {}
    real_load = torch.load

    def spy(*args, **kwargs):
        seen["weights_only"] = kwargs.get("weights_only")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    hubert_loader.load_hubert(STATE_DICT, device="cpu", is_half=False)
    assert seen.get("weights_only") is True
