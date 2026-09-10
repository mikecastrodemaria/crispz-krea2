"""Poids FP8/INT8 stockes DEJA a l'echelle: le weight_scale fourni ne s'applique pas.
Porte de crispz-klein 1.34.1 (kleinFinalcutFP16FP8_comfyQuant rendait du bruit).

Run:  .venv/Scripts/python tests/test_prescaled_fp8.py
"""
import os
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

torch.manual_seed(0)
E4 = torch.float8_e4m3fn
# Cles qui passent la garde d'architecture Krea 2 du lecteur (txtfusion / blocks.0.attn.wq).
K1, K2 = "txtfusion.layerwise_blocks.0.probe.weight", "blocks.0.attn.wq.weight"


def _pair(n):
    w = torch.randn(n, n) * 0.02
    return w, (w.abs().max() / 448.0).reshape(())


def test_the_detector_separates_the_layouts():
    w, s = _pair(64)
    assert not P._stored_at_scale((w / s).to(E4).float(), s, E4)       # FP8 normal
    assert P._stored_at_scale(w.to(E4).float(), s, E4)                  # deja a l'echelle
    assert not P._stored_at_scale((w * 50).to(E4).float(), torch.tensor(0.5), E4)
    s8 = (w.abs().max() / 127.0).reshape(())
    q8 = torch.round(w / s8).clamp(-127, 127).to(torch.int8)
    assert not P._stored_at_scale(q8.float(), s8, torch.int8)          # INT8 normal
    full = torch.randint(-127, 128, (4, 3), dtype=torch.int8)
    full[0, 0] = 127
    assert not P._stored_at_scale(full.float(), torch.full((4, 1), 0.9), torch.int8)
    assert not P._stored_at_scale(w.to(E4).float(), torch.tensor([120], dtype=torch.uint8), E4)
    print("OK test_the_detector_separates_the_layouts")


def test_the_reader_handles_a_file_mixing_both_layouts():
    w1, s1 = _pair(32)            # deja a l'echelle
    w2, s2 = _pair(48)            # normal (autre taille: pas d'ambiguite)
    p = os.path.join(tempfile.mkdtemp(), "mixed.safetensors")
    save_file({K1: w1.to(E4), K1 + "_scale": s1.float(),
               K2: (w2 / s2).to(E4), K2 + "_scale": s2.float()}, p)
    out = P._read_comfy_state_dict(p)
    sd = out[0] if isinstance(out, tuple) else out
    for k, w in ((K1, w1), (K2, w2)):
        rel = ((sd[k].float() - w).norm() / w.norm()).item()
        assert rel < 0.1, (k, rel)          # sans le correctif: ~1.0 sur K1
    print("OK test_the_reader_handles_a_file_mixing_both_layouts")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All prescaled-FP8 tests passed.")
