"""Unit tests for the Krea 2 LoRA alpha fold.

Regression guard: diffusers' Krea 2 converter
(_convert_non_diffusers_krea2_lora_to_diffusers) consumes only the `.lora_A`/`.lora_B`
keys and never pops the `.alpha` ones, so a LoRA carrying alphas died on
    ValueError: `state_dict` should be empty at this point but has
                dict_keys(['blocks.0.attn.gate.alpha', ...])
It never applied that scaling either, which the fold now bakes into the up weights.

No model is loaded: the state dicts are synthetic and tiny.

Run:  .venv/Scripts/python tests/test_lora_alpha.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_pipeline as P  # noqa: E402

DIM, RANK = 16, 4
MODULES = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
           "mlp.gate", "mlp.up", "mlp.down")


def _trainer_sd(n_blocks=2, alpha=8.0):
    """The Krea 2 trainer layout: 'diffusion_model.' prefix, blocks.N.attn|mlp.<module>,
    lora_A/lora_B + one alpha per module."""
    sd = {}
    for i in range(n_blocks):
        for mod in MODULES:
            base = f"diffusion_model.blocks.{i}.{mod}"
            sd[f"{base}.lora_A.weight"] = torch.randn(RANK, DIM)
            sd[f"{base}.lora_B.weight"] = torch.randn(DIM, RANK)
            sd[f"{base}.alpha"] = torch.tensor(alpha)
    return sd


def test_fold_scales_the_up_weight_and_removes_the_key():
    sd = _trainer_sd(n_blocks=1, alpha=8.0)
    up_before = sd["diffusion_model.blocks.0.attn.gate.lora_B.weight"].clone()
    out, folded = P.fold_lora_alpha(sd)
    assert folded == len(MODULES), folded
    assert not [k for k in out if k.endswith(".alpha")]
    # alpha 8 / rank 4 -> x2 on the up weight, the down weight untouched.
    assert torch.allclose(out["diffusion_model.blocks.0.attn.gate.lora_B.weight"],
                          up_before * 2.0)
    assert torch.equal(out["diffusion_model.blocks.0.attn.gate.lora_A.weight"],
                       sd["diffusion_model.blocks.0.attn.gate.lora_A.weight"])


def test_fold_keeps_the_product_identical():
    """What reaches the model is B @ A: folding must not change it. The Krea 2 converter
    dropped the alphas silently, so this also RESTORES a scaling that was being lost."""
    sd = _trainer_sd(n_blocks=1, alpha=2.0)
    a = sd["diffusion_model.blocks.0.mlp.up.lora_A.weight"]
    b = sd["diffusion_model.blocks.0.mlp.up.lora_B.weight"]
    expected = (b @ a) * (2.0 / RANK)
    out, _ = P.fold_lora_alpha(sd)
    got = (out["diffusion_model.blocks.0.mlp.up.lora_B.weight"]
           @ out["diffusion_model.blocks.0.mlp.up.lora_A.weight"])
    assert torch.allclose(got, expected, atol=1e-6)


def test_fold_drops_an_orphan_alpha():
    out, folded = P.fold_lora_alpha({"blocks.0.attn.gate.alpha": torch.tensor(4.0)})
    assert folded == 0 and out == {}


def test_needs_repair_only_for_the_faulty_layout():
    assert P._lora_needs_repair(_trainer_sd(n_blocks=1)) is True
    # No alpha -> the converter is happy, nothing to do.
    no_alpha = {k: v for k, v in _trainer_sd(n_blocks=1).items() if not k.endswith(".alpha")}
    assert P._lora_needs_repair(no_alpha) is False
    # A LoKr has its own path (_apply_lokrs_to): the LoRA repair must never claim it.
    lokr = {"diffusion_model.blocks.0.attn.gate.alpha": torch.tensor(4.0),
            "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.zeros(DIM, DIM),
            "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.zeros(DIM, DIM)}
    assert P._lora_needs_repair(lokr) is False


def test_lora_source_leaves_a_healthy_file_on_the_tested_path():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "healthy.safetensors")
    save_file({k: v for k, v in _trainer_sd(n_blocks=1).items()
               if not k.endswith(".alpha")}, p)
    src, kw = P._lora_source(p)
    assert src == d and kw == {"weight_name": "healthy.safetensors"}
    q = os.path.join(d, "faulty.safetensors")
    save_file(_trainer_sd(n_blocks=1), q)
    src, kw = P._lora_source(q)
    assert isinstance(src, dict) and kw == {}
    assert not [k for k in src if k.endswith(".alpha")]


def test_unreadable_file_falls_back_to_the_path():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "not-a-safetensors.safetensors")
    with open(p, "wb") as f:
        f.write(b"garbage")
    src, kw = P._lora_source(p)                       # must not raise
    assert src == d and kw == {"weight_name": os.path.basename(p)}


def test_folded_dict_passes_the_diffusers_converter():
    """The bug itself: the trainer layout used to raise ValueError('state_dict should be
    empty ... attn.gate.alpha')."""
    from diffusers import Krea2Pipeline
    sd = _trainer_sd(n_blocks=2)
    try:
        Krea2Pipeline.lora_state_dict(dict(sd))
    except ValueError as e:
        assert "should be empty" in str(e), e
    else:
        raise AssertionError("diffusers no longer fails: this repair may be obsolete")
    folded, _ = P.fold_lora_alpha(sd)
    out = Krea2Pipeline.lora_state_dict(folded)
    mods = {k.split(".lora_")[0] for k in out}
    assert any(m.endswith("attn.to_gate") for m in mods), mods
    assert any(m.endswith("attn.to_out.0") for m in mods), mods


def test_meta_params_detects_a_meta_module():
    assert P._meta_params(torch.nn.Linear(4, 4)) == []
    with torch.device("meta"):
        ghost = torch.nn.Linear(4, 4)
    assert "weight" in P._meta_params(ghost)
    assert P._meta_params(None) == []


if __name__ == "__main__":
    for fn in (test_fold_scales_the_up_weight_and_removes_the_key,
               test_fold_keeps_the_product_identical,
               test_fold_drops_an_orphan_alpha,
               test_needs_repair_only_for_the_faulty_layout,
               test_lora_source_leaves_a_healthy_file_on_the_tested_path,
               test_unreadable_file_falls_back_to_the_path,
               test_folded_dict_passes_the_diffusers_converter,
               test_meta_params_detects_a_meta_module):
        fn()
        print(f"OK {fn.__name__}")
    print("All Krea 2 LoRA alpha tests passed.")
