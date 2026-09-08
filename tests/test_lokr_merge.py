"""LoKr LyCORIS: fusion dans les poids (dW = w1 (x) w2).

Ni peft ni diffusers ne savent poser un LoKr sur un pipeline -- pas une occurrence de
'lokr' dans loaders/lora_conversion_utils.py. Le probleme n'etait meme pas qu'il
echouait: il ne disait RIEN. Ses cles d'ai-toolkit s'appellent
'diffusion_model.blocks.0.attn.wk.lokr_w1', ni '.lora_A/B' ni le prefixe 'lora_unet_',
donc la garde checkpoint le prenait pour un modele et la garde LoRA le passait tel quel
a peft, qui n'appliquait aucun de ses facteurs et se taisait. Le rendu sortait comme si
l'adaptateur n'avait pas ete choisi.

Il est desormais fusionne. La conversion de cles passe par _krea2_rename, la table du
fork -- celle qui sert deja a convertir un single-file Comfy en dossier diffusers --
donc elle ne peut pas diverger du chargement du modele. Verifie sur le fichier reel
avant d'ecrire le code: les 256 modules de Ashen3/SNOFS Krea2/snofs_krea_v1_4 tombent
tous sur un poids existant, forme comprise.

Run:  .venv/Scripts/python tests/test_lokr_merge.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_krea2_lokr")
os.makedirs(TMP, exist_ok=True)


class _FakeTransformer:
    """Juste ce que _merge_lokr consomme: named_parameters()."""

    def __init__(self, shapes):
        self._p = {k: torch.nn.Parameter(torch.zeros(*s)) for k, s in shapes.items()}

    def named_parameters(self):
        return list(self._p.items())

    def __getitem__(self, k):
        return self._p[k].data


def _write(name, tensors):
    p = os.path.join(TMP, name)
    save_file(tensors, p)
    return p


def test_the_kronecker_product_lands_on_the_right_weight():
    """blocks.0.attn.wk -> transformer_blocks.0.attn.to_k.weight, via _krea2_rename."""
    w1, w2 = torch.randn(2, 2), torch.randn(3, 5)
    p = _write("lokr_wk.safetensors", {
        "diffusion_model.blocks.0.attn.wk.alpha": torch.tensor(1e10),
        "diffusion_model.blocks.0.attn.wk.lokr_w1": w1,
        "diffusion_model.blocks.0.attn.wk.lokr_w2": w2,
    })
    key = "transformer_blocks.0.attn.to_k.weight"
    assert P._krea2_rename("blocks.0.attn.wk.weight")[0] == key
    t = _FakeTransformer({key: (6, 10)})
    n, problems = P._merge_lokr(t, p, 1.0)
    assert (n, problems) == (1, []), (n, problems)
    assert torch.allclose(t[key], torch.kron(w1, w2), atol=1e-5)
    print("OK test_the_kronecker_product_lands_on_the_right_weight")


def test_every_attention_and_mlp_leaf_is_mapped():
    """Les feuilles que SNOFS touche reellement (releve sur le fichier: wq/wk/wv/wo,
    gate, mlp up/gate/down). Une seule non mappee = un merge a moitie vide."""
    leaves = {"attn.wq": "attn.to_q", "attn.wk": "attn.to_k", "attn.wv": "attn.to_v",
              "attn.wo": "attn.to_out.0", "attn.gate": "attn.to_gate",
              "mlp.up": "ff.up", "mlp.gate": "ff.gate", "mlp.down": "ff.down"}
    for src, dst in leaves.items():
        r = P._krea2_rename(f"blocks.3.{src}.weight")
        assert r and r[0] == f"transformer_blocks.3.{dst}.weight", (src, r)
        assert r[1] is None, (src, r)
    print("OK test_every_attention_and_mlp_leaf_is_mapped")


def test_the_lora_weight_scales_the_delta():
    w1, w2 = torch.randn(2, 2), torch.randn(2, 2)
    p = _write("lokr_w.safetensors", {
        "diffusion_model.blocks.0.attn.wq.lokr_w1": w1,
        "diffusion_model.blocks.0.attn.wq.lokr_w2": w2,
    })
    key = "transformer_blocks.0.attn.to_q.weight"
    t = _FakeTransformer({key: (4, 4)})
    P._merge_lokr(t, p, 0.5)
    assert torch.allclose(t[key], 0.5 * torch.kron(w1, w2), atol=1e-5)
    print("OK test_the_lora_weight_scales_the_delta")


def test_full_factors_use_no_scalar():
    """SNOFS: w1 et w2 pleines, alpha = 1e10 (sentinelle lora_dim). Aucun scalaire."""
    mod = {"lokr_w1": torch.randn(2, 2), "lokr_w2": torch.randn(2, 2),
           "alpha": torch.tensor(1e10)}
    assert P._lokr_scale(mod, None) == 1.0
    assert torch.allclose(P._lokr_delta(mod),
                          torch.kron(mod["lokr_w1"], mod["lokr_w2"]), atol=1e-5)
    print("OK test_full_factors_use_no_scalar")


def test_factored_factors_use_alpha_over_rank():
    """Forme factorisee: w1 = w1_a @ w1_b, rang 2, alpha 8 -> echelle 4, comme peft."""
    a, b = torch.randn(4, 2), torch.randn(2, 4)
    mod = {"lokr_w1_a": a, "lokr_w1_b": b, "lokr_w2": torch.randn(2, 2),
           "alpha": torch.tensor(8.0)}
    assert P._lokr_scale(mod, 2) == 4.0
    assert torch.allclose(P._lokr_delta(mod),
                          torch.kron(a @ b, mod["lokr_w2"]) * 4.0, atol=1e-4)
    print("OK test_factored_factors_use_alpha_over_rank")


def test_an_unmapped_module_is_reported_not_dropped():
    """La regle de la maison: rien ne disparait en silence. Un mapping qui derive
    donnerait un merge a moitie vide et un rendu presque normal -- le pire des cas."""
    p = _write("lokr_orphan.safetensors", {
        "diffusion_model.blocks.0.attn.wINVENTED.lokr_w1": torch.randn(2, 2),
        "diffusion_model.blocks.0.attn.wINVENTED.lokr_w2": torch.randn(2, 2),
    })
    t = _FakeTransformer({"transformer_blocks.0.attn.to_q.weight": (4, 4)})
    n, problems = P._merge_lokr(t, p, 1.0)
    assert n == 0 and len(problems) == 1, (n, problems)
    assert torch.count_nonzero(t["transformer_blocks.0.attn.to_q.weight"]) == 0
    print("OK test_an_unmapped_module_is_reported_not_dropped")


def test_a_shape_mismatch_is_reported_not_applied():
    p = _write("lokr_badshape.safetensors", {
        "diffusion_model.blocks.0.attn.wq.lokr_w1": torch.randn(2, 2),
        "diffusion_model.blocks.0.attn.wq.lokr_w2": torch.randn(2, 2),
    })
    t = _FakeTransformer({"transformer_blocks.0.attn.to_q.weight": (8, 8)})
    n, problems = P._merge_lokr(t, p, 1.0)
    assert n == 0 and len(problems) == 1 and "vs weight" in problems[0], problems
    print("OK test_a_shape_mismatch_is_reported_not_applied")


def _lycoris(name, suffixes):
    sd = {}
    for i in range(6):
        b = f"diffusion_model.blocks.{i}.attn.wq"
        sd[b + ".alpha"] = torch.tensor(1.0)
        for s in suffixes:
            sd[f"{b}.{s}"] = torch.zeros(4, 4)
    return _write(name, sd)


def test_routing_lokr_supported_loha_refused():
    lokr = _lycoris("route_lokr.safetensors", ("lokr_w1", "lokr_w2"))
    loha = _lycoris("route_loha.safetensors",
                    ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"))
    # dans le dossier des checkpoints: refuses tous les deux, mais pas pour la meme
    # raison -- le LoKr, lui, s'entend dire ou aller.
    assert "LoRA folder" in (P._safetensors_unsupported(lokr) or "")
    assert "LoHa" in (P._safetensors_unsupported(loha) or "")
    # dans le dossier LoRA: le LoKr passe (il sera fusionne), le LoHa est nomme.
    assert P._lora_unsupported(lokr) is None
    assert "LoHa" in (P._lora_unsupported(loha) or "")
    # et le jeu envoye a peft ne contient plus le LoKr, mais garde le LoHa pour
    # qu'il y soit refuse par son nom plutot que de disparaitre.
    s = [(lokr, 1.0), (loha, 1.0)]
    assert P._lokr_set(s) == [(lokr, 1.0)]
    assert P._peft_set(s) == [(loha, 1.0)]
    print("OK test_routing_lokr_supported_loha_refused")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All LoKr merge tests passed.")
