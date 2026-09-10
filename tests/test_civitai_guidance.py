"""La guidance recommandee par CivitAI, convertie a la convention de Krea 2.

CivitAI publie le `cfgScale` des images d'exemple, soit une CFG STANDARD (ComfyUI,
A1111): uncond + cfg*(cond-uncond), 1.0 = pas de guidance. Krea 2 a sa propre
convention: cond + g*(cond-uncond), c'est-a-dire une CFG standard de 1 + g, et 0.0 coupe
la guidance. Le bouton 'Apply CivitAI recommended settings' recopiait le cfg tel quel:
les 14 Krea 2 de la bibliotheque recommandent tous 1.0 -- pas de guidance chez eux --, ce
qui reglait g = 1.0, CFG ACTIVEE a l'echelle standard 2, deux passes par step sur des
Turbo distilles entraines pour g = 0. L'inverse de ce que la communaute utilisait.

Run:  .venv/Scripts/python tests/test_civitai_guidance.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P


def test_standard_cfg_maps_to_krea2_guidance():
    assert P.guidance_from_standard_cfg(1.0) == 0.0     # 'cfg 1' = pas de guidance
    assert P.guidance_from_standard_cfg(1) == 0.0
    assert P.guidance_from_standard_cfg(5.5) == 4.5     # une valeur facon Raw
    assert P.guidance_from_standard_cfg(0.5) == 0.0     # jamais negatif
    assert P.guidance_from_standard_cfg("n/a") is None  # illisible: on ne touche a rien
    print("OK test_standard_cfg_maps_to_krea2_guidance")


def test_the_civitai_button_no_longer_turns_cfg_on():
    """Le cas reel: consensus 'cfg 1.0', 8 steps -> guidance 0, steps 8."""
    import cz_civitai
    import cz_ui as U
    old = (cz_civitai.load_civitai_sidecar, U.resolve_checkpoint)
    cz_civitai.load_civitai_sidecar = lambda p: {
        "recommended": {"n": 10, "steps": 8, "guidance": 1.0, "sampler": "Euler"}}
    U.resolve_checkpoint = lambda n: "F:/fake/turbo_merge.safetensors"
    try:
        out = U._ui_civitai_reco("turbo_merge.safetensors")
    finally:
        cz_civitai.load_civitai_sidecar, U.resolve_checkpoint = old
    msg, steps_u, guidance_u = out[0], out[1], out[2]
    assert guidance_u.get("value") == 0.0, guidance_u
    assert steps_u.get("value") == 8, steps_u
    assert "ComfyUI" in msg and "0 = off" in msg, msg
    print("OK test_the_civitai_button_no_longer_turns_cfg_on")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All CivitAI-guidance tests passed.")
