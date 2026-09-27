"""The guidance CivitAI recommends, converted to Krea 2's convention.

CivitAI publishes the `cfgScale` of the example images, that is to say a STANDARD CFG (ComfyUI,
A1111): uncond + cfg*(cond-uncond), 1.0 = no guidance. Krea 2 has its own
convention: cond + g*(cond-uncond), that is to say a standard CFG of 1 + g, and 0.0 turns
the guidance off. The 'Apply CivitAI recommended settings' button copied the cfg as it was:
the library's 14 Krea 2 all recommend 1.0 -- no guidance on them --, which
set g = 1.0, CFG ENABLED at the standard scale of 2, two passes per step on distilled
Turbos trained for g = 0. The opposite of what the community was using.

Run:  .venv/Scripts/python tests/test_civitai_guidance.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P


def test_standard_cfg_maps_to_krea2_guidance():
    assert P.guidance_from_standard_cfg(1.0) == 0.0     # 'cfg 1' = no guidance
    assert P.guidance_from_standard_cfg(1) == 0.0
    assert P.guidance_from_standard_cfg(5.5) == 4.5     # a Raw-style value
    assert P.guidance_from_standard_cfg(0.5) == 0.0     # never negative
    assert P.guidance_from_standard_cfg("n/a") is None  # unreadable: we touch nothing
    print("OK test_standard_cfg_maps_to_krea2_guidance")


def test_the_civitai_button_no_longer_turns_cfg_on():
    """The real case: a 'cfg 1.0' consensus, 8 steps -> guidance 0, steps 8."""
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
