"""Unit tests for the forced-aspect-ratio radio on Krea 2: crop works, extend is
gated out (no inpaint pipeline) but the config plumbing exists for family parity.

Run:  .venv/Scripts/python tests/test_force_ratio.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import cz_pipeline as czp  # noqa: E402


def test_crop_still_crops():
    out = czp._crop_to_ratio(Image.new("RGB", (512, 768)), 16, 9)
    assert out.size[0] == 512 and out.size[1] < 768
    assert abs(out.size[0] / out.size[1] - 16 / 9) < 0.02


def test_mode_setter_normalises():
    old = czp.FORCE_RATIO_MODE
    try:
        czp.set_force_ratio_mode("EXTEND")
        assert czp.FORCE_RATIO_MODE == "extend"
        czp.set_force_ratio_mode("nimporte quoi")
        assert czp.FORCE_RATIO_MODE == "crop"
    finally:
        czp.FORCE_RATIO_MODE = old


def test_extend_raises_unsupported():
    # Krea 2 n'a pas de pipeline inpaint: l'extend doit echouer avec le message
    # UnsupportedFeature clair, pas un crash obscur.
    raised = False
    try:
        czp._extend_to_ratio(Image.new("RGB", (512, 768)), 16, 9, "", 6, 1)
    except Exception as e:
        raised = "Krea 2" in str(e) or "inpaint" in str(e).lower()
    assert raised, "extend sans pipeline inpaint doit lever le message clair"


def test_ui_radio_mapping():
    import cz_ui
    old_r, old_m = czp.FORCE_RATIO, czp.FORCE_RATIO_MODE
    try:
        cz_ui._ui_set_force_ratio("Crop to fit", "1152 x 896  (9:7)")
        assert czp.FORCE_RATIO and czp.FORCE_RATIO_MODE == "crop"
        cz_ui._ui_set_force_ratio("Off", "1152 x 896  (9:7)")
        assert czp.FORCE_RATIO == ""
    finally:
        czp.FORCE_RATIO, czp.FORCE_RATIO_MODE = old_r, old_m


if __name__ == "__main__":
    for fn in (test_crop_still_crops, test_mode_setter_normalises,
               test_extend_raises_unsupported, test_ui_radio_mapping):
        fn()
        print(f"OK {fn.__name__}")
    print("All force-ratio tests passed.")
