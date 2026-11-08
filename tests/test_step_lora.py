"""Step-distill LoRAs (Krea 2 Turbo 3-step) + 'owner/repo:file' LoRA specs.

The 3-step distill is a LoRA that REPLACES the step count: it wants 3 steps, guidance
0.0, on the Turbo base. The app has to (1) recognise it under its several file names,
(2) fetch it from the HF Hub on first use from an 'owner/repo:file' spec, and (3) say
so in the log when it is used outside its recipe -- without ever blocking the render.

Neither GPU nor network: the download is a stub that writes an empty file.

Run:  .venv/Scripts/python tests/test_step_lora.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402

SPEC = "lvladikov/Krea2-Turbo-Distill-3step-LoRA:krea2_turbo_3step_rank_64_lora.safetensors"
FILE = "krea2_turbo_3step_rank_64_lora.safetensors"


class _Lib:
    """An empty LoRA folder, swapped in for the test; the download writes an empty file."""

    def __enter__(self):
        self.dir = tempfile.mkdtemp()
        self.old = (P.LORAS_DIR, list(P.LORAS_EXTRA_DIRS), P._download_hf_lora, list(P.LORAS))
        P.LORAS_DIR, P.LORAS_EXTRA_DIRS = self.dir, []
        self.calls = []

        def fake(repo, fname):
            self.calls.append((repo, fname))
            if getattr(self, "fail", False):
                raise RuntimeError("offline")
            p = os.path.join(P.LORAS_DIR, fname)
            open(p, "wb").close()
            return p
        P._download_hf_lora = fake
        return self

    def __exit__(self, *exc):
        P.LORAS_DIR, P.LORAS_EXTRA_DIRS, P._download_hf_lora, P.LORAS = self.old
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def test_spec_parsing():
    assert P.split_hf_lora_spec(SPEC) == ("lvladikov/Krea2-Turbo-Distill-3step-LoRA", FILE)
    # A plain file name, a subfolder path and a Windows path are NOT HF specs.
    for s in (FILE, "Style/noir.safetensors", r"D:\loras\x.safetensors", "", None,
              "owner/repo", "owner/repo:sub/file.safetensors"):
        assert P.split_hf_lora_spec(s) is None, s


def test_step_lora_recognised_under_its_names():
    # The author's two files, a re-upload's suffix, a full path, any case.
    for n in (FILE, "krea2_turbo_3step_rank_64_lora_comfyui.safetensors",
              "krea2_turbo_3step_rank_64_lora_comfyui_lora_f16.safetensors",
              r"D:\loras\KREA2_TURBO_3STEP_rank_64_lora.safetensors"):
        e = P.step_lora_for(n)
        assert e and e["steps"] == 3 and e["guidance"] == 0.0, n
    for n in ("ideokrea_v1_lora_f16.safetensors", "", None, "turbo_3step.safetensors"):
        assert P.step_lora_for(n) is None, n


def test_ensure_lora_downloads_once():
    with _Lib() as lib:
        assert P.ensure_lora(SPEC) == FILE
        assert lib.calls == [("lvladikov/Krea2-Turbo-Distill-3step-LoRA", FILE)]
        assert os.path.isfile(os.path.join(lib.dir, FILE))
        # Present now -> no second download, whichever spelling is used.
        assert P.ensure_lora(SPEC) == FILE
        assert P.ensure_lora(FILE) == FILE
        assert len(lib.calls) == 1
        # A plain name that is not a known step LoRA passes through untouched.
        assert P.ensure_lora("Style/noir.safetensors") == "Style/noir.safetensors"
        assert P.ensure_lora("") is None


def test_known_step_lora_name_fetches_from_its_repo():
    with _Lib() as lib:
        assert P.ensure_lora(FILE) == FILE
        assert lib.calls == [("lvladikov/Krea2-Turbo-Distill-3step-LoRA", FILE)]
        # The ComfyUI twin is recognised but is NOT the file we ship: no download.
        twin = "krea2_turbo_3step_rank_64_lora_comfyui.safetensors"
        assert P.ensure_lora(twin) == twin
        assert len(lib.calls) == 1


def test_download_failure_drops_the_slot_not_the_render():
    with _Lib() as lib:
        lib.fail = True
        assert P.ensure_lora(SPEC) is None
        assert P.ensure_lora(SPEC, download=False) is None
        P.LORAS = []
        P.set_loras([(SPEC, 1.0), ("None", 1.0)])
        assert P.LORAS == [], P.LORAS


def test_set_loras_accepts_a_spec_and_keeps_the_stack():
    with _Lib() as lib:
        style = os.path.join(lib.dir, "ideokrea_v1_lora_f16.safetensors")
        open(style, "wb").close()
        P.LORAS = []
        P.set_loras([(SPEC, 1.0), ("ideokrea_v1_lora_f16.safetensors", 0.6)])
        names = [os.path.basename(p) for p, _ in P.LORAS]
        assert names == [FILE, "ideokrea_v1_lora_f16.safetensors"], names
        assert [w for _, w in P.LORAS] == [1.0, 0.6]
        assert P.active_step_lora()["steps"] == 3


def test_recipe_warning():
    loras = [(FILE, 1.0), ("ideokrea_v1_lora_f16.safetensors", 0.6)]
    turbo = "krea/Krea-2-Turbo"
    # The recipe, and the measured-usable range (the 5-step stack from the field).
    assert P.step_lora_warning(3, 0.0, loras, turbo) is None
    assert P.step_lora_warning(5, 0.0, loras, turbo) is None
    assert P.step_lora_warning(6, 0.0, loras, turbo) is None
    # The Turbo default (8 steps) left in place: named, with the preset to pick.
    w = P.step_lora_warning(8, 0.0, loras, turbo)
    assert w and "8 steps" in w and "Turbo 3 steps (distill LoRA)" in w, w
    # Guidance on breaks the distilled trajectory.
    w = P.step_lora_warning(3, 4.5, loras, turbo)
    assert w and "guidance 4.5" in w, w
    # A Raw base is named too.
    w = P.step_lora_warning(3, 0.0, loras, "krea/Krea-2-Raw")
    assert w and "Raw" in w, w
    # No step LoRA active -> silence, whatever the settings.
    assert P.step_lora_warning(28, 4.5, [("ideokrea_v1_lora_f16.safetensors", 0.6)], turbo) is None
    assert P.step_lora_warning(28, 4.5, [], turbo) is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} step LoRA tests passed.")
