"""A replacement text encoder (Models > Checkpoints > Text encoder).

Ported from crispz-klein 1.34.0. Krea 2 reads TWELVE hidden states of the Qwen3-VL encoder at
FIXED indices (text_encoder_select_layers = 2, 5, ..., 35, as many as
transformer.config.num_text_layers), 2560 wide: another encoder only plugs in when
it has the same family (qwen3_vl), the same width and the same number of layers. Fewer
layers: the indices overflow on the first prompt. More: the transformer reads other
depths than the ones it was trained on, with no error. The refusal must say so BEFORE
reading 8 GB.

These tests also lock down what would make the option silently dangerous:
  - a change of encoder empties the embeddings cache (otherwise the old encodings
    stay served) and the encoder is part of the cache KEY -- id(enc) alone is not
    enough, CPython recycles the ids of freed objects;
  - an encoder discarded, or that fails to load, never costs a render:
    _ensure_base falls back on the base repo's, and the metadata says so;
  - the metadata names the encoder that REALLY ran, by its folder name
    and never by its path (which would end up in the shared PNGs);
  - the queue keeps the job's encoder.

No model loaded, no network: the base repo's encoder config is replaced,
and both Krea2Pipeline and the encoder's class are fakes.

Run:  .venv/Scripts/python tests/test_text_encoder.py

"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_imageio
import cz_pipeline as P

# The encoder of krea/Krea-2-Turbo (and -Raw): the text part is kept under text_config,
# the vision one (1024 wide) does not count.
QWEN3VL_4B = {"model_type": "qwen3_vl", "architectures": ["Qwen3VLModel"],
              "text_config": {"model_type": "qwen3_vl_text", "hidden_size": 2560,
                              "num_hidden_layers": 36},
              "vision_config": {"model_type": "qwen3_vl", "hidden_size": 1024, "depth": 24}}
QWEN3VL_8B = {**QWEN3VL_4B, "text_config": {**QWEN3VL_4B["text_config"], "hidden_size": 4096}}
QWEN3_4B_TEXT = {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36,
                 "architectures": ["Qwen3ForCausalLM"]}


def _layers(n):
    return {**QWEN3VL_4B, "text_config": {**QWEN3VL_4B["text_config"], "num_hidden_layers": n}}


def _folder(cfg, sub=None, name="enc"):
    root = tempfile.mkdtemp(prefix="te_")
    d = os.path.join(root, name)
    p = os.path.join(d, sub) if sub else d
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return d


class _Base:
    """Replaces the encoder config of the base repo (no network, no HF)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __enter__(self):
        self.old = P._base_text_encoder_config
        P._base_text_encoder_config = lambda base=None: self.cfg

    def __exit__(self, *a):
        P._base_text_encoder_config = self.old


class _FakeEncoderClass:
    """A dummy encoder class: it remembers what from_pretrained receives, or fails."""
    calls = []
    boom = None

    @classmethod
    def from_pretrained(cls, where, **kw):
        cls.calls.append((where, kw))
        if cls.boom:
            raise cls.boom
        return cls()


def test_dims_are_read_under_text_config():
    assert P._enc_dims(QWEN3VL_4B) == (2560, 36, "qwen3_vl"), P._enc_dims(QWEN3VL_4B)
    assert P._enc_dims(QWEN3_4B_TEXT) == (2560, 36, "qwen3"), P._enc_dims(QWEN3_4B_TEXT)
    print("OK test_dims_are_read_under_text_config")


def test_same_architecture_is_accepted():
    with _Base(QWEN3VL_4B):
        assert P._text_encoder_problem(_folder(QWEN3VL_4B)) is None
        # weights in a text_encoder/ subfolder (a copy of a diffusers repo)
        assert P._text_encoder_problem(_folder(QWEN3VL_4B, "text_encoder")) is None
    print("OK test_same_architecture_is_accepted")


def test_another_width_is_refused_with_both_numbers():
    with _Base(QWEN3VL_4B):
        why = P._text_encoder_problem(_folder(QWEN3VL_8B))
    assert why and "4096" in why and "2560" in why, why
    print("OK test_another_width_is_refused_with_both_numbers")


def test_the_layer_count_is_checked_both_ways():
    """FIXED indices up to 35: fewer layers overflows, more reads other
    depths with no error. Both are refused, with the numbers to back it."""
    with _Base(QWEN3VL_4B):
        for n in (28, 40):
            why = P._text_encoder_problem(_folder(_layers(n)))
            assert why and str(n) in why and "36" in why, why
    print("OK test_the_layer_count_is_checked_both_ways")


def test_a_text_only_qwen3_is_refused_by_family():
    """The same width, the same depth, another architecture: the family is enough."""
    with _Base(QWEN3VL_4B):
        why = P._text_encoder_problem(_folder(QWEN3_4B_TEXT))
    assert why and "'qwen3'" in why and "qwen3_vl" in why, why
    print("OK test_a_text_only_qwen3_is_refused_by_family")


def test_gguf_single_file_and_empty_folder_are_refused_with_the_reason():
    with _Base(QWEN3VL_4B):
        assert "GGUF" in P._text_encoder_problem(r"F:\x\qwen3-vl-4b-q8_0.gguf")
        assert "FOLDER" in P._text_encoder_problem(r"F:\x\qwen3_vl_4b.safetensors")
        assert "config.json" in P._text_encoder_problem(tempfile.mkdtemp())
    print("OK test_gguf_single_file_and_empty_folder_are_refused_with_the_reason")


def test_an_unreadable_base_config_leaves_it_to_the_load():
    """Nothing to compare: no refusal, the loading will settle it (and fall back when needed)."""
    with _Base(None):
        assert P._text_encoder_problem(_folder(QWEN3VL_8B)) is None
    print("OK test_an_unreadable_base_config_leaves_it_to_the_load")


def test_hf_ids_may_carry_a_subfolder():
    assert P._split_hf_src("owner/repo") == ("owner/repo", None)
    assert P._split_hf_src("owner/repo/text_encoder") == ("owner/repo", "text_encoder")
    assert P._split_hf_src("owner/repo/a/b") == ("owner/repo", "a/b")
    print("OK test_hf_ids_may_carry_a_subfolder")


def test_the_class_comes_from_the_base_repo_model_index():
    """The class is the one diffusers would have loaded, read from model_index.json
    (Qwen3VLModel on Krea 2). A dummy module: importing the real transformers class
    pulls torchao, which crashes with no visible GPU."""
    mod = types.ModuleType("_fake_te_lib")

    class Qwen3VLModel:
        pass

    mod.Qwen3VLModel = Qwen3VLModel
    sys.modules["_fake_te_lib"] = mod
    base = tempfile.mkdtemp(prefix="base_")
    idx = os.path.join(base, "model_index.json")
    try:
        with open(idx, "w", encoding="utf-8") as f:
            json.dump({"_class_name": "Krea2Pipeline",
                       "text_encoder": ["_fake_te_lib", "Qwen3VLModel"]}, f)
        assert P._encoder_class(base) is Qwen3VLModel
        # a model_index with no text_encoder: a named error, not a bare KeyError
        with open(idx, "w", encoding="utf-8") as f:
            json.dump({"_class_name": "Krea2Pipeline"}, f)
        try:
            P._encoder_class(base)
        except RuntimeError as e:
            assert "text encoder" in str(e), e
        else:
            raise AssertionError("un model_index sans text_encoder doit lever")
    finally:
        sys.modules.pop("_fake_te_lib", None)
    print("OK test_the_class_comes_from_the_base_repo_model_index")


def test_the_encoder_loads_in_bf16_from_its_subfolder():
    """DTYPE and the subfolder, nothing else: torchao only touches the transformer."""
    d = _folder(QWEN3VL_4B, "text_encoder")
    old = P._encoder_class
    _FakeEncoderClass.calls, _FakeEncoderClass.boom = [], None
    try:
        P._encoder_class = lambda base=None: _FakeEncoderClass
        enc = P._load_text_encoder(d)
    finally:
        P._encoder_class = old
    assert isinstance(enc, _FakeEncoderClass), enc
    (where, kw), = _FakeEncoderClass.calls
    assert where == d and kw == {"torch_dtype": P.DTYPE, "subfolder": "text_encoder"}, (where, kw)
    print("OK test_the_encoder_loads_in_bf16_from_its_subfolder")


def test_changing_the_encoder_frees_the_pipe_and_the_cache():
    old = (P.TEXT_ENCODER, P._BASE_PIPE, P._TEXT_ENCODER_ACTIVE)
    try:
        P.TEXT_ENCODER = ""
        P._BASE_PIPE = object()
        P._TEXT_ENCODER_ACTIVE = r"D:\enc\previous"
        P._EMBED_CACHE[("k",)] = ("v",)
        P.set_text_encoder(r"D:\enc\qwen3-vl-abl")
        assert P.TEXT_ENCODER == r"D:\enc\qwen3-vl-abl"
        assert P._BASE_PIPE is None, "le pipeline doit etre libere"
        assert not P._EMBED_CACHE, "les anciens encodages resteraient servis"
        assert P._TEXT_ENCODER_ACTIVE == "", "plus de pipe -> plus d'encodeur charge"
        # the same value: nothing moves, no pointless reload
        sentinel = P._BASE_PIPE = object()
        P.set_text_encoder(r"D:\enc\qwen3-vl-abl")
        assert P._BASE_PIPE is sentinel
    finally:
        P.TEXT_ENCODER, P._BASE_PIPE, P._TEXT_ENCODER_ACTIVE = old
        P._EMBED_CACHE.clear()
    print("OK test_changing_the_encoder_frees_the_pipe_and_the_cache")


class FakePipe:
    def __init__(self):
        self.text_encoder = object()
        self._execution_device = "cpu"
        self.n = 0

    def encode_prompt(self, prompt=None, device=None, **kw):
        self.n += 1
        return tuple(torch.zeros(1, 4, 12, 8) for _ in P._EMBED_OUTS)


def test_the_embed_key_carries_the_encoder():
    """The same prompt, the same pipe object, two encoders: two encodings."""
    P._embed_cache_clear()
    old = (P._TEXT_ENCODER_ACTIVE, P._EMBED_CACHE_MAX)
    try:
        P._EMBED_CACHE_MAX = 8
        pipe = FakePipe()
        P._TEXT_ENCODER_ACTIVE = ""
        P._cached_prompt_embeds(pipe, "p", {})
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 1, pipe.n
        P._TEXT_ENCODER_ACTIVE = r"D:\enc\qwen3-vl-abl"
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 2, "un encodage de l'autre encodeur a ete resservi"
    finally:
        P._TEXT_ENCODER_ACTIVE, P._EMBED_CACHE_MAX = old
        P._embed_cache_clear()
    print("OK test_the_embed_key_carries_the_encoder")


def test_metadata_names_the_encoder_that_ran_and_never_its_path():
    old = (P.TEXT_ENCODER, P._TEXT_ENCODER_ACTIVE, P.ZIMAGE_TRANSFORMER)
    path = r"C:\Users\someone\models\text_encoders\qwen3-vl-4b-abliterated"
    try:
        P.ZIMAGE_TRANSFORMER = None
        P.TEXT_ENCODER = P._TEXT_ENCODER_ACTIVE = path
        m = P._gen_meta("txt2img", "p")
        assert m["text_encoder"] == "qwen3-vl-4b-abliterated", m
        assert "someone" not in json.dumps(m), "chemin local dans les metadonnees"
        # independent of the transformer override (which alone makes base_repo written)
        assert "base_repo" not in m, m
        P.ZIMAGE_TRANSFORMER = r"F:\models\un_checkpoint.safetensors"
        m = P._gen_meta("txt2img", "p")
        assert m["text_encoder"] == "qwen3-vl-4b-abliterated", m
        assert m["base_repo"] == P.BASE_REPO, m
        P.ZIMAGE_TRANSFORMER = None
        # asked for but discarded at load time: named apart
        P._TEXT_ENCODER_ACTIVE = ""
        m = P._gen_meta("txt2img", "p")
        assert "text_encoder" not in m, m
        assert m["text_encoder_not_applied"] == "qwen3-vl-4b-abliterated", m
        P.TEXT_ENCODER = ""
        m = P._gen_meta("txt2img", "p")
        assert "text_encoder" not in m and "text_encoder_not_applied" not in m, m
    finally:
        P.TEXT_ENCODER, P._TEXT_ENCODER_ACTIVE, P.ZIMAGE_TRANSFORMER = old
    assert P._encoder_label(r"D:\m\qwen3-vl-uncensored\text_encoder") == "qwen3-vl-uncensored"
    assert P._encoder_label("owner/repo/sub") == "owner/repo/sub"
    line = cz_imageio._a1111_parameters({"prompt": "p", "text_encoder": "qwen3-vl-4b-abliterated"})
    assert "Text encoder: qwen3-vl-4b-abliterated" in line, line
    print("OK test_metadata_names_the_encoder_that_ran_and_never_its_path")


def test_the_list_finds_encoder_folders():
    d = _folder(QWEN3VL_4B, name="qwen3-vl-4b-abliterated")
    root = os.path.dirname(d)
    os.makedirs(os.path.join(root, "empty"))
    old = P.TEXT_ENCODERS_DIR
    try:
        P.TEXT_ENCODERS_DIR = root
        found = P.list_text_encoders()
    finally:
        P.TEXT_ENCODERS_DIR = old
    assert d in found, found
    assert not any(f.endswith("empty") for f in found), found
    print("OK test_the_list_finds_encoder_folders")


def test_the_queue_keeps_the_encoder():
    import cz_ui as U
    calls = []
    old = (P.TEXT_ENCODER, P.set_text_encoder)
    try:
        P.TEXT_ENCODER = r"D:\enc\qwen3-vl-abl"
        ms = U._q_model_state()
        assert ms["text_encoder"] == r"D:\enc\qwen3-vl-abl", ms
        P.set_text_encoder = lambda s: calls.append(s)
        U._q_restore_model_state(ms)
        assert calls == [r"D:\enc\qwen3-vl-abl"], calls
        # a snapshot from before the option: we do not touch the current encoder
        calls.clear()
        U._q_restore_model_state({k: v for k, v in ms.items() if k != "text_encoder"})
        assert calls == [], calls
    finally:
        P.TEXT_ENCODER, P.set_text_encoder = old
    print("OK test_the_queue_keeps_the_encoder")


def test_the_ui_persists_only_a_valid_encoder():
    """A named refusal: nothing changed, nothing written to preferences.json. Otherwise applied and
    remembered. _save_prefs_keys is stubbed: no file of the repo is touched."""
    import cz_ui as U
    saved, applied = [], []
    old = (U._save_prefs_keys, P.set_text_encoder, P.TEXT_ENCODER)
    try:
        U._save_prefs_keys = lambda d: saved.append(dict(d))
        P.set_text_encoder = lambda s: applied.append(s)
        P.TEXT_ENCODER = ""
        with _Base(QWEN3VL_4B):
            msg = U._ui_set_text_encoder(_folder(QWEN3VL_8B, name="qwen3-vl-8b"))
            assert "not applied" in msg and "4096" in msg and "2560" in msg, msg
            assert saved == [] and applied == [], (saved, applied)
            ok = _folder(QWEN3VL_4B, name="qwen3-vl-4b-abliterated")
            msg = U._ui_set_text_encoder(ok)
            assert "qwen3-vl-4b-abliterated" in msg, msg
            assert applied == [ok] and saved == [{"text_encoder": ok}], (applied, saved)
            # back to the default: applied and remembered too
            U._ui_set_text_encoder("")
            assert applied[-1] == "" and saved[-1] == {"text_encoder": ""}, (applied, saved)
    finally:
        U._save_prefs_keys, P.set_text_encoder, P.TEXT_ENCODER = old
    print("OK test_the_ui_persists_only_a_valid_encoder")


# --- _ensure_base: the encoder arrives in from_pretrained, or the fallback ------------

class FakeKrea2Pipeline:
    """A dummy Krea2Pipeline: it remembers from_pretrained's kwargs."""
    last = None

    def __init__(self, kw):
        self.transformer = kw.get("transformer")
        self.text_encoder = kw.get("text_encoder", "BASE_ENCODER")
        self.scheduler = object()      # no .config -> _apply_sampler does nothing
        self.vae = types.SimpleNamespace(config=types.SimpleNamespace(),
                                         enable_slicing=lambda: None,
                                         enable_tiling=lambda: None)

    @classmethod
    def from_pretrained(cls, repo, **kw):
        cls.last = (repo, kw)
        return cls(kw)

    def to(self, dev):
        return self


_ENSURE_STATE = ("TEXT_ENCODER", "_TEXT_ENCODER_ACTIVE", "_BASE_PIPE", "_DERIVED",
                 "_LOADED_KEY", "_BASE_SCHED_CONFIG", "_APPLIED_LORAS", "_APPLIED_LOKRS",
                 "ZIMAGE_TRANSFORMER", "LORAS", "OFFLOAD_MODE", "_load_transformer",
                 "_encoder_class")


def _run_ensure_base(src, base_cfg, boom=None):
    """_ensure_base on a fake pipeline: (pipe, (repo, kwargs), the active one, the metadata)."""
    import diffusers
    had = "Krea2Pipeline" in vars(diffusers)
    old_attr = vars(diffusers).get("Krea2Pipeline")
    saved = {k: getattr(P, k) for k in _ENSURE_STATE}
    _FakeEncoderClass.calls, _FakeEncoderClass.boom = [], boom
    FakeKrea2Pipeline.last = None
    try:
        diffusers.Krea2Pipeline = FakeKrea2Pipeline
        P._load_transformer = lambda: "TRANSFORMER"
        P._encoder_class = lambda base=None: _FakeEncoderClass
        P.free_vram()
        P.ZIMAGE_TRANSFORMER, P.LORAS, P.OFFLOAD_MODE = None, [], "none"
        P.TEXT_ENCODER = src
        with _Base(base_cfg):
            pipe = P._ensure_base()
        return pipe, FakeKrea2Pipeline.last, P._TEXT_ENCODER_ACTIVE, P._gen_meta("txt2img", "p")
    finally:
        for k, v in saved.items():
            setattr(P, k, v)
        if had:
            diffusers.Krea2Pipeline = old_attr
        else:
            vars(diffusers).pop("Krea2Pipeline", None)
        P._embed_cache_clear()


def test_ensure_base_hands_the_encoder_to_from_pretrained():
    d = _folder(QWEN3VL_4B, name="qwen3-vl-4b-abliterated")
    pipe, (repo, kw), active, meta = _run_ensure_base(d, QWEN3VL_4B)
    assert repo == P.BASE_REPO, repo
    assert isinstance(kw.get("text_encoder"), _FakeEncoderClass), kw
    assert pipe.text_encoder is kw["text_encoder"]
    assert kw["transformer"] == "TRANSFORMER", "le transformer (quantifie) reste charge a part"
    assert active == d, active
    assert meta["text_encoder"] == "qwen3-vl-4b-abliterated", meta
    print("OK test_ensure_base_hands_the_encoder_to_from_pretrained")


def test_a_misfit_or_failing_encoder_falls_back_to_the_base_one():
    """Never a render lost for an encoder: refused at the config (without reading the weights)
    or failing at load time, _ensure_base loads the base repo's and says so."""
    for src, boom in ((_folder(QWEN3VL_8B, name="qwen3-vl-8b"), None),
                      (_folder(QWEN3VL_4B, name="qwen3-vl-broken"), OSError("truncated shard"))):
        pipe, (repo, kw), active, meta = _run_ensure_base(src, QWEN3VL_4B, boom=boom)
        assert pipe is not None and pipe.text_encoder == "BASE_ENCODER"
        assert "text_encoder" not in kw, kw
        assert active == "", active
        assert "text_encoder" not in meta, meta
        assert meta["text_encoder_not_applied"] == os.path.basename(src), meta
        if boom is None:
            assert not _FakeEncoderClass.calls, "refuse a la config: aucun poids lu"
    # with no replacement encoder, nothing changes
    pipe, (repo, kw), active, meta = _run_ensure_base("", QWEN3VL_4B)
    assert "text_encoder" not in kw and active == "", kw
    assert "text_encoder" not in meta and "text_encoder_not_applied" not in meta, meta
    print("OK test_a_misfit_or_failing_encoder_falls_back_to_the_base_one")



def test_default_picked_in_the_ui_survives_a_restart():
    """Choosing "Default" writes "" into the preferences: on a restart, a value from
    config.txt must not come back over it. The environment always wins."""
    cfg = {"text_encoder": r"D:\enc\from-config"}
    assert P._resolve_text_encoder({}, {}, cfg) == r"D:\enc\from-config"
    assert P._resolve_text_encoder({}, {"text_encoder": ""}, cfg) == ""
    assert P._resolve_text_encoder({}, {"text_encoder": r"D:\enc\ui"}, cfg) == r"D:\enc\ui"
    assert P._resolve_text_encoder({"KREA2_TEXT_ENCODER": r"D:\enc\env"},
                                   {"text_encoder": ""}, cfg) == r"D:\enc\env"
    print("OK test_default_picked_in_the_ui_survives_a_restart")


def test_compatible_encoders_in_the_hf_cache_are_listed():
    """An encoder downloaded from HF lives in the HF cache: the list must show it.
    Not a diffusers pipeline, not a config with no weights; another size is named next to it."""
    import json as _json
    import os as _os
    import tempfile as _tempfile
    ref = {"model_type": "fam", "hidden_size": 64, "num_hidden_layers": 2}
    wide = {"model_type": "fam", "hidden_size": 128, "num_hidden_layers": 2}
    root = _tempfile.mkdtemp(prefix="hfcache_")

    def snap(repo, sub=None, cfg=ref, weights=True, pipeline=False):
        d = _os.path.join(root, "models--" + repo.replace("/", "--"), "snapshots", "r1")
        p = _os.path.join(d, sub) if sub else d
        _os.makedirs(p, exist_ok=True)
        with open(_os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
            _json.dump(cfg, f)
        if weights:
            open(_os.path.join(p, "model.safetensors"), "wb").close()
        if pipeline:
            with open(_os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
                f.write("{}")

    snap("a/fits")
    snap("b/fits-in-sub", sub="enc")
    snap("c/wider", cfg=wide)
    snap("d/pipeline", sub="text_encoder", pipeline=True)
    snap("e/config-only", weights=False)
    snap("f/no-shape", cfg={"_class_name": "AutoencoderKL"})
    old = (P._hf_cache_dir, P._base_text_encoder_config)
    try:
        P._hf_cache_dir = lambda: root
        P._base_text_encoder_config = lambda base=None: ref
        got = [v for _l, v in P.list_cached_text_encoders()]
        other, width = P.cached_text_encoder_mismatches()
        import cz_ui as U
        hint = U._te_hint()
        choices = [v for _l, v in U._te_choices()]
    finally:
        P._hf_cache_dir, P._base_text_encoder_config = old
    assert got == ["a/fits", "b/fits-in-sub/enc"], got
    assert all(v in choices for v in got), choices
    assert [h for h, _w in other] == ["c/wider"] and width == 64, (other, width)
    assert "128" in hint and "64" in hint and "c/wider" in hint, hint
    print("OK test_compatible_encoders_in_the_hf_cache_are_listed")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All text-encoder tests passed.")
