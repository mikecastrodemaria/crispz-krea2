"""Encodeur texte de remplacement (Models > Checkpoints > Text encoder).

Porte depuis crispz-klein 1.34.0. Krea 2 lit DOUZE etats caches de l'encodeur Qwen3-VL a
des indices FIXES (text_encoder_select_layers = 2, 5, ..., 35, autant que
transformer.config.num_text_layers), larges de 2560: un autre encodeur ne se branche que
s'il a la meme famille (qwen3_vl), la meme largeur et le meme nombre de couches. Moins
de couches: les indices debordent au premier prompt. Plus: le transformer lit d'autres
profondeurs que celles de son entrainement, sans erreur. Le refus doit le dire AVANT de
lire 8 Go.

Ces tests verrouillent aussi ce qui rendrait l'option dangereuse en silence:
  - un changement d'encodeur vide le cache d'embeddings (sinon les anciens encodages
    restent servis) et l'encodeur fait partie de la CLE du cache -- id(enc) seul ne
    suffit pas, CPython recycle les id d'objets liberes;
  - un encodeur ecarte, ou qui echoue au chargement, ne coute jamais un rendu:
    _ensure_base retombe sur celui du repo de base, et les metadonnees le disent;
  - les metadonnees nomment l'encodeur qui a REELLEMENT tourne, par son nom de dossier
    et jamais par son chemin (qui finirait dans les PNG partages);
  - la file garde l'encodeur du job.

Aucun modele charge, aucun reseau: la config de l'encodeur du repo de base est remplacee,
et Krea2Pipeline comme la classe de l'encodeur sont des faux.

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

# Encodeur de krea/Krea-2-Turbo (et -Raw): la partie texte est rangee sous text_config,
# la vision (1024 de large) ne compte pas.
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
    """Remplace la config de l'encodeur du repo de base (pas de reseau, pas de HF)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __enter__(self):
        self.old = P._base_text_encoder_config
        P._base_text_encoder_config = lambda base=None: self.cfg

    def __exit__(self, *a):
        P._base_text_encoder_config = self.old


class _FakeEncoderClass:
    """Classe d'encodeur factice: retient ce que from_pretrained recoit, ou echoue."""
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
        # poids dans un sous-dossier text_encoder/ (copie d'un repo diffusers)
        assert P._text_encoder_problem(_folder(QWEN3VL_4B, "text_encoder")) is None
    print("OK test_same_architecture_is_accepted")


def test_another_width_is_refused_with_both_numbers():
    with _Base(QWEN3VL_4B):
        why = P._text_encoder_problem(_folder(QWEN3VL_8B))
    assert why and "4096" in why and "2560" in why, why
    print("OK test_another_width_is_refused_with_both_numbers")


def test_the_layer_count_is_checked_both_ways():
    """Indices FIXES jusqu'a 35: moins de couches deborde, plus lit d'autres
    profondeurs sans erreur. Les deux sont refuses, nombres a l'appui."""
    with _Base(QWEN3VL_4B):
        for n in (28, 40):
            why = P._text_encoder_problem(_folder(_layers(n)))
            assert why and str(n) in why and "36" in why, why
    print("OK test_the_layer_count_is_checked_both_ways")


def test_a_text_only_qwen3_is_refused_by_family():
    """Meme largeur, meme profondeur, autre architecture: la famille suffit."""
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
    """Rien a comparer: pas de refus, le chargement tranchera (et retombera au besoin)."""
    with _Base(None):
        assert P._text_encoder_problem(_folder(QWEN3VL_8B)) is None
    print("OK test_an_unreadable_base_config_leaves_it_to_the_load")


def test_hf_ids_may_carry_a_subfolder():
    assert P._split_hf_src("owner/repo") == ("owner/repo", None)
    assert P._split_hf_src("owner/repo/text_encoder") == ("owner/repo", "text_encoder")
    assert P._split_hf_src("owner/repo/a/b") == ("owner/repo", "a/b")
    print("OK test_hf_ids_may_carry_a_subfolder")


def test_the_class_comes_from_the_base_repo_model_index():
    """La classe est celle que diffusers aurait chargee, lue dans model_index.json
    (Qwen3VLModel chez Krea 2). Module factice: importer la vraie classe transformers
    tire torchao, qui plante sans GPU visible."""
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
        # model_index sans text_encoder: erreur nommee, pas un KeyError nu
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
    """DTYPE et le sous-dossier, rien d'autre: torchao ne touche que le transformer."""
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
        # meme valeur: rien ne bouge, pas de rechargement inutile
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
    """Meme prompt, meme objet pipe, deux encodeurs: deux encodages."""
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
        # independant du transformer override (qui seul fait ecrire base_repo)
        assert "base_repo" not in m, m
        P.ZIMAGE_TRANSFORMER = r"F:\models\un_checkpoint.safetensors"
        m = P._gen_meta("txt2img", "p")
        assert m["text_encoder"] == "qwen3-vl-4b-abliterated", m
        assert m["base_repo"] == P.BASE_REPO, m
        P.ZIMAGE_TRANSFORMER = None
        # demande mais ecarte au chargement: nomme a part
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
        # snapshot d'avant l'option: on ne touche pas a l'encodeur courant
        calls.clear()
        U._q_restore_model_state({k: v for k, v in ms.items() if k != "text_encoder"})
        assert calls == [], calls
    finally:
        P.TEXT_ENCODER, P.set_text_encoder = old
    print("OK test_the_queue_keeps_the_encoder")


def test_the_ui_persists_only_a_valid_encoder():
    """Refus nomme: rien de change, rien d'ecrit dans preferences.json. Sinon applique et
    memorise. _save_prefs_keys est remplace: aucun fichier du depot n'est touche."""
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
            # retour au defaut: applique et memorise aussi
            U._ui_set_text_encoder("")
            assert applied[-1] == "" and saved[-1] == {"text_encoder": ""}, (applied, saved)
    finally:
        U._save_prefs_keys, P.set_text_encoder, P.TEXT_ENCODER = old
    print("OK test_the_ui_persists_only_a_valid_encoder")


# --- _ensure_base: l'encodeur arrive dans from_pretrained, ou le repli ---------------

class FakeKrea2Pipeline:
    """Krea2Pipeline factice: retient les kwargs de from_pretrained."""
    last = None

    def __init__(self, kw):
        self.transformer = kw.get("transformer")
        self.text_encoder = kw.get("text_encoder", "BASE_ENCODER")
        self.scheduler = object()      # pas de .config -> _apply_sampler ne fait rien
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
    """_ensure_base sur un faux pipeline: (pipe, (repo, kwargs), actif, metadonnees)."""
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
    """Jamais un rendu perdu pour un encodeur: refuse a la config (sans lire les poids)
    ou en echec au chargement, _ensure_base charge celui du repo de base et le dit."""
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
    # sans encodeur de remplacement, rien ne change
    pipe, (repo, kw), active, meta = _run_ensure_base("", QWEN3VL_4B)
    assert "text_encoder" not in kw and active == "", kw
    assert "text_encoder" not in meta and "text_encoder_not_applied" not in meta, meta
    print("OK test_a_misfit_or_failing_encoder_falls_back_to_the_base_one")



def test_default_picked_in_the_ui_survives_a_restart():
    """Choisir "Default" ecrit "" dans les preferences: au redemarrage, une valeur de
    config.txt ne doit pas revenir par-dessus. L'environnement gagne toujours."""
    cfg = {"text_encoder": r"D:\enc\from-config"}
    assert P._resolve_text_encoder({}, {}, cfg) == r"D:\enc\from-config"
    assert P._resolve_text_encoder({}, {"text_encoder": ""}, cfg) == ""
    assert P._resolve_text_encoder({}, {"text_encoder": r"D:\enc\ui"}, cfg) == r"D:\enc\ui"
    assert P._resolve_text_encoder({"KREA2_TEXT_ENCODER": r"D:\enc\env"},
                                   {"text_encoder": ""}, cfg) == r"D:\enc\env"
    print("OK test_default_picked_in_the_ui_survives_a_restart")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All text-encoder tests passed.")
