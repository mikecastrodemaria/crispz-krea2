"""crispz-krea2 - coeur Krea 2 (diffusers, BF16 + quantif fp8): chargement du pipeline
txt2img, LoRA / modele de base, generation, + l'etat mutable runtime.

Fork de crispz-qwen-edit. Krea 2 partage le VAE (AutoencoderKLQwenImage) et la famille
d'encodeur texte (Qwen3-VL) de Qwen-Image, d'ou le choix de cette base.

  - base txt2img -> Krea2Pipeline (Krea2Transformer2DModel, 12.9B)

CE QUE KREA 2 NE SAIT PAS FAIRE (diffusers n'expose AUCUN de ces pipelines) :
  - img2img  -> pas de Krea2Img2ImgPipeline  => refine / upscale-refine indisponibles
  - inpaint  -> pas de Krea2InpaintPipeline  => inpaint / outpaint / reframe indisponibles
  - edition par instruction (onglet Omni)    => indisponible
Ces capacites sont declarees dans CAPABILITIES ci-dessous ; cz_ui s'en sert pour MASQUER
les onglets/controles correspondants, et get_pipe() leve UnsupportedFeature en filet de
securite (CLI, API, appels directs).

CHARGEMENT: from_pretrained sur un repo/dossier diffusers. Krea2Transformer2DModel
n'a pas de from_single_file upstream -> les .safetensors Civitai (bf16 et FP8/INT8
'scaled', ConvRot compris) passent par NOTRE conversion en dossier diffusers, en
cache disque (cf. _converted_folder), et les .gguf ComfyUI-GGUF (arch 'krea2')
suivent le meme chemin, dequantifies en bf16 par la lib gguf. Voir README.
Le bf16 seul deborde une carte 32 Go (35,6 Go au pic, ~59 s/step) -> on quantifie a la
volee via torchao (fp8 weight-only): 22,8 Go, ~2,4 s/step, qualite equivalente.

CFG: convention Krea 2 -> `guidance_scale` DIRECT (velocite = cond + g*(cond-uncond),
guidance actif des que g > 0). Ce n'est PAS la convention Qwen (true_cfg_scale + un
guidance_scale distille a 1.0). Turbo est distille -> g = 0.0 et 8 steps.

L'API publique du module reste identique a l'upstream (memes noms, ex.
ZIMAGE_TRANSFORMER, SAMPLER_CHOICES) pour ne casser ni cz_ui ni cz_cli.

app lit l'etat courant via cz_pipeline.NAME (BASE_REPO, ZIMAGE_TRANSFORMER, ...) et pose
cz_pipeline._PROGRESS / cz_pipeline._STOP depuis les handlers UI.
Ne depend que de cz_core / cz_esrgan / cz_imageio (jamais de app ni de gradio).
"""

import os
import gc
import sys
import time
import json
import threading

import numpy as np
import torch
from PIL import Image

import cz_core
from cz_core import (
    CONFIG, HERE, DEVICE, DTYPE,
    DEFAULT_TILE, DEFAULT_OVERLAP, DEFAULT_REFINE_TILE, DEFAULT_REFINE_OVERLAP,
    _prefs, _is_single_file, _log, _dbg,
)

# Modele Krea 2 de base (txt2img). Turbo par defaut: distille, 8 steps, guidance 0.
# Raw = checkpoint mid-training non distille (28-52 steps + CFG) -> beaucoup plus lent,
# surtout destine au fine-tuning / entrainement de LoRA. Repos GATED: il faut accepter la
# licence sur huggingface.co AVEC LE COMPTE DU TOKEN (cf. README). Surcharge via env
# ZIMAGE_MODEL (compat) ou KREA_MODEL, ou prefs.
DEFAULT_BASE_REPO = (os.environ.get("KREA_MODEL") or "krea/Krea-2-Turbo")
# Pas de modele d'edition par instruction chez Krea 2 (conserve pour compat d'API).
DEFAULT_OMNI_REPO = None

# ----------------------------------------------------------------------------
# Capacites de CETTE famille de modele. cz_ui lit ce dict pour masquer les onglets
# et controles sans pipeline derriere -> une seule source de verite, pas de liste
# d'onglets caches codee en dur dans l'UI.
# ----------------------------------------------------------------------------
CAPABILITIES = {
    "txt2img": True,
    "img2img": False,   # pas de Krea2Img2ImgPipeline  -> refine, upscale-refine, harmonize
    "inpaint": False,   # pas de Krea2InpaintPipeline  -> inpaint, outpaint, reframe(contain)
    "omni": False,      # pas d'edition par instruction
    "lora": True,       # Krea2Transformer2DModel herite de PeftAdapterMixin
    "single_file": False,   # pas de FromOriginalModelMixin -> ni .safetensors Civitai ni GGUF
    "esrgan": True,     # upscale pur ESRGAN: independant du modele de diffusion
}


class UnsupportedFeature(RuntimeError):
    """Feature absente de cette famille de modele. Levee par get_pipe() en filet de
    securite: l'UI masque deja les controles concernes via CAPABILITIES."""


def supports(feature):
    """True si la famille courante expose `feature` (cle de CAPABILITIES)."""
    return bool(CAPABILITIES.get(feature, False))
from cz_esrgan import load_esrgan, esrgan_upscale
from cz_imageio import _now_stamp

# Vitesse: autorise TF32 (matmul/cudnn) sur GPU. Gain gratuit sur Ampere+ pour les
# operations fp32 residuelles; les poids restent BF16. Sans effet hors CUDA.
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# Modele Z-Image courant. Un repo HF / dossier diffusers -> BASE_REPO. Un fichier
# single-file (.safetensors Civitai) passe comme "modele" -> transformer override
# (le VAE et l'encodeur Qwen3 restent tires du repo de base).
_zmodel = os.environ.get("ZIMAGE_MODEL") or _prefs.get("zimage_model") or DEFAULT_BASE_REPO
ZIMAGE_TRANSFORMER = os.environ.get("ZIMAGE_TRANSFORMER") or _prefs.get("zimage_transformer") or None
if _is_single_file(_zmodel):
    ZIMAGE_TRANSFORMER = _zmodel
    BASE_REPO = DEFAULT_BASE_REPO
else:
    BASE_REPO = _zmodel

# Dossiers de modeles Z-Image: checkpoints single-file a switcher + LoRA a appliquer.
CHECKPOINTS_DIR = (os.environ.get("CHECKPOINTS_DIR") or _prefs.get("checkpoints_dir")
                   or CONFIG.get("checkpoints_dir") or os.path.join(HERE, "checkpoints"))
# Dossier checkpoints supplementaire (optionnel) -> fusionne avec CHECKPOINTS_DIR dans
# la meme liste de checkpoints. Vide par defaut; configurable via UI / prefs / config / env.
CHECKPOINTS_EXTRA_DIR = (os.environ.get("CHECKPOINTS_EXTRA_DIR") or _prefs.get("checkpoints_extra_dir")
                         or CONFIG.get("checkpoints_extra_dir") or "").strip()
LORAS_DIR = (os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
             or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))
# LoRA actives: liste de (chemin, poids). Plusieurs LoRA combinables (multi-slots).
LORAS = []
LORA_WEIGHT = float(CONFIG.get("default_lora_weight", 1.0))  # poids par defaut des slots


def _lora_weight_range():
    """Bornes des curseurs de poids LoRA (config 'lora_weight_min'/'lora_weight_max').
    Defaut -2..2: les poids NEGATIFS sont valides et utiles (ils inversent l'effet de la
    LoRA). Defensif: valeurs illisibles ou min >= max -> on retombe sur le defaut."""
    try:
        lo = float(CONFIG.get("lora_weight_min", -2.0))
        hi = float(CONFIG.get("lora_weight_max", 2.0))
    except (TypeError, ValueError):
        _log("lora_weight_min/max: not a number, using -2..2")
        return -2.0, 2.0
    if lo >= hi:
        _log(f"lora_weight_min ({lo}) >= lora_weight_max ({hi}), using -2..2")
        return -2.0, 2.0
    return lo, hi


LORA_WEIGHT_MIN, LORA_WEIGHT_MAX = _lora_weight_range()
# Le poids par defaut doit rester dans les bornes (sinon le curseur naitrait hors plage).
LORA_WEIGHT = min(LORA_WEIGHT_MAX, max(LORA_WEIGHT_MIN, LORA_WEIGHT))
# LoRA appliquees AU DEMARRAGE (ex. Lightning 8-step). config 'default_loras' = liste de
# noms (dans LORAS_DIR) ou de paires [nom, poids]. Resolues en (chemin, poids).
for _spec in (CONFIG.get("default_loras") or []):
    _nm, _w = (_spec if isinstance(_spec, (list, tuple)) and len(_spec) == 2
               else (_spec, LORA_WEIGHT))
    if _nm and _nm not in ("None", "none"):
        _p = _nm if os.path.isabs(_nm) else os.path.join(LORAS_DIR, _nm)
        if os.path.isfile(_p):
            LORAS.append((_p, float(_w)))
# Modele Omni/Edit: aucun chez Krea 2 (pas d'equivalent Qwen-Image-Edit). Reste vide ->
# l'onglet correspondant n'apparait pas (cf. omni_on dans cz_ui). Conserve comme variable
# car cz_ui / cz_cli la lisent.
OMNI_MODEL = (os.environ.get("ZIMAGE_OMNI_MODEL") or CONFIG.get("zimage_omni_model")
              or DEFAULT_OMNI_REPO or "").strip()

# Caches process-wide. Un pipeline "base" (txt2img ZImagePipeline) detient les
# composants; img2img / inpaint en derivent via from_pipe -> poids partages, pas de
# VRAM en double. Clef de cache = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE, LORAS).
_BASE_PIPE = None
_DERIVED = {}
_LOADED_KEY = None
# LoRA reellement posees sur _BASE_PIPE (liste de (chemin, poids)). Sert a echanger les
# LoRA a chaud sans recharger le modele: si ca diverge de LORAS, _apply_loras resynchronise.
_APPLIED_LORAS = []

# Palier 2 (cohabitation VRAM): offload CPU de la passe diffusion. none = tout en VRAM.
# model = decharge par sous-module (bon compromis). sequential = plus agressif, plus lent.
# N'est PAS de la quantif: les poids restent BF16, ils transitent RAM <-> GPU.
# Krea 2 est gros (12.9B, 26 Go en bf16) : on initialise depuis la config (default_cpu_offload, defaut
# 'model' pour ce fork) ou l'env CZ_OFFLOAD -> offload actif DES le 1er chargement (anti-OOM).
OFFLOAD_CHOICES = ("none", "model", "sequential")
OFFLOAD_MODE = (os.environ.get("CZ_OFFLOAD") or CONFIG.get("default_cpu_offload") or "none")
if OFFLOAD_MODE not in OFFLOAD_CHOICES:
    OFFLOAD_MODE = "none"

# Guidance Krea 2. Convention DIFFERENTE de Qwen: le curseur de l'UI pilote directement
# `guidance_scale`, la velocite valant cond + g*(cond-uncond); la guidance est active des
# que g > 0 (equivaut au CFG usuel d'echelle 1+g). Turbo est DISTILLE -> 0.0 (pas de CFG,
# un seul forward par step). Raw (non distille) veut ~4.5. 0.0 est donc une valeur VALIDE
# ici, d'ou l'absence de repli "or 4.0". Override via env KREA_CFG.
_g = os.environ.get("KREA_CFG")
if _g is None:
    _g = CONFIG.get("default_guidance")
GUIDANCE = float(_g) if _g not in (None, "") else 0.0

# Quantification du transformer au chargement (torchao). Krea 2 en bf16 pese 26 Go et
# DEBORDE une carte 32 Go (pic 35,6 Go -> spill RAM via PCIe, ~59 s/step). En fp8
# weight-only: 22,8 Go et ~2,4 s/step, pour une qualite visuellement equivalente (les
# activations restent en bf16). Mesure sur RTX 5090 / 1024x1024 / 8 steps.
#   "float8_weight_only" (defaut, natif sur sm_89+/Blackwell) | "int8_weight_only"
#   | "int4_weight_only" | "float8_dynamic" | "none" (bf16 brut, exige >32 Go de VRAM)
QUANT_MODE = (os.environ.get("KREA_QUANT") or CONFIG.get("quantization")
              or "float8_weight_only").strip().lower()
QUANT_CHOICES = ("float8_weight_only", "int8_weight_only", "int4_weight_only",
                 "float8_dynamic", "none")
if QUANT_MODE not in QUANT_CHOICES:
    QUANT_MODE = "float8_weight_only"


def set_quant_mode(mode):
    """Change le schema de quantification. Invalide le pipe (rechargement au prochain run)."""
    global QUANT_MODE
    mode = (mode or "none").strip().lower()
    if mode not in QUANT_CHOICES or mode == QUANT_MODE:
        return
    QUANT_MODE = mode
    free_vram()
    _log(f"quantization -> {QUANT_MODE} (reload on next run)")


def _quant_config():
    """TorchAoConfig pour QUANT_MODE, ou None si desactive/indisponible.
    torchao >= 0.17 exige un objet AOBaseConfig (les chaines ne sont plus acceptees)."""
    if QUANT_MODE == "none":
        return None
    try:
        from diffusers import TorchAoConfig
        import torchao.quantization as q
    except Exception as e:
        _log(f"[WARN] torchao unavailable ({e}) -> loading in bf16. "
             "Krea 2 in bf16 needs >32 GB of VRAM: expect a very slow spill to system RAM.")
        return None
    cls = {
        "float8_weight_only": getattr(q, "Float8WeightOnlyConfig", None),
        "int8_weight_only": getattr(q, "Int8WeightOnlyConfig", None),
        "int4_weight_only": getattr(q, "Int4WeightOnlyConfig", None),
        "float8_dynamic": getattr(q, "Float8DynamicActivationFloat8WeightConfig", None),
    }.get(QUANT_MODE)
    if cls is None:
        _log(f"[WARN] scheme '{QUANT_MODE}' not found in torchao -> bf16")
        return None
    return TorchAoConfig(cls())

# Force ratio (facon Fooocus) pour upscale/img2img: si defini, l'image d'ENTREE est
# recadree au centre a ce ratio avant traitement (crop to fit). Vide = ratio natif preserve
# (defaut). Format: 'W:H' ou 'WxH' (ex. '13:19', '832x1216'). Pilotable par l'UI (case a
# cocher + dropdown Aspect ratio) via set_force_ratio, ou par config.txt 'force_upscale_ratio'.
FORCE_RATIO = (os.environ.get("CZ_FORCE_RATIO") or CONFIG.get("force_upscale_ratio") or "").strip()
# Comment atteindre le ratio force: 'crop' = recadrage centre (perd les bords, defaut),
# 'extend' = outpaint des bandes manquantes. NB Krea 2: pas de pipeline inpaint ->
# 'extend' leve le message UnsupportedFeature clair (l'UI ne propose que Off/Crop).
FORCE_RATIO_MODE = (os.environ.get("CZ_FORCE_RATIO_MODE")
                    or CONFIG.get("force_ratio_mode") or "crop").strip().lower()
# Passe de fusion des raccords du mode extend (sans objet tant que Krea 2 n'a pas
# d'inpaint, garde pour parite de config avec la famille). 0 = desactive.
try:
    EXTEND_DENOISE = float(CONFIG.get("force_ratio_extend_denoise", 0.22) or 0.0)
except Exception:
    EXTEND_DENOISE = 0.22

# Sampler / scheduler. Le pipeline Z-Image impose un schedule `sigmas` custom: seuls
# les schedulers dont set_timesteps accepte `sigmas` fonctionnent. En pratique -> Euler
# flow-matching (natif, defaut), UniPC (multistep) et LCM flow-matching (interessant sur
# les modeles distilles/Turbo: peu de steps, guidance ~0-1).
# Les DPM++ 2M / DPM2a / DPM++ SDE (dpmpp_sde) de diffusers ne prennent PAS de sigmas
# custom -> incompatibles (DPMSolverSDEScheduler exige en plus torchsde). Non exposes.
SAMPLER_CHOICES = ("euler", "unipc", "lcm")
SAMPLER = (os.environ.get("ZIMAGE_SAMPLER") or CONFIG.get("default_sampler") or "euler").strip().lower()
if SAMPLER not in SAMPLER_CHOICES:
    SAMPLER = "euler"

# Schedule de sigmas (= le "scheduler" facon ComfyUI). sgm_uniform = natif Z-Image
# (linspace + dynamic shift). beta/karras/exponential = re-mapping des sigmas applique
# PAR-DESSUS le schedule du pipeline (FlowMatchEuler/UniPC: use_*_sigmas). beta -> scipy.
SCHEDULE_CHOICES = ("sgm_uniform", "beta", "karras", "exponential")
# 'simple' (ComfyUI) designe EXACTEMENT le schedule natif expose ici sous 'sgm_uniform':
# les sigmas par defaut que le pipeline passe au scheduler sont linspace(1, 1/n, n),
# ce que ComfyUI appelle 'simple' sur un modele flow-matching. Accepte
# en entree partout (config/env/CLI/XYZ) pour recopier une recette CivitAI au mot pres,
# mais normalise vers le nom canonique: metadonnees et presets ne portent qu'un seul nom.
_SCHEDULE_ALIASES = {"simple": "sgm_uniform"}
SCHEDULE_INPUTS = SCHEDULE_CHOICES + tuple(_SCHEDULE_ALIASES)   # listes ouvertes (CLI/XYZ)


def _norm_schedule(name, default="sgm_uniform"):
    """Nom de schedule -> nom canonique (alias resolus). Inconnu -> `default`."""
    n = (name or "").strip().lower()
    n = _SCHEDULE_ALIASES.get(n, n)
    return n if n in SCHEDULE_CHOICES else default


SCHEDULE = _norm_schedule(os.environ.get("ZIMAGE_SCHEDULE") or CONFIG.get("default_schedule"))
_SCHEDULE_FLAG = {"beta": "use_beta_sigmas", "karras": "use_karras_sigmas",
                  "exponential": "use_exponential_sigmas"}  # sgm_uniform -> aucun flag (natif)
# Config natif du scheduler du modele (capture au 1er chargement) -> base de construction
# des autres samplers (conserve shift/flow params quel que soit le sampler courant).
_BASE_SCHED_CONFIG = None

# Hook de progression UI (gradio gr.Progress). None hors UI (CLI/serveur). Pose par
# les handlers via cz_pipeline._PROGRESS = ...
_PROGRESS = None
# Stop "facon Fooocus": flag global + interruption des pipelines diffusers. Pose par
# les handlers via cz_pipeline._STOP = ... et par request_stop().
_STOP = False

# Verrou GPU: serialise TOUTES les generations. Gradio ne serialise pas les events de
# LISTENERS differents (Generate manuel vs Run queue vs detaileur): deux threads peuvent
# alors appeler le MEME pipeline partage et stepper le MEME scheduler -> son index
# depasse la fin ("IndexError: index 31 is out of bounds for dimension 0 with size 31",
# scheduling_flow_match_euler_discrete.step). RLock: les imbrications d'un meme thread
# (txt2img_run -> generate, process_one -> _refine_whole) restent libres.
_GPU_LOCK = threading.RLock()


def _gpu_serial(fn):
    """Decorateur: execute fn sous _GPU_LOCK (une seule generation GPU a la fois)."""
    import functools

    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        with _GPU_LOCK:
            return fn(*args, **kwargs)
    return _locked

# Gestion du seed (facon Fooocus):
#  _LAST_SEED         = seed CONCRET du dernier rendu (un -1 aleatoire est resolu en
#                       valeur reelle) -> bouton "Reuse last seed" + metadonnees justes.
#  _NO_SEED_INCREMENT = True -> tout un batch utilise le meme seed (pas de +i par image).
_LAST_SEED = -1
_NO_SEED_INCREMENT = False
# True -> en txt2img+upscale, sauve AUSSI l'image txt2img d'origine (avant l'upscale).
_SAVE_PRE_UPSCALE = bool(CONFIG.get("save_pre_upscale", False))


def set_no_seed_increment(v):
    global _NO_SEED_INCREMENT
    _NO_SEED_INCREMENT = bool(v)


def set_save_pre_upscale(v):
    global _SAVE_PRE_UPSCALE
    _SAVE_PRE_UPSCALE = bool(v)



def set_guidance(g):
    global GUIDANCE
    GUIDANCE = float(g)


def _cfg(negative=None):
    """kwargs CFG pour Krea 2. Convention du modele: `guidance_scale` DIRECT, la velocite
    valant cond + g*(cond-uncond), guidance active des que g > 0. Pas de `true_cfg_scale`
    (ca, c'etait Qwen). A g = 0 (Turbo distille) la guidance est desactivee et le negative
    prompt est ignore par le pipeline -> on ne l'envoie pas, ca evite un forward inutile."""
    g = float(GUIDANCE)
    kw = {"guidance_scale": g}
    if g > 0:
        kw["negative_prompt"] = (negative or None)
    return kw


def _qwen_call(pipe, **kw):
    """Appelle le pipeline en tolerant les variations d'API diffusers: si la version
    installee ne connait pas un kwarg de guidance, on le retire et on relance plutot que
    de crasher la generation. (Nom conserve pour ne pas casser l'API interne du fork.)"""
    try:
        return pipe(**kw)
    except TypeError as e:
        if "negative_prompt" in kw:
            kw.pop("negative_prompt", None)
            _dbg(f"krea2 call: retry sans negative_prompt ({e})")
            return pipe(**kw)
        raise


# Alias explicite: le reste du fork appelle _qwen_call, mais le nom ment desormais.
_krea_call = _qwen_call


def _scheduler_accepts_sigmas(sched):
    """Le pipeline Z-Image appelle set_timesteps(..., sigmas=<schedule custom>). Un
    scheduler dont set_timesteps n'accepte pas `sigmas` plante a la generation."""
    import inspect
    try:
        return "sigmas" in inspect.signature(sched.set_timesteps).parameters
    except Exception:
        return False


def _build_scheduler(sampler, schedule, config):
    """Construit le scheduler choisi (sampler x schedule) depuis le config natif du modele.
    schedule (sgm_uniform/beta/karras/exponential) = remapping des sigmas (use_*_sigmas)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    kw = {}
    flag = _SCHEDULE_FLAG.get((schedule or "").lower())
    if flag:
        kw[flag] = True
    name = (sampler or "euler").lower()
    if name == "unipc":
        from diffusers import UniPCMultistepScheduler
        try:
            return UniPCMultistepScheduler.from_config(config, use_flow_sigmas=True, **kw)
        except Exception:
            return UniPCMultistepScheduler.from_config(config, **kw)
    if name == "lcm":
        # LCM flow-matching: accepte les sigmas custom du pipeline ET les flags de
        # schedule. Repli sur Euler si la version de diffusers ne l'expose pas.
        try:
            from diffusers import FlowMatchLCMScheduler
            return FlowMatchLCMScheduler.from_config(config, **kw)
        except Exception as e:
            _log(f"sampler 'lcm' unavailable ({e}); falling back to euler")
    return FlowMatchEulerDiscreteScheduler.from_config(config, **kw)


def _apply_sampler(pipe):
    """Pose le scheduler courant (SAMPLER x SCHEDULE) sur un pipe. Verifie la compatibilite
    (sigmas custom) et retombe sur Euler/sgm_uniform si KO -> jamais de crash a la generation."""
    if _BASE_SCHED_CONFIG is None:
        return
    from diffusers import FlowMatchEulerDiscreteScheduler
    try:
        sched = _build_scheduler(SAMPLER, SCHEDULE, _BASE_SCHED_CONFIG)
        if not _scheduler_accepts_sigmas(sched):
            raise ValueError(f"{type(sched).__name__} n'accepte pas les sigmas custom de Z-Image")
        pipe.scheduler = sched
        _dbg(f"sampler applied: {SAMPLER}/{SCHEDULE} -> {type(pipe.scheduler).__name__}")
    except Exception as e:
        _log(f"sampler '{SAMPLER}/{SCHEDULE}' incompatible ({e}); fallback Euler/sgm_uniform")
        try:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(_BASE_SCHED_CONFIG)
        except Exception:
            pass


def _reapply_sampler_all():
    """Re-applique le scheduler courant a tous les pipes en cache (base + derives)."""
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            _apply_sampler(p)


def set_sampler(name):
    """Change le sampler (euler/unipc) et le re-applique aux pipes en cache (pas de
    rechargement). Pas d'effet sur le pipe Omni (scheduler propre)."""
    global SAMPLER
    name = (name or "euler").strip().lower()
    if name not in SAMPLER_CHOICES:
        name = "euler"
    if name != SAMPLER:
        SAMPLER = name
        _reapply_sampler_all()
        _log(f"sampler -> {SAMPLER}")
    return f"Sampler: {SAMPLER} / {SCHEDULE}"


def set_schedule(name):
    """Change le schedule de sigmas (sgm_uniform/beta/karras/exponential, alias 'simple'
    = sgm_uniform) et le re-applique aux pipes en cache."""
    global SCHEDULE
    name = _norm_schedule(name)
    if name != SCHEDULE:
        SCHEDULE = name
        _reapply_sampler_all()
        _log(f"schedule -> {SCHEDULE}")
    return f"Sampler: {SAMPLER} / {SCHEDULE}"


def _progress(frac, desc=""):
    if _PROGRESS is not None:
        try:
            _PROGRESS(min(1.0, max(0.0, float(frac))), desc)
        except Exception:
            pass


# ---- Feedback de chargement des modeles (terminal + UI) ----
# from_pretrained est bloquant et silencieux (le 1er chargement telecharge depuis HF ->
# plusieurs minutes). On execute le chargement dans un thread et on rafraichit toutes les
# ~2s une ligne terminal + la barre Gradio (temps ecoule + VRAM allouee). Config bloc
# "load_progress"; enabled=false -> chargement direct (aucun thread, zero cout).
_LOAD_CFG = CONFIG.get("load_progress") if isinstance(CONFIG.get("load_progress"), dict) else {}
LOAD_PROGRESS_ENABLED = bool(_LOAD_CFG.get("enabled", True))
_LOAD_TARGET_GB = float(_LOAD_CFG.get("target_vram_gb", 14.0))
_LOAD_HEARTBEAT = float(_LOAD_CFG.get("heartbeat_s", 2.0))


def _fmt_load(label, elapsed, vram_gb):
    """Texte de progression de chargement (pur, testable). VRAM > 0 -> phase chargement
    en memoire; sinon phase download/lecture disque."""
    if vram_gb > 0.05:
        return f"{label}... {elapsed:.0f}s | {vram_gb:.1f} GB in VRAM"
    return f"{label}... {elapsed:.0f}s (downloading / reading, first run only)"


def _load_pct(elapsed, vram_gb, target_gb=None):
    """% honnete: base sur la VRAM allouee / cible une fois le chargement en memoire
    commence (plafonne 0.95); pendant le download (VRAM~0) petite barre temporelle."""
    target_gb = target_gb or _LOAD_TARGET_GB
    if vram_gb <= 0.05:
        return min(0.12, elapsed / 600.0)
    return min(0.95, vram_gb / max(1.0, float(target_gb)))


def _load_monitor(label, fn):
    """Execute fn() (chargement bloquant) dans un thread et rafraichit terminal + UI
    (temps + VRAM) toutes les ~2s. Renvoie le resultat de fn (releve son exception)."""
    if not LOAD_PROGRESS_ENABLED:
        return fn()
    box = {}

    def _work():
        try:
            box["v"] = fn()
        except BaseException as e:   # noqa: BLE001 - on re-leve dans le thread principal
            box["e"] = e

    th = threading.Thread(target=_work, daemon=True)
    t0 = time.time()
    th.start()
    while True:
        th.join(timeout=_LOAD_HEARTBEAT)
        el = time.time() - t0
        vram = (torch.cuda.memory_allocated() / 1024 ** 3) if DEVICE == "cuda" else 0.0
        line = _fmt_load(label, el, vram)
        if cz_core.LOG_LEVEL >= 1:
            sys.stderr.write("\r[crispz][load] " + line + "        ")
            sys.stderr.flush()
        _progress(_load_pct(el, vram), "Loading " + line)
        if not th.is_alive():
            break
    if cz_core.LOG_LEVEL >= 1:
        sys.stderr.write("\n")
        sys.stderr.flush()
    if "e" in box:
        raise box["e"]
    return box.get("v")


def request_stop():
    """Demande l'arret: stoppe la boucle de debruitage en cours (pipe._interrupt) et
    les boucles batch/tuiles (_STOP). Quasi-immediat (s'arrete au pas suivant)."""
    global _STOP
    _STOP = True
    n = 0
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            try:
                p._interrupt = True
                n += 1
            except Exception:
                pass
    _log(f"STOP requested (interrupt set on {n} pipeline(s))")
    return "Stopping..."


def set_zimage_model(repo_or_path):
    """Change le modele Krea 2. Un repo HF / dossier diffusers -> BASE_REPO.
    Un fichier single-file (.safetensors Civitai, .gguf) -> transformer override."""
    global BASE_REPO, ZIMAGE_TRANSFORMER
    if not repo_or_path:
        return
    if _is_single_file(repo_or_path):
        # Changement de transformer seul: PAS de free_vram -> _ensure_base echangera
        # uniquement le transformer (VAE + encodeur texte gardes en VRAM).
        if repo_or_path != ZIMAGE_TRANSFORMER:
            ZIMAGE_TRANSFORMER = repo_or_path
            _log("Krea 2 transformer (single-file) changed -> transformer swap on next run")
    elif repo_or_path != BASE_REPO:
        # Le repo de base change: VAE/encodeur/tokenizer changent aussi -> reload complet.
        BASE_REPO = repo_or_path
        free_vram()
        _log("Krea 2 base repo changed -> will reload")


def set_zimage_transformer(path):
    """Definit (ou enleve avec '' / None) le transformer single-file.

    NE libere PAS le pipeline: a repo de base identique, _ensure_base ne rechargera que
    le transformer (_swap_transformer) et gardera VAE + encodeur texte en VRAM."""
    global ZIMAGE_TRANSFORMER
    path = path or None
    if path != ZIMAGE_TRANSFORMER:
        ZIMAGE_TRANSFORMER = path
        _log(f"Krea 2 transformer -> {path or '(repo de base)'} "
             "-> transformer swap on next run (base components kept)")


def _safetensors_unsupported(path):
    """Renvoie une raison (str) si le .safetensors n'est PAS chargeable par diffusers,
    sinon None. Lit juste l'en-tete (rapide). Deux cas non supportes:
      - FP8 (F8_E4M3 / F8_E5M2) -> "FP8"
      - quantifie INT8/INT4 facon ComfyUI / SVDQuant-Nunchaku (tenseurs I8/U8 + facteurs
        'weight_scale') -> "INT8/INT4 quantized". diffusers ne dequantifie pas ce schema.
      - SVDQuant / Nunchaku (tenseurs nommes '*.qweight') -> "SVDQuant/Nunchaku INT4".
        Ce schema n'utilise PAS 'weight_scale', d'ou une detection dediee.
    Prendre le build BF16/FP16 non quantifie, ou un .gguf."""
    try:
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(min(n, 3_000_000)).decode("utf-8", "ignore"))
        has_fp8 = has_int = has_scale = has_qweight = False
        lora_keys = 0
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            dt = str(v.get("dtype", "")).upper()
            if dt.startswith("F8"):
                has_fp8 = True
            elif dt in ("I8", "I4", "U8", "U4", "UINT8", "INT8"):
                has_int = True
            if k.endswith("weight_scale") or k.endswith("scale_weight"):
                has_scale = True
            if k.endswith(".qweight"):
                has_qweight = True
            if (".lora_down." in k or ".lora_up." in k or ".lora_A." in k
                    or ".lora_B." in k or k.startswith(("lora_unet_", "lora_te"))):
                lora_keys += 1
        # Fichier LoRA range dans le dossier checkpoints (erreur classique): le charger
        # comme transformer envoie diffusers chercher une config par defaut (SD1.5) ->
        # 404 'stable-diffusion-v1-5 does not appear to have a file named config.json'.
        if lora_keys >= 4:
            return "LoRA file, not a checkpoint - move it to the LoRA folder and pick it in Models > LoRA"
        # '*.qweight' = poids pre-quantifies (SVDQuant/Nunchaku, GPTQ-like). Signal net:
        # un checkpoint BF16/FP16 normal n'a jamais de 'qweight'. Pas dequantifiable
        # par notre circuit (schema INT4 a groupes), contrairement aux FP8/INT8
        # 'scaled' ComfyUI qui, EUX, passent par la conversion (_converted_folder).
        if has_qweight:
            return "SVDQuant/Nunchaku INT4"
    except Exception:
        pass
    return None


def _safetensors_is_fp8(path):
    """Compat: ancien predicat FP8 seul. Prefere _safetensors_unsupported()."""
    return _safetensors_unsupported(path) == "FP8"


# Architecture attendue dans les .gguf. Un GGUF de diffusion declare son archi dans
# 'general.architecture': 'flux' (city96 FLUX.1 dev/schnell/krea), 'qwen_image',
# 'krea2' (Krea 2 = architecture PROPRE qui exige ComfyUI + son encodeur/VAE),
# 'llama'/'gemma3'... pour les LLM. On ne charge que 'qwen_image' ici.
GGUF_ARCH = str(CONFIG.get("gguf_arch") or "qwen_image").strip().lower()

_GGUF_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _gguf_skip(f, t):
    """Avance le flux au-dela d'une valeur GGUF sans la lire (strings et arrays inclus)."""
    import struct
    if t == 8:                                   # string
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        return
    if t == 9:                                   # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et in _GGUF_FIXED:
            f.seek(struct.calcsize(_GGUF_FIXED[et]) * n, 1)
        else:
            for _ in range(n):
                _gguf_skip(f, et)
        return
    f.seek(struct.calcsize(_GGUF_FIXED[t]), 1)


def _gguf_arch(path, max_kv=64):
    """'general.architecture' d'un .gguf -- lit seulement l'en-tete (quelques Ko), jamais
    les poids. Renvoie 'qwen_image' / 'flux' / 'krea2' / 'llama'... ou None si illisible
    (dans ce cas on ne filtre pas: mieux vaut tenter que d'ecarter un modele valide)."""
    import struct
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.seek(4 + 8, 1)                     # version (u32) + tensor_count (u64)
            nkv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(min(nkv, max_kv)):
                kl = struct.unpack("<Q", f.read(8))[0]
                if kl > 4096:                    # en-tete incoherent -> on abandonne
                    return None
                key = f.read(kl).decode("utf-8", "replace")
                t = struct.unpack("<I", f.read(4))[0]
                if key == "general.architecture" and t == 8:
                    n = struct.unpack("<Q", f.read(8))[0]
                    return f.read(n).decode("utf-8", "replace").strip().lower()
                _gguf_skip(f, t)
    except Exception as e:
        _dbg(f"gguf header read failed {path}: {e}")
    return None


def _checkpoint_dirs():
    """Dossiers a scanner pour les checkpoints single-file: principal + extra (si defini),
    sans doublon de chemin."""
    dirs = [CHECKPOINTS_DIR]
    if CHECKPOINTS_EXTRA_DIR and CHECKPOINTS_EXTRA_DIR not in dirs:
        dirs.append(CHECKPOINTS_EXTRA_DIR)
    return dirs


def list_checkpoints():
    """Dossiers diffusers locaux utilisables comme modele de base Krea 2.

    Les .safetensors single-file (Civitai bf16 et FP8/INT8 'scaled') sont
    proposes, comme les .gguf ComfyUI-GGUF (arch 'krea2'): tout passe par la
    CONVERSION en dossier diffusers au premier chargement (cache disque, cf.
    _converted_folder; le GGUF est dequantifie en bf16 - il n'economise que le
    telechargement, pas la VRAM). Restent ecartes: LoRA egarees et SVDQuant
    (schema INT4 non dequantifiable), avec la raison en console.

    Un dossier n'est retenu que s'il contient un model_index.json (layout diffusers).
    Les repos HF officiels sont ajoutes par l'UI (ZIMAGE_BASE_REPOS), pas ici."""
    out, seen = [], set()
    for d in _checkpoint_dirs():
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isdir(p):
                if f not in seen and os.path.isfile(os.path.join(p, "model_index.json")):
                    seen.add(f)
                    out.append(f)
            elif f.lower().endswith(".safetensors"):
                bad = _safetensors_unsupported(p)
                if bad:
                    _log(f"skipped {f}: {bad}")
                elif f not in seen:
                    seen.add(f)
                    out.append(f)
            elif f.lower().endswith(".gguf"):
                if f not in seen:
                    seen.add(f)
                    out.append(f)
    return sorted(out)


def resolve_checkpoint(name):
    """Chemin absolu d'un checkpoint single-file depuis son nom de fichier, cherche dans
    les dossiers checkpoints (principal puis extra). Renvoie name tel quel s'il est deja
    absolu; fallback sur le dossier principal si introuvable."""
    if not name or os.path.isabs(name):
        return name
    for d in _checkpoint_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(CHECKPOINTS_DIR, name)


def list_loras():
    """LoRA (.safetensors / .ckpt / .pt) du dossier loras, RECURSIF (sous-dossiers inclus).
    Renvoie des chemins RELATIFS a LORAS_DIR avec des '/' (ex. 'sous-dossier/ma_lora.safetensors')
    -> set_loras / resolve les resolvent via os.path.join(LORAS_DIR, name)."""
    if not os.path.isdir(LORAS_DIR):
        return []
    exts = (".safetensors", ".ckpt", ".pt")
    out = []
    for root, _dirs, files in os.walk(LORAS_DIR):
        for f in files:
            if f.lower().endswith(exts):
                rel = os.path.relpath(os.path.join(root, f), LORAS_DIR).replace(os.sep, "/")
                out.append(rel)
    return sorted(out)


def set_checkpoints_dir(path):
    global CHECKPOINTS_DIR
    if path:
        CHECKPOINTS_DIR = path


def set_checkpoints_extra_dir(path):
    """Definit (ou efface avec '' / None) le dossier checkpoints supplementaire."""
    global CHECKPOINTS_EXTRA_DIR
    CHECKPOINTS_EXTRA_DIR = (path or "").strip()


def set_loras_dir(path):
    global LORAS_DIR
    if path:
        LORAS_DIR = path


def _read_safetensors_metadata(path):
    """Lit le header JSON (__metadata__) d'un .safetensors SANS charger les poids."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
    return (json.loads(header.decode("utf-8")) or {}).get("__metadata__", {}) or {}


def lora_keywords(path):
    """Extrait les mots-cles / trigger words d'une LoRA depuis ses metadonnees:
    champs trigger explicites + top tags d'entrainement (ss_tag_frequency)."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        meta = _read_safetensors_metadata(path)
    except Exception as e:
        _dbg(f"lora metadata read failed: {e}")
        return ""
    words = []
    for k in ("ss_trigger_words", "modelspec.trigger_phrase", "trigger_words",
              "activation text", "ss_activation_text"):
        v = meta.get(k)
        if v:
            words.append(v if isinstance(v, str) else ", ".join(map(str, v)))
    tf = meta.get("ss_tag_frequency")
    if tf:
        try:
            d = json.loads(tf) if isinstance(tf, str) else tf
            counts = {}
            for ds in d.values():
                for tag, c in ds.items():
                    counts[tag] = counts.get(tag, 0) + int(c)
            words.extend(sorted(counts, key=counts.get, reverse=True)[:15])
        except Exception:
            pass
    seen, out = set(), []
    for w in words:
        for part in str(w).split(","):
            part = part.strip()
            if part and part.lower() not in seen:
                seen.add(part.lower())
                out.append(part)
    return ", ".join(out)


def set_loras(slots):
    """Definit les LoRA actives. slots = liste de (nom_ou_None, poids). Resout les
    noms en chemins, ignore les None.

    NE recharge PAS le modele: les LoRA sont echangees A CHAUD sur le transformer deja
    en VRAM (_apply_loras, appele par _ensure_base au run suivant)."""
    global LORAS
    new = []
    for name, weight in slots:
        if name and name not in ("None", "none", ""):
            p = name if os.path.isabs(name) else os.path.join(LORAS_DIR, name)
            new.append((p, float(weight)))
    if new != LORAS:
        LORAS = new
        _log("LoRAs -> " + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in new) or "(none)")
             + " -> applied on next run (hot-swap, no model reload)")


def set_omni_model(repo):
    """Definit le modele Omni/Edit (repo HF ou dossier). Invalide le pipe omni."""
    global OMNI_MODEL
    repo = (repo or "").strip()
    if repo != OMNI_MODEL:
        OMNI_MODEL = repo
        _DERIVED.pop("omni", None)
        _log(f"Omni model -> {repo or '(none)'}")


def check_omni_available():
    """Onglet Edit = Qwen-Image-Edit (modele d'edition par instruction). Verifie que le
    repo d'edition configure existe sur Hugging Face (API publique).

    Krea 2 n'a PAS de modele d'edition (CAPABILITIES['omni'] False,
    DEFAULT_OMNI_REPO None): reponse claire au lieu d'un AttributeError sur
    None.strip() si le bouton est atteint quand meme."""
    import urllib.request
    repo = ((OMNI_MODEL or DEFAULT_OMNI_REPO) or "").strip()
    if not CAPABILITIES.get("omni") or not repo:
        return ("**Omni/Edit is not available for Krea 2** - no "
                "instruction-edit model exists for this model family. Use "
                "crispz-qwen-edit (Qwen-Image-Edit) for reference/edit work.")
    try:
        req = urllib.request.Request("https://huggingface.co/api/models/" + repo,
                                     headers={"User-Agent": "crispz-krea2"})
        with urllib.request.urlopen(req, timeout=8) as r:
            if r.status == 200:
                return (f"**Edit model ready:** `{repo}`. The Edit tab edits an input image "
                        "from an instruction prompt. Change it in config.txt "
                        "`zimage_omni_model` (or the Models tab).")
    except Exception:
        pass
    return (f"Edit model `{repo}` not reachable (network/HF). It downloads on first use of "
            "the Edit tab. Override via config.txt `zimage_omni_model`.")


def set_offload_mode(mode):
    """Change le mode d'offload CPU. Invalide le pipe (hooks poses au chargement)."""
    global OFFLOAD_MODE
    mode = mode if mode in OFFLOAD_CHOICES else "none"
    if mode != OFFLOAD_MODE:
        OFFLOAD_MODE = mode
        free_vram()
        _log(f"offload -> {OFFLOAD_MODE}: pipeline invalidated -> will reload")


def free_vram():
    """Libere le pipeline de base + les pipelines derives et rend la VRAM
    (palier 3: unload sur inactivite ou endpoint /unload). Rechargement paresseux."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _APPLIED_LORAS
    _BASE_PIPE = None
    _DERIVED = {}
    _LOADED_KEY = None
    _APPLIED_LORAS = []      # plus de pipe -> plus d'adaptateur pose
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# Au-dela de ce cote (px) on active l'attention slicing (whole-image 2K+ -> evite le
# spill VRAM 32 Go). En-dessous (tuiles 1024, txt2img 1024/1536) -> slicing OFF =
# attention SDPA native = RAPIDE (comme ComfyUI). Reglable via config attention_slice_above.
_SLICE_ABOVE = int(CONFIG.get("attention_slice_above", 1664))

# Garde-fou: au-dela de ce cote (px), un refine "whole image" (refine_tile=0) est auto-
# tuile (tuile 1024). Defaut = le seuil de slicing: au-dela, un whole-image serait slice
# (lent: ~120s en 2K) ET risque le spill VRAM (4K -> crash). Tuiler est plus rapide ET sur.
_AUTO_TILE_ABOVE = int(CONFIG.get("auto_refine_tile_above", _SLICE_ABOVE))

# Taille de la tuile employee par cet auto-tuilage. "auto" (defaut) = calculee par
# _pick_refine_tile ; un entier fige la taille (ancien comportement : 1024).
# Mesure (RTX 5090, sortie 4096x4096, denoise 0.40, overlap 64) : le cout par pixel est
# PLAT de 768 a 1024 (1.78 / 1.83 / 1.79 us/px) et ne grimpe qu'au-dela (2.41 a 1536,
# 3.00 a 2048). Le temps suit donc la SURFACE TUILEE (n x tuile^2), pas la taille de la
# tuile. Or a 1024 la grille deborde : pas de 960 sur 4096 -> la derniere tuile est
# rabattue et recouvre la precedente sur 832px au lieu de 64, soit 1.56x la surface de
# l'image. A 896 le pas tombe juste (1.20x) -> 36.7s au lieu de 46.9s sur la meme image,
# a nombre de tuiles (25) et de coutures (8) IDENTIQUE.
# Bornes [768, 1024] : en dessous on multiplie tuiles et coutures et chaque tuile voit
# moins de contexte (le rendu derive - un arriere-plan flou se reconstruit differemment,
# verifie visuellement) ; au-dessus l'attention devient superlineaire.
_AUTO_TILE_MIN = int(CONFIG.get("auto_refine_tile_min", 768))
_AUTO_TILE_MAX = int(CONFIG.get("auto_refine_tile_max", 1024))
_AUTO_TILE_SIZE = str(CONFIG.get("auto_refine_tile", "auto")).strip().lower()


def _pick_refine_tile(w, h, overlap):
    """Tuile qui minimise la surface tuilee pour couvrir w x h (= le cout reel de la passe).

    A surface egale on garde la PLUS GRANDE tuile : moins de coutures et plus de contexte
    par tuile. Un entier dans auto_refine_tile court-circuite le calcul (taille figee)."""
    if _AUTO_TILE_SIZE not in ("auto", "", "0"):
        try:
            return round_to_multiple(int(_AUTO_TILE_SIZE))
        except ValueError:
            _log(f"config auto_refine_tile='{_AUTO_TILE_SIZE}' invalide (attendu 'auto' ou "
                 "un entier) -> calcul automatique")
    lo = max(256, _AUTO_TILE_MIN)
    hi = max(lo, _AUTO_TILE_MAX)
    ov = max(0, int(overlap))
    cands = []
    for t in range(lo, hi + 1, 32):
        step = max(16, t - ov)
        n = len(range(0, max(1, int(w)), step)) * len(range(0, max(1, int(h)), step))
        cands.append((n * t * t, -t, t))       # surface mini, puis plus grande tuile
    return min(cands)[2]

# Plafond de denoise pour le refine TUILE. En tuiles, chaque tuile est rediffusee avec le
# prompt global -> a fort denoise la diffusion reconstruit le sujet (ex: la tasse) DANS
# chaque tuile = duplications. On plafonne donc le denoise par tuile (le contenu existant
# guide alors la diffusion, facon Ultimate SD Upscale). Le refine "whole image" garde le
# denoise demande (pas de duplication possible: une seule passe sur toute la compo).
# Reglable via config refine_tile_denoise_cap (0 = pas de plafond).
_TILE_DENOISE_CAP = float(CONFIG.get("refine_tile_denoise_cap", 0.40))

# Prompt utilise pour le refine TUILE. Le prompt global decrit TOUTE la composition (pas
# la tuile) -> le passer a chaque tuile pousse la diffusion a recreer le sujet (la tasse)
# dans des tuiles qui ne sont que du fond. Par defaut on passe donc un prompt VIDE: chaque
# tuile se contente d'affiner le detail local. Valeurs config refine_tile_prompt:
#   "" (defaut) = prompt vide par tuile
#   "global"/"scene" = reutilise le prompt de la scene (ancien comportement)
#   tout autre texte = prompt generique applique a chaque tuile (ex: "high detail, sharp")
_TILE_PROMPT = str(CONFIG.get("refine_tile_prompt", ""))


def _tile_prompt(scene_prompt):
    """Prompt a utiliser par tuile selon la config (vide par defaut, anti-duplication)."""
    if _TILE_PROMPT.strip().lower() in ("global", "scene"):
        return scene_prompt or ""
    return _TILE_PROMPT


def _set_slicing(pipe, longest_side):
    """Active/desactive l'attention slicing selon le plus grand cote a traiter. Appele
    avant CHAQUE passe de diffusion (txt2img/refine/tuile/inpaint/outpaint/omni)."""
    try:
        if int(longest_side) > _SLICE_ABOVE:
            pipe.enable_attention_slicing()
        else:
            pipe.disable_attention_slicing()
    except Exception:
        pass


def _vram_str():
    """Pic VRAM PyTorch reserve / total (pour reperer la saturation -> spill RAM partagee
    Windows = lenteur extreme, et TDR/'CUDA unknown error'). Ne voit PAS la VRAM des
    autres process (ComfyUI, etc.) -> utiliser nvidia-smi pour le total reel."""
    if DEVICE != "cuda":
        return ""
    try:
        resv = torch.cuda.memory_reserved() / 1024**3
        tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f" | VRAM {resv:.1f}/{tot:.0f} Go"
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# Krea 2 (diffusers, BF16 + quantif torchao) : un unique pipeline "base" txt2img.
# img2img / inpaint derives via from_pipe (poids partages, pas de VRAM en double).
# ----------------------------------------------------------------------------
def _is_gguf_path(p):
    return bool(p) and str(p).lower().endswith(".gguf")


def _effective_offload(tpath=None):
    """Offload REELLEMENT applique. Un transformer GGUF quantifie ne se deplace pas sur le
    GPU via .to(cuda) ni en sequential -> seul enable_model_cpu_offload le pose sur le GPU
    pendant le forward. On force donc 'model' pour un base GGUF, quel que soit le reglage."""
    off = OFFLOAD_MODE
    t = ZIMAGE_TRANSFORMER if tpath is None else tpath
    if DEVICE == "cuda" and _is_gguf_path(t) and off != "model":
        off = "model"
    return off


# ----------------------------------------------------------------------------
# Conversion single-file (Comfy/Civitai) -> dossier diffusers, en cache disque.
#
# Krea2Transformer2DModel n'a pas de from_single_file dans diffusers: la table
# de correspondance de cles n'existe pas upstream. On l'implemente ICI: le
# checkpoint Comfy est un miroir 1:1 du modele diffusers (430 cles des deux
# cotes, verifie sur le re-export officiel) - pur renommage, plus UN reshape
# (mod.lin (36864,) -> scale_shift_table (6, 6144)). Les variantes FP8/INT8
# 'scaled' ComfyUI (y compris ConvRot) sont dequantifiees en bf16 pendant la
# conversion (meme circuit que crispz-studio, valide sur Z-Image).
#
# Le resultat est ecrit UNE FOIS en dossier diffusers (cache/krea2_convert/...)
# puis recharge par le chemin from_pretrained EXISTANT (quantification torchao
# comprise): premiere selection = conversion (minutes, ~26 Go ecrits), les
# suivantes = chargement normal.
# ----------------------------------------------------------------------------
import re as _re
import hashlib as _hashlib

_KREA2_LEAF_MAP = {
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.gate.weight": "attn.to_gate.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.down.weight": "ff.down.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
}
_KREA2_TOP_MAP = {
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias": "time_mod_proj.bias",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
    "last.norm.scale": "final_layer.norm.weight",
    "last.modulation.lin": "final_layer.scale_shift_table",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
}


def _krea2_rename(k):
    """Nom Comfy -> (nom diffusers, reshape_rows|None). None si cle inconnue."""
    if k in _KREA2_TOP_MAP:
        return _KREA2_TOP_MAP[k], None
    m = _re.match(r"^blocks\.(\d+)\.(.+)$", k)
    if m:
        if m.group(2) == "mod.lin":
            # (6*dim,) -> (6, dim): la table de modulation est stockee a plat
            return f"transformer_blocks.{m.group(1)}.scale_shift_table", 6
        leaf = _KREA2_LEAF_MAP.get(m.group(2))
        if leaf:
            return f"transformer_blocks.{m.group(1)}.{leaf}", None
        return None
    m = _re.match(r"^txtfusion\.(layerwise|refiner)_blocks\.(\d+)\.(.+)$", k)
    if m:
        leaf = _KREA2_LEAF_MAP.get(m.group(3))
        if leaf:
            return f"text_fusion.{m.group(1)}_blocks.{m.group(2)}.{leaf}", None
    return None


def _hadamard_ortho(n):
    """Matrice 'regular hadamard' du ConvRot comfy-quants -- ATTENTION, ce n'est
    PAS la construction de Sylvester: base H4 precise, etendue par produits de
    Kronecker jusqu'a n (puissance de 4), normalisee 1/sqrt(n). Orthonormee ET
    symetrique -> la reconstruction re-multiplie par la meme matrice. (Porte de
    crispz-studio, verifie contre comfy_quants/formats/convrot.py.)"""
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]])
    H = h4
    while H.shape[0] < n:
        H = torch.kron(H, h4)
    if H.shape[0] != n:
        raise ValueError(f"convrot groupsize {n} is not a power of 4")
    return H / (float(n) ** 0.5)


def _read_comfy_state_dict(path):
    """Lit un single-file Krea 2 (Comfy/Civitai) en RAM, dequantifie en DTYPE:
      - bundle AIO: seules les cles 'model.diffusion_model.*' sont gardees;
      - X.weight (F8/I8) * X.weight_scale / X.scale_weight -> bf16;
      - blob X.comfy_quant declarant 'convrot' -> rotation Hadamard DEFAITE
        apres descale (sinon les poids sont un bruit total);
      - cles de quantification consommees/jetees.
    Lecture SEQUENTIELLE en ordre physique (data_offsets): un HDD s'effondre en
    acces aleatoire (mesure crispz-studio: 349 s -> debit disque)."""
    from safetensors import safe_open
    t0 = time.time()
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8", "ignore"))
    entries = [(k, v) for k, v in hdr.items()
               if k != "__metadata__" and isinstance(v, dict)]
    prefix = ""
    if any(k.startswith("model.diffusion_model.") for k, _ in entries):
        prefix = "model.diffusion_model."
        entries = [(k, v) for k, v in entries if k.startswith(prefix)]
    # Garde d'architecture: sans marqueur Krea 2, refus clair (un checkpoint
    # FLUX/Z-Image egare chargerait du bruit).
    if not any(k[len(prefix):].startswith(("txtfusion.", "blocks.0.attn.wq"))
               for k, _ in entries):
        raise RuntimeError(
            f"{os.path.basename(path)}: not a Krea 2 checkpoint (no txtfusion/"
            f"blocks markers) - this build only converts Krea 2 single files.")
    entries.sort(key=lambda kv: kv[1].get("data_offsets", [0])[0])
    raw, qcfg = {}, {}
    # comfy-quants declare le schema soit en blobs PAR TENSEUR (X.comfy_quant),
    # soit CENTRALEMENT dans __metadata__._quantization_metadata (StableYogi
    # INT8: {"layers": {"blocks.0.attn.gate": {"format": "int8_tensorwise",
    # "convrot": true, "convrot_groupsize": 256}}}). Ignorer cette variante
    # rend les poids en BRUIT TOTAL (rotation jamais defaite) - observe sur
    # realismByStableYogi_v25INT8Turbo. Les blobs par tenseur gagnent (ecrits
    # apres, ils ecrasent l'entree metadata du meme layer).
    try:
        qm = json.loads((hdr.get("__metadata__") or {}).get(
            "_quantization_metadata") or "{}")
        for lk, lv in (qm.get("layers") or {}).items():
            if isinstance(lv, dict):
                qcfg[lk[len(prefix):] if prefix and lk.startswith(prefix) else lk] = lv
        if qcfg:
            _dbg(f"quantization metadata: {len(qcfg)} layer(s) declared in header")
    except Exception as e:
        _dbg(f"_quantization_metadata unreadable: {e}")
    with safe_open(path, framework="pt", device="cpu") as f:
        for k, _ in entries:
            kk = k[len(prefix):]
            if kk.endswith(".comfy_quant"):
                try:
                    qcfg[kk[:-len(".comfy_quant")]] = json.loads(
                        bytes(f.get_tensor(k).tolist()).decode("utf-8"))
                except Exception as e:
                    _dbg(f"comfy_quant blob unreadable {k}: {e}")
                continue
            raw[kk] = f.get_tensor(k)
    # Le calcul de dequantification (cast fp32 + scales + un-rotation) est
    # limite par la bande passante memoire en CPU (~9 min mesurees sur un INT8
    # 12.9B): on le fait sur le GPU quand il est disponible, tenseur par
    # tenseur (~400 Mo max en VRAM), retour bf16 en RAM. convert_device: auto
    # (defaut, cuda si present) | cpu.
    dev = "cpu"
    try:
        if (torch.cuda.is_available()
                and str(CONFIG.get("convert_device", "auto")).lower() != "cpu"):
            dev = "cuda"
    except Exception:
        pass
    _had, sd = {}, {}
    n_dq = n_rot = 0
    for k in list(raw.keys()):
        if (k.endswith((".weight_scale", ".scale_weight", ".scale_input",
                        ".input_scale")) or k.endswith("scaled_fp8")):
            continue
        t = raw.pop(k)
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                       torch.int8, torch.uint8):
            s = None
            for cand in (k + "_scale",
                         (k[:-len(".weight")] + ".scale_weight")
                         if k.endswith(".weight") else None):
                if cand and cand in raw:
                    s = raw[cand]
                    break
            t = t.to(dev).to(torch.float32)
            if s is not None:
                t = t * s.to(dev).to(torch.float32)
            cfg = qcfg.get(k[:-len(".weight")]) if k.endswith(".weight") else None
            if cfg and cfg.get("convrot"):
                g = int(cfg.get("convrot_groupsize", 256) or 256)
                if t.dim() == 2 and g > 1 and t.shape[1] % g == 0:
                    if g not in _had:
                        _had[g] = _hadamard_ortho(g).to(dev)
                    t = (t.view(t.shape[0], -1, g) @ _had[g]).reshape(t.shape[0], -1)
                    n_rot += 1
            t = t.to(DTYPE).cpu()
            n_dq += 1
        elif t.is_floating_point() and t.dtype != DTYPE:
            t = t.to(DTYPE)
        sd[k] = t
    raw.clear()
    if dev != "cpu":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if n_dq:
        _log(f"dequantized {n_dq} tensors"
             + (f", {n_rot} un-rotated (ConvRot)" if n_rot else "")
             + (f" on {dev}" if n_dq else "")
             + f" in {time.time() - t0:.1f}s")
    return sd


def _read_gguf_state_dict(path):
    """Lit un .gguf Krea 2 (export ComfyUI-GGUF, arch 'krea2') et dequantifie
    TOUT en DTYPE via la lib gguf (Q8_0/Q6_K/... -> float32 -> bf16). Les noms
    de tenseurs sont les MEMES que le layout Comfy safetensors (verifie: 430
    cles identiques) -> la meme table de renommage s'applique ensuite, et
    gguf.quants.dequantize renvoie deja l'orientation torch (verifie sur
    blocks.0.attn.wk: (1536, 6144)).

    NB: contrairement a crispz-studio (Z-Image), le GGUF ne peut PAS rester
    quantifie en VRAM ici (pas de from_single_file pour cette architecture):
    il est converti UNE FOIS en bf16 (cache disque), la quantification runtime
    reste torchao (float8_weight_only). Le Q8_0 ne fait donc gagner que le
    telechargement, pas la VRAM."""
    import gguf
    from gguf import GGUFReader
    t0 = time.time()
    r = GGUFReader(path)
    arch = ""
    for fld in r.fields.values():
        if fld.name == "general.architecture":
            try:
                arch = bytes(fld.parts[fld.data[0]]).decode("utf-8", "ignore")
            except Exception:
                pass
    names = [t.name for t in r.tensors]
    if arch != "krea2" and not any(n.startswith("txtfusion.") for n in names):
        raise RuntimeError(
            f"{os.path.basename(path)}: GGUF architecture '{arch or '?'}' is "
            "not Krea 2 - this build only converts Krea 2 files.")
    sd = {}
    n_dq = 0
    for t in r.tensors:
        arr = gguf.quants.dequantize(t.data, t.tensor_type)
        if t.tensor_type.name not in ("F32", "F16", "BF16"):
            n_dq += 1
        sd[t.name] = torch.from_numpy(np.ascontiguousarray(arr)).to(DTYPE)
    _log(f"GGUF dequantized: {n_dq} quantized tensor(s) of {len(sd)} "
         f"in {time.time() - t0:.1f}s")
    return sd


def _expected_transformer_keys():
    """(config_dict, {cle: shape}) du transformer du repo de base, SANS charger
    les poids (init_empty_weights). Le config.json vient du cache HF local du
    repo de base -> il faut avoir charge le modele officiel au moins une fois."""
    from accelerate import init_empty_weights
    from diffusers import Krea2Transformer2DModel
    from huggingface_hub import hf_hub_download
    try:
        cfg_path = hf_hub_download(BASE_REPO, "transformer/config.json")
    except Exception as e:
        raise RuntimeError(
            f"cannot fetch transformer/config.json from {BASE_REPO} ({e}). "
            "Load the official base model once (network) before converting "
            "single-file checkpoints.") from e
    with open(cfg_path, "r", encoding="utf-8") as f:
        conf = json.load(f)
    with init_empty_weights():
        m = Krea2Transformer2DModel.from_config(conf)
    return conf, {k: tuple(v.shape) for k, v in m.state_dict().items()}


def _convert_cache_dir():
    """Dossier du cache de conversion. Config convert_cache: 'auto' (defaut) =
    <app>/cache/krea2_convert, chemin = dossier custom, 'off' = pas de cache
    (conversion refusee: 26 Go par entree, un tmp jetable n'a pas de sens)."""
    mode = str(CONFIG.get("convert_cache", "auto") or "auto")
    if mode.lower() == "off":
        return None
    if mode.lower() == "auto":
        return os.path.join(cz_core.HERE, "cache", "krea2_convert")
    return mode


def _prune_convert_cache(keep):
    """Evince les conversions les moins recemment utilisees au-dela de
    convert_cache_max_gb (0 = illimite). `keep` = dossier a ne jamais evincer."""
    root = _convert_cache_dir()
    cap = float(CONFIG.get("convert_cache_max_gb", 80) or 0)
    if not root or not os.path.isdir(root) or cap <= 0:
        return
    entries = []
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if not os.path.isdir(p) or os.path.abspath(p) == os.path.abspath(keep):
            continue
        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(p) for f in fs)
        entries.append((os.path.getmtime(p), p, size))
    total = sum(s for _, _, s in entries)
    keep_size = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(keep) for f in fs) if os.path.isdir(keep) else 0
    total += keep_size
    for mt, p, s in sorted(entries):
        if total <= cap * 1e9:
            break
        import shutil
        shutil.rmtree(p, ignore_errors=True)
        total -= s
        _log(f"convert cache: evicted {os.path.basename(p)} ({s / 1e9:.1f} GB)")


def _converted_folder(path):
    """Dossier diffusers du single-file `path`: conversion a la premiere
    demande (cache par (chemin, taille, mtime)), reutilisation ensuite."""
    root = _convert_cache_dir()
    if not root:
        raise RuntimeError(
            "convert_cache is 'off': converting a Krea 2 single file needs the "
            "on-disk cache (~26 GB per checkpoint). Set convert_cache to 'auto' "
            "or a folder path in config.txt.")
    st = os.stat(path)
    sig = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    key = _hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    stem = _re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(path))[0])[:40]
    dst = os.path.join(root, f"{stem}_{key}")
    stamp = os.path.join(dst, "source.json")
    weights = os.path.join(dst, "transformer", "diffusion_pytorch_model.safetensors")
    if os.path.isfile(stamp) and os.path.isfile(weights):
        try:
            if json.load(open(stamp, "r", encoding="utf-8")).get("sig") == sig:
                os.utime(dst, None)          # LRU
                return dst
        except Exception:
            pass
    _log(f"converting {os.path.basename(path)} to diffusers layout (first time "
         "only: reads the full file, writes ~26 GB bf16 to the convert cache) ...")
    t0 = time.time()
    conf, expected = _expected_transformer_keys()
    sd = (_read_gguf_state_dict(path) if path.lower().endswith(".gguf")
          else _read_comfy_state_dict(path))
    out, unknown = {}, []
    for k, t in sd.items():
        r = _krea2_rename(k)
        if r is None:
            unknown.append(k)
            continue
        name, rows = r
        if rows is not None and t.dim() == 1 and t.numel() % rows == 0:
            t = t.view(rows, -1)
        out[name] = t
    missing = [k for k in expected if k not in out]
    bad = [f"{k} {tuple(out[k].shape)}!={expected[k]}"
           for k in out if k in expected and tuple(out[k].shape) != expected[k]]
    if unknown or missing or bad:
        raise RuntimeError(
            f"{os.path.basename(path)}: conversion mismatch - "
            f"{len(unknown)} unknown key(s) {unknown[:3]}, "
            f"{len(missing)} missing {missing[:3]}, "
            f"{len(bad)} shape mismatch(es) {bad[:2]}. The file is probably not "
            f"a standard Krea 2 export; please report it.")
    os.makedirs(os.path.join(dst, "transformer"), exist_ok=True)
    with open(os.path.join(dst, "transformer", "config.json"), "w",
              encoding="utf-8") as f:
        json.dump(conf, f, indent=2)
    from safetensors.torch import save_file
    tmp = weights + ".tmp"
    save_file(out, tmp)
    os.replace(tmp, weights)
    with open(stamp, "w", encoding="utf-8") as f:
        json.dump({"sig": sig, "source": os.path.basename(path)}, f)
    _log(f"converted in {time.time() - t0:.1f}s -> {dst}")
    _prune_convert_cache(dst)
    return dst


def _load_transformer():
    """Charge UNIQUEMENT le transformer courant (sans le reste du pipeline), depuis un repo
    diffusers, quantifie a la volee selon QUANT_MODE.

    Un single-file (.safetensors Civitai bf16/FP8/INT8 'scaled', ou .gguf
    ComfyUI-GGUF) passe par la CONVERSION en dossier diffusers
    (_converted_folder, cache disque) puis par ce meme chemin from_pretrained.

    Utilise au chargement complet ET pour l'echange a chaud (_swap_transformer)."""
    from diffusers import Krea2Transformer2DModel
    repo = ZIMAGE_TRANSFORMER or BASE_REPO
    single = bool(ZIMAGE_TRANSFORMER and _is_single_file(ZIMAGE_TRANSFORMER))
    if single:
        bad = None if ZIMAGE_TRANSFORMER.lower().endswith(".gguf") \
            else _safetensors_unsupported(ZIMAGE_TRANSFORMER)
        if bad:  # LoRA egaree / SVDQuant; un .gguf est valide par son arch
            raise UnsupportedFeature(
                f"{os.path.basename(ZIMAGE_TRANSFORMER)}: {bad}.")
    qc = _quant_config()
    kw = {"subfolder": "transformer", "torch_dtype": DTYPE}
    if qc is not None:
        kw["quantization_config"] = qc
        _log(f"loading Krea 2 transformer ({QUANT_MODE}): {repo} ... "
             "(reads ~26 GB bf16 into RAM, writes ~13 GB quantized to VRAM)")
    else:
        _log(f"loading Krea 2 transformer (bf16, not quantized): {repo} ... "
             "[WARN] ~26 GB: overflows a 32 GB card, expect ~59 s/step")
    label = (f"transformer {os.path.basename(str(repo).rstrip('/'))}"
             + (f" ({QUANT_MODE})" if qc is not None else " (bf16)"))

    def _repo_loader():
        # La conversion (couteuse) n'est declenchee QU'ICI, donc uniquement sur
        # un vrai MISS du cache de quantification.
        return _converted_folder(ZIMAGE_TRANSFORMER) if single else repo
    if qc is not None and _quant_cache_dir():
        try:
            return _load_transformer_quant_cached(
                ZIMAGE_TRANSFORMER if single else repo, _repo_loader, kw, label)
        except Exception as e:
            _log(f"quant cache path failed ({e}); falling back to direct load")
    return _load_monitor(
        label,
        lambda: Krea2Transformer2DModel.from_pretrained(_repo_loader(), **kw))


# ----------------------------------------------------------------------------
# Cache de QUANTIFICATION: la forme torchao (~13 Go) est serialisee UNE FOIS
# puis rechargee directement -> plus de lecture des ~26 Go bf16 ni de quantize
# a chaque chargement (demarrage d'app compris). Pickle .bin impose:
# les tenseurs torchao ne se serialisent pas en safetensors.
# ----------------------------------------------------------------------------
def _quant_cache_dir():
    """Config quant_cache: 'auto' (defaut) = <app>/cache/krea2_quant,
    chemin = dossier custom, 'off' = desactive (chargement direct)."""
    # DEFAUT OFF depuis le 23/08: sur torch 2.8 + ce torchao, les tenseurs
    # DEPICKLES perdent leurs kernels rapides et les rendus sont ~3.3x plus
    # lents (mesure A/B: 316 s direct vs 1060 s via pickle, memes reglages),
    # alors que le chargement, lui, tombe bien a ~4 s. La correction etait
    # invisible aux tests (corr pixel 1.0000: les maths restent justes, seule
    # la vitesse casse). Opt-in conserve pour re-tester sur un futur couple
    # torch/torchao ou la serialisation garde les kernels.
    mode = str(CONFIG.get("quant_cache", "off") or "off")
    if mode.lower() == "off":
        return None
    if mode.lower() == "auto":
        return os.path.join(cz_core.HERE, "cache", "krea2_quant")
    return mode


def _prune_quant_cache(keep):
    """Evince les formes quantifiees les moins recemment utilisees au-dela de
    quant_cache_max_gb (0 = illimite)."""
    root = _quant_cache_dir()
    cap = float(CONFIG.get("quant_cache_max_gb", 60) or 0)
    if not root or not os.path.isdir(root) or cap <= 0:
        return
    entries = []
    for d in os.listdir(root):
        pth = os.path.join(root, d)
        if not os.path.isdir(pth) or os.path.abspath(pth) == os.path.abspath(keep):
            continue
        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(pth) for f in fs)
        entries.append((os.path.getmtime(pth), pth, size))
    total = sum(sz for _, _, sz in entries)
    if os.path.isdir(keep):
        total += sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(keep) for f in fs)
    for _mt, pth, sz in sorted(entries):
        if total <= cap * 1e9:
            break
        import shutil
        shutil.rmtree(pth, ignore_errors=True)
        total -= sz
        _log(f"quant cache: evicted {os.path.basename(pth)} ({sz / 1e9:.1f} GB)")


def _repo_sig(src):
    """Identite stable de la SOURCE du transformer, dans l'ordre:
    FICHIER single-file original -> chemin + taille + mtime DU FICHIER (la cle
    survit ainsi a la suppression du cache de conversion - lecon apprise: la
    cle historique pointait le dossier converti, et purger krea2_convert
    invalidait toutes les formes quantifiees); dossier diffusers -> ses poids;
    repo HF -> son identifiant."""
    try:
        if os.path.isfile(str(src)):
            st = os.stat(src)
            return f"{os.path.abspath(src)}|{st.st_size}|{int(st.st_mtime)}"
        w = os.path.join(str(src), "transformer",
                         "diffusion_pytorch_model.safetensors")
        if os.path.isfile(w):
            st = os.stat(w)
            return f"{os.path.abspath(src)}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        pass
    return str(src)


def _quant_entry_complete(d):
    """Une entree du quant-cache est utilisable: config + au moins un shard."""
    if not os.path.isfile(os.path.join(d, "config.json")):
        return False
    try:
        return any(f.startswith("diffusion_pytorch_model") and f.endswith(".bin")
                   for f in os.listdir(d))
    except OSError:
        return False


def _load_transformer_quant_cached(sig_source, repo_loader, kw, label):
    """Charge le transformer via le cache de quantification.

    sig_source = le FICHIER original (single-file) ou l'id de repo: la cle ne
    depend PAS du cache de conversion. repo_loader = callable -> dossier/repo a
    charger en cas de MISS (c'est LUI qui declenche l'eventuelle conversion:
    sur un HIT, aucune conversion n'a lieu, meme si krea2_convert a ete purge).

    HIT: from_pretrained direct de la forme torchao picklee (~13 Go, ~4 s).
    MISS: repo_loader() -> chargement normal (bf16 + quantize) PUIS
    save_pretrained pour toutes les fois suivantes (echec non-fatal).
    Les entrees a CLE HISTORIQUE (dossier converti) sont migrees par stem:
    renommees et re-stampees, pas resérialisees."""
    from diffusers import Krea2Transformer2DModel
    root = _quant_cache_dir()
    sig = f"{_repo_sig(sig_source)}|{QUANT_MODE}"
    key = _hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    base = _re.sub(r"[^A-Za-z0-9_-]+", "_",
                   os.path.splitext(os.path.basename(str(sig_source).rstrip("/\\")))[0])
    stem = base[:40]
    dst = os.path.join(root, f"{stem}_{key}")
    stamp = os.path.join(dst, "source.json")

    def _hit():
        os.utime(dst, None)                           # LRU
        _log(f"loading Krea 2 transformer (pre-quantized cache, "
             f"{QUANT_MODE}): reads ~13 GB, no dequant/quantize step ...")
        return _load_monitor(
            f"transformer {stem} ({QUANT_MODE}, pre-quantized)",
            lambda: Krea2Transformer2DModel.from_pretrained(
                dst, torch_dtype=DTYPE, use_safetensors=False))

    if os.path.isfile(stamp):
        try:
            with open(stamp, "r", encoding="utf-8") as f:
                if json.load(f).get("sig") == sig:
                    return _hit()
        except Exception:
            pass
    # Migration des entrees historiques (cle = dossier converti, detruit par un
    # nettoyage legitime du cache de conversion): meme stem => meme modele.
    # On adopte la plus recente complete, on jette les autres du meme stem.
    if os.path.isdir(root) and not os.path.isdir(dst):
        cands = [os.path.join(root, d) for d in os.listdir(root)
                 if d.startswith(base[:24]) and os.path.join(root, d) != dst]
        def _adoptable(c):
            # n'adopter qu'une entree HISTORIQUE (cle batie sur le dossier
            # converti) du MEME schema de quantification. Deux refus:
            #  - autre schema: un pickle float8 servi apres un passage a int8
            #    serait faux;
            #  - cle nouvelle generation du MEME fichier (sig commencant par
            #    son chemin): si on arrive ici c'est que la source a CHANGE,
            #    l'entree est perimee et doit etre reconvertie, pas adoptee.
            try:
                with open(os.path.join(c, "source.json"), "r",
                          encoding="utf-8") as f:
                    old_sig = json.load(f).get("sig", "")
            except Exception:
                return False
            if not old_sig.endswith("|" + QUANT_MODE):
                return False
            return not old_sig.startswith(
                os.path.abspath(str(sig_source)) + "|")
        cands = [c for c in cands
                 if os.path.isdir(c) and _quant_entry_complete(c)
                 and _adoptable(c)]
        if cands:
            best = max(cands, key=os.path.getmtime)
            try:
                os.rename(best, dst)
                with open(stamp, "w", encoding="utf-8") as f:
                    json.dump({"sig": sig, "quant": QUANT_MODE,
                               "migrated_from": os.path.basename(best)}, f)
                _log(f"quant cache: migrated legacy entry "
                     f"{os.path.basename(best)} -> keyed on the original file")
                for extra in cands:
                    if extra != best and os.path.isdir(extra):
                        import shutil
                        shutil.rmtree(extra, ignore_errors=True)
                        _log(f"quant cache: dropped duplicate legacy entry "
                             f"{os.path.basename(extra)}")
                return _hit()
            except OSError as e:
                _dbg(f"quant cache migration failed: {e}")
    src_repo = repo_loader()      # conversion eventuelle ICI, sur vrai MISS
    model = _load_monitor(
        label, lambda: Krea2Transformer2DModel.from_pretrained(src_repo, **kw))
    try:
        t0 = time.time()
        os.makedirs(dst, exist_ok=True)
        model.save_pretrained(dst, safe_serialization=False)
        with open(stamp, "w", encoding="utf-8") as f:
            json.dump({"sig": sig, "quant": QUANT_MODE}, f)
        _log(f"quantized form cached in {time.time() - t0:.0f}s -> next loads "
             "of this checkpoint skip the 26 GB read and the quantize step")
        _prune_quant_cache(dst)
    except Exception as e:
        _log(f"quant cache save failed (non-fatal, direct loads continue): {e}")
        import shutil
        shutil.rmtree(dst, ignore_errors=True)
    return model


def _lora_names(loras):
    return [f"cz_lora_{i}" for i in range(len(loras))]


def _clear_loras(pipe):
    """Retire TOUT adaptateur LoRA du pipe pour repartir d'un etat vierge.

    unload_lora_weights() seul laisse, selon les versions diffusers/peft, un peft_config
    residuel sur le transformer -> le load suivant avertit ('Already found a peft_config')
    et, comme on reutilise les memes noms d'adaptateurs (cz_lora_i), l'ancien adaptateur
    peut rester en place (mauvaise LoRA appliquee). On supprime donc explicitement les
    adaptateurs restants par nom apres l'unload."""
    try:
        pipe.unload_lora_weights()
    except Exception as e:
        _dbg(f"unload_lora_weights: {e}")
    try:
        listed = pipe.get_list_adapters() or {}
        names = sorted({n for lst in listed.values() for n in (lst or [])})
        if names:
            pipe.delete_adapters(names)
            _dbg(f"cleared leftover LoRA adapters: {names}")
    except Exception as e:
        _dbg(f"delete_adapters: {e}")


def _apply_loras(pipe, force=False):
    """Synchronise les adaptateurs LoRA du pipe avec LORAS, SANS recharger le modele.

    Le transformer reste en VRAM; seuls les adaptateurs PEFT bougent:
      - memes fichiers, poids differents -> set_adapters (immediat)
      - jeu de LoRA different            -> unload_lora_weights + reload des LoRA (~1s)
    Les pipes derives (from_pipe) partagent ce transformer -> ils suivent automatiquement.
    Renvoie True si applique, False si echec (le caller retombe sur un reload complet)."""
    global _APPLIED_LORAS
    if not force and _APPLIED_LORAS == LORAS:
        return True
    old_paths = [p for p, _ in _APPLIED_LORAS]
    new_paths = [p for p, _ in LORAS]
    try:
        if not force and old_paths and old_paths == new_paths:
            # Seuls les poids changent -> re-ponderation instantanee.
            pipe.set_adapters(_lora_names(LORAS), [float(w) for _, w in LORAS])
            _APPLIED_LORAS = list(LORAS)
            _log("LoRA weights updated in place (no reload): "
                 + ", ".join(f"{os.path.basename(p)}@{w}" for p, w in LORAS))
            return True
        if old_paths or force:
            _clear_loras(pipe)
        names, weights = [], []
        for i, (p, w) in enumerate(LORAS):
            if os.path.isfile(p):
                an = f"cz_lora_{i}"
                _log(f"applying LoRA: {os.path.basename(p)} (weight {w})")
                # Passer le dossier + weight_name (et non le chemin complet) : sinon
                # diffusers en mode offline (HF_HUB_OFFLINE) refuse "must specify a
                # weight_name". Marche aussi online et avec un fichier local direct.
                pipe.load_lora_weights(os.path.dirname(p) or ".",
                                       weight_name=os.path.basename(p), adapter_name=an)
                names.append(an)
                weights.append(float(w))
            else:
                _log(f"LoRA file not found, ignored: {p}")
        if names:
            pipe.set_adapters(names, weights)
        _APPLIED_LORAS = list(LORAS)
        if not force:
            _log("LoRAs hot-swapped (no model reload)")
        return True
    except Exception as e:
        _log(f"LoRA hot-swap failed ({e}); falling back to a full reload")
        _APPLIED_LORAS = []
        return False


def _swap_transformer(pipe):
    """Remplace SEULEMENT le transformer du pipeline deja en cache: le VAE, l'encodeur de
    texte, le tokenizer et le scheduler restent en VRAM (c'est eux le gros du temps de
    chargement). Valable uniquement a repo de base + offload EFFECTIF identiques.

    Renvoie True si l'echange a reussi, False -> le caller fait un reload complet."""
    global _APPLIED_LORAS, _DERIVED
    t0 = time.time()
    old_t = _LOADED_KEY[1] if _LOADED_KEY else None
    # Passer de/vers un GGUF change l'offload EFFECTIF (un GGUF impose 'model') -> les
    # hooks accelerate et le placement different: on ne bricole pas, on recharge.
    if _effective_offload(old_t) != _effective_offload(ZIMAGE_TRANSFORMER):
        _log("transformer swap skipped (GGUF changes the effective offload) -> full reload")
        return False
    try:
        _log(f"switching Krea 2 transformer -> {ZIMAGE_TRANSFORMER or BASE_REPO} "
             "(keeping VAE + text encoder in VRAM)")
        new_t = _load_transformer()
        old = getattr(pipe, "transformer", None)
        off = _effective_offload()
        # Offload: les hooks accelerate sont poses sur les composants. Il faut les retirer
        # avant l'echange, sinon le nouveau transformer n'en a pas et l'ancien garde les siens.
        if DEVICE == "cuda" and off in ("model", "sequential"):
            try:
                pipe.remove_all_hooks()
            except Exception as e:
                _dbg(f"remove_all_hooks: {e}")
        try:
            pipe.register_modules(transformer=new_t)   # API diffusers (met a jour le config)
        except Exception:
            pipe.transformer = new_t
        # Liberer l'ANCIEN transformer AVANT de poser le nouveau sur le GPU: sinon
        # ancien + nouveau + VAE/encodeurs depassent la VRAM -> spill en RAM partagee
        # qui ne se resorbe pas (mesure sur une grille XYZ multi-checkpoints cote
        # studio: 1.7 s/step -> 300-600 s/step, puis crash). Les pipes derives
        # (from_pipe) pointent aussi sur l'ancien -> a purger d'abord, sinon
        # `del old` ne libere rien (from_pipe est gratuit, il sera reconstruit).
        _DERIVED = {}
        del old
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if DEVICE == "cuda":
            if off == "model":
                pipe.enable_model_cpu_offload()
            elif off == "sequential":
                pipe.enable_sequential_cpu_offload()
            else:
                new_t.to(DEVICE)       # jamais un GGUF ici (offload force a 'model')
        # Les adaptateurs LoRA etaient poses sur l'ancien transformer -> a reposer.
        _APPLIED_LORAS = []
        if LORAS:
            _apply_loras(pipe, force=True)
        _log(f"transformer switched in {time.time() - t0:.1f}s "
             "(VAE + text encoder kept, no full reload)")
        return True
    except Exception as e:
        _log(f"transformer hot-swap failed ({e}); falling back to a full reload")
        _APPLIED_LORAS = []
        return False


def _ensure_base():
    """Charge (si besoin) le pipeline de base txt2img. Gere le transformer
    single-file/GGUF et l'offload. Cache par (repo, transformer, offload).

    Deux echanges a chaud evitent un rechargement complet (transformer + VAE + encodeur
    texte, des dizaines de secondes):
      - LoRA differentes            -> _apply_loras (adaptateurs PEFT seuls)
      - transformer different, meme repo de base + offload -> _swap_transformer."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _BASE_SCHED_CONFIG, _APPLIED_LORAS
    key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE)
    _dbg(f"_ensure_base key={key} cached={_LOADED_KEY}")
    if _BASE_PIPE is not None and _LOADED_KEY == key:
        if _apply_loras(_BASE_PIPE):
            _dbg("base pipeline: reusing cached (no reload)")
            return _BASE_PIPE
        _dbg("base pipeline: LoRA hot-swap failed -> free + reload")
        free_vram()
    elif _BASE_PIPE is not None:
        # Seul le transformer change (meme repo de base + meme offload) ? -> on ne recharge
        # QUE le transformer et on garde VAE + encodeur texte en VRAM.
        if (_LOADED_KEY and _LOADED_KEY[0] == BASE_REPO and _LOADED_KEY[2] == OFFLOAD_MODE
                and _swap_transformer(_BASE_PIPE)):
            _LOADED_KEY = key
            return _BASE_PIPE
        _dbg("base pipeline: key changed -> free + reload")
        free_vram()
    from diffusers import Krea2Pipeline
    t0 = time.time()
    # Le transformer est TOUJOURS charge separement (contrairement aux autres forks): c'est
    # la seule facon de lui appliquer la quantification torchao, indispensable pour tenir
    # en 32 Go. Sans override explicite, on quantifie celui du repo de base.
    kwargs = {"transformer": _load_transformer()}
    _log(f"loading Krea 2 base: {BASE_REPO} (offload={OFFLOAD_MODE}, dtype=bf16, "
         f"quant={QUANT_MODE}) ... first time downloads ~35.7 GB from HF (gated), then cached")
    pipe = _load_monitor(f"Krea 2 base {BASE_REPO}",
                         lambda: Krea2Pipeline.from_pretrained(BASE_REPO, torch_dtype=DTYPE,
                                                               **kwargs))
    # Capture le config natif (flow-matching) du scheduler -> base pour construire les
    # autres samplers (euler/dpm2a/dpmpp2m) sans perdre shift/flow params.
    try:
        _BASE_SCHED_CONFIG = dict(pipe.scheduler.config)
    except Exception:
        _BASE_SCHED_CONFIG = None
    # LoRA (sur le transformer du base -> partage par les pipes derives).
    # force=True: pipe neuf, aucun adaptateur pose -> on (re)pose tout.
    _APPLIED_LORAS = []
    if LORAS:
        _apply_loras(pipe, force=True)
    # Attention slicing: POSE PAR APPEL via _set_slicing (selon la resolution traitee),
    # PAS au chargement. En tuile/1024 -> slicing OFF = attention SDPA native, rapide
    # (comme ComfyUI). Whole-image 2K+ -> slicing ON pour eviter le spill VRAM 32 Go.
    # enable_*_cpu_offload gere lui-meme le device -> ne PAS faire .to(cuda) alors.
    # IMPORTANT: un transformer GGUF quantifie ne se deplace PAS sur le GPU via .to(cuda)
    # (offload=none) ni en sequential -> il reste sur CPU = ULTRA lent (VRAM vide, ~500s/step).
    # Seul enable_model_cpu_offload (accelerate) le pose correctement sur le GPU pendant le
    # forward. On force donc 'model' pour un base GGUF, quel que soit le reglage UI/config.
    _off = _effective_offload()
    if _off != OFFLOAD_MODE:
        _log(f"quantized base: offload '{OFFLOAD_MODE}' forced to '{_off}' (a quantized "
             f"transformer does not run on GPU in none/sequential -> would stay on CPU, "
             f"~500s/step)")
    if DEVICE == "cuda" and _off == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and _off == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    # VAE tiling/slicing: indispensable pour l'img2img/upscale. Qwen-Image est gros (~20B
    # transformer + encodeur texte) -> sans tiling le VAE peut faire deborder la VRAM (spill
    # RAM partagee = tres lent). Tuiler le VAE plafonne ce pic (comme le "tiled decode" de
    # ComfyUI). Le VAE est partage par les pipes derives.
    try:
        pipe.vae.config.force_upcast = False   # VAE en bf16 (fp32 lent sur Blackwell) -- TOUJOURS
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception as e:
        _dbg(f"VAE tiling not available: {e}")
    _apply_sampler(pipe)   # pose le sampler choisi (euler par defaut) sur le pipe de base
    _BASE_PIPE = pipe
    _DERIVED = {"txt2img": pipe}
    _LOADED_KEY = key
    _log(f"Krea 2 base ready in {time.time() - t0:.1f}s (sampler={SAMPLER}/{SCHEDULE})")
    return pipe


_UNSUPPORTED_MSG = {
    "img2img": ("Refine / upscale-refine / harmonize are unavailable with Krea 2: "
                "diffusers exposes no Krea2Img2ImgPipeline. ESRGAN-only upscaling still "
                "works (it never goes through the diffusion model)."),
    "inpaint": ("Inpaint / Outpaint / Reframe(contain) are unavailable with Krea 2: "
                "diffusers exposes no Krea2InpaintPipeline. Reframe in 'cover' mode (a "
                "plain crop, no diffusion) still works."),
    "omni": ("Instruction editing is unavailable: Krea 2 has no editing model equivalent "
             "to Qwen-Image-Edit."),
}


def get_pipe(kind="img2img"):
    """Renvoie le pipeline demande. Krea 2 n'expose que le txt2img.

    Filet de securite: l'UI masque deja les controles img2img/inpaint/omni via
    CAPABILITIES, mais le CLI et les appels directs passent encore par ici -> on leve
    UnsupportedFeature avec un message actionnable plutot que de laisser un ImportError
    diffusers ou un silencieux repli sur le txt2img (qui produirait une image sans
    rapport avec l'entree)."""
    base = _ensure_base()
    if kind in _DERIVED:
        _dbg(f"get_pipe('{kind}'): reuse derived")
        return _DERIVED[kind]
    if kind in _UNSUPPORTED_MSG and not supports(kind):
        raise UnsupportedFeature(_UNSUPPORTED_MSG[kind])
    cls = None
    if cls is None:
        return base
    _log(f"deriving {kind} pipeline (shared weights, no extra VRAM)")
    # Un transformer GGUF est QUANTIFIE: on ne peut pas le recaster en dtype (.to(DTYPE)
    # leve "Casting a quantized model is unsupported"). On saute donc le recast bf16 dans
    # ce cas (le compute_dtype est deja bf16). Sinon (bf16 plein): recast defensif Blackwell
    # (certains from_pipe upcastent en float32 -> tres lent sans tensor cores fp32).
    quantized = bool(ZIMAGE_TRANSFORMER) and ZIMAGE_TRANSFORMER.lower().endswith(".gguf")
    try:
        # GGUF quantifie: torch_dtype=None EXPLICITE -> sinon from_pipe met float32 par
        # defaut et caste le modele quantifie -> ValueError "Casting a quantized model".
        p = cls.from_pipe(base, torch_dtype=None) if quantized else cls.from_pipe(base, torch_dtype=DTYPE)
    except TypeError:
        p = cls.from_pipe(base)
    try:
        if not quantized:
            p = p.to(DTYPE)
        p.vae.config.force_upcast = False
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        _log(f"img2img bf16 recast failed ({e})")
    _apply_sampler(p)   # meme sampler que le base (au cas ou from_pipe recree le scheduler)
    # Diagnostic vitesse: si le pipe derive n'est PAS sur cuda -> img2img/refine tourne
    # sur CPU = ultra lent. On le force sur DEVICE en mode plein VRAM (offload gere seul).
    # NB: offload EFFECTIF (un base GGUF force 'model' meme si l'UI dit 'none'): en
    # offload, un transformer "sur CPU" est normal -> un .to(cuda) casserait les hooks.
    try:
        tdev = next(p.transformer.parameters()).device
        if DEVICE == "cuda" and _effective_offload() == "none" and tdev.type != "cuda":
            _log(f"{kind} pipeline was on {tdev} -> moving to {DEVICE}")
            p = p.to(DEVICE)
            tdev = next(p.transformer.parameters()).device
        _log(f"{kind} pipeline ready: transformer={tdev}")
    except Exception as e:
        _dbg(f"device check failed: {e}")
    _DERIVED[kind] = p
    return p


def _load_omni():
    """Krea 2 n'a pas de modele d'edition par instruction (pas d'equivalent
    Qwen-Image-Edit chez Krea). Conserve pour compat d'API: l'onglet correspondant
    est masque par l'UI via CAPABILITIES['omni'] = False."""
    raise UnsupportedFeature(_UNSUPPORTED_MSG["omni"])


@_gpu_serial
def generate_omni(refs, prompt, negative, width, height, steps, seed):
    """Edition par instruction Qwen-Image-Edit: edite une (ou plusieurs, via 2509) image(s)
    d'entree selon le prompt d'instruction. Conserve la signature de l'upstream (cz_ui).
    width/height sont ignores: l'edition preserve les dimensions de l'image d'entree."""
    refs = [r.convert("RGB") for r in (refs or []) if r is not None]
    if not refs:
        raise ValueError("Edit needs at least one input image.")
    pipe = get_pipe("omni")
    _log(f"edit: {len(refs)} image(s), {int(steps)} steps, cfg {GUIDANCE:.1f} ...")
    _progress(0.1, f"Editing ({len(refs)} image(s))...")
    _set_slicing(pipe, max(max(r.size) for r in refs))
    t0 = time.time()
    # 2509/Plus accepte une liste d'images; la revision de base prend une seule image.
    image_arg = refs if len(refs) > 1 else refs[0]
    out = _qwen_call(
        pipe,
        image=image_arg,
        prompt=prompt or "",
        num_inference_steps=int(steps),
        generator=_make_generator(seed),
        **_cfg(negative),
    ).images[0]
    _log(f"edit done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def load_pipe():
    """Compat: pipeline img2img (etage de raffinement)."""
    return get_pipe("img2img")


@_gpu_serial
def generate(prompt, width, height, steps, seed, negative_prompt=""):
    """txt2img Krea 2: genere une image depuis un prompt. guidance_scale direct
    (= curseur guidance, ~4.0), ~30-50 steps conseilles. Le negative prompt agit grace au
    vrai CFG (cf. _cfg)."""
    pipe = get_pipe("txt2img")
    w = round_to_multiple(int(width))
    h = round_to_multiple(int(height))
    _log(f"txt2img: {w}x{h}, {int(steps)} steps, cfg {GUIDANCE:.1f} ...")
    _dbg(f"txt2img seed={seed} dtype=bf16 device={DEVICE} offload={OFFLOAD_MODE} "
         f"transformer={'single-file' if ZIMAGE_TRANSFORMER else 'repo'}")
    if DEVICE == "cuda":
        _dbg(f"VRAM before: alloc={torch.cuda.memory_allocated()/1024**3:.2f} Go")
    _progress(0.1, f"Generating {w}x{h} ({int(steps)} steps)...")
    _set_slicing(pipe, max(w, h))
    t0 = time.time()
    img = _qwen_call(
        pipe,
        prompt=prompt or "",
        width=w, height=h,
        num_inference_steps=int(steps),
        generator=_make_generator(seed),
        **_cfg(negative_prompt),
    ).images[0]
    _log(f"txt2img done in {time.time() - t0:.1f}s")
    if DEVICE == "cuda":
        _dbg(f"VRAM peak: alloc={torch.cuda.max_memory_allocated()/1024**3:.2f} Go | "
             f"reserved={torch.cuda.max_memory_reserved()/1024**3:.2f} Go")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return img


def round_to_multiple(x, m=16):
    return max(m, int(round(x / m) * m))


def set_force_ratio(spec):
    """Definit le ratio force pour upscale/img2img: 'W:H' / 'WxH' (ex '13:19', '832x1216')
    ou '' pour desactiver (ratio natif preserve). Pilote par le radio UI."""
    global FORCE_RATIO
    FORCE_RATIO = (spec or "").strip()
    _log(f"force ratio -> {FORCE_RATIO or '(off, ratio natif preserve)'}")


def set_force_ratio_mode(mode):
    """'crop' (recadrage centre) ou 'extend' (outpaint -- indisponible sur Krea 2)."""
    global FORCE_RATIO_MODE
    FORCE_RATIO_MODE = "extend" if str(mode or "").strip().lower() == "extend" else "crop"
    _log(f"force ratio mode -> {FORCE_RATIO_MODE}")


def _parse_ratio(spec):
    """(w, h) depuis 'W:H', 'WxH', ou un label '832 x 1216 | 13:19'; sinon None."""
    import re
    if not spec:
        return None
    m = re.search(r"(\d+)\s*[:xX×]\s*(\d+)", str(spec))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if a > 0 and b > 0 else None


def _crop_to_ratio(image, ratio_w, ratio_h):
    """Recadre (centre) l'image au ratio ratio_w:ratio_h en gardant l'aire maximale."""
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur > target:                       # trop large -> couper les cotes
        nw = max(1, int(round(h * target)))
        x0 = (w - nw) // 2
        return image.crop((x0, 0, x0 + nw, h))
    nh = max(1, int(round(w / target)))    # trop haut -> couper haut/bas
    y0 = (h - nh) // 2
    return image.crop((0, y0, w, y0 + nh))


def _extend_to_ratio(image, ratio_w, ratio_h, prompt, steps, seed):
    """Amene l'image au ratio cible en l'ETENDANT (outpaint) au lieu de recadrer.
    Sur Krea 2 il n'y a PAS de pipeline inpaint: outpaint_directions leve le message
    UnsupportedFeature clair. Garde pour parite de code avec la famille (l'UI ne
    propose pas ce mode ici; seul un force_ratio_mode='extend' en config y mene)."""
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur < target:                       # trop etroit -> elargir gauche + droite
        pad = target * h - w
        return outpaint_directions(image, None, ["left", "right"], prompt, steps, seed,
                                   expand=pad / (2.0 * w))
    pad = w / target - h                   # trop large -> etendre haut + bas
    return outpaint_directions(image, None, ["top", "bottom"], prompt, steps, seed,
                               expand=pad / (2.0 * h))


def _reframe_canvas(image, ratio_w, ratio_h, overlap=8):
    """Place l'image dans un canevas plus grand au ratio cible (expansion sur 1 axe),
    + un masque (blanc = a remplir, noir = a garder, avec un petit overlap)."""
    from PIL import ImageDraw
    image = image.convert("RGB")
    w, h = image.size
    r = ratio_w / ratio_h
    # Alignement sur 32 (patch 2 x VAE 16): evite les erreurs de conv (no engine).
    if w / h < r:  # trop etroit -> elargir
        nw, nh = round_to_multiple(int(round(h * r)), 32), round_to_multiple(h, 32)
    else:          # trop large -> agrandir en hauteur
        nw, nh = round_to_multiple(w, 32), round_to_multiple(int(round(w / r)), 32)
    nw, nh = max(nw, round_to_multiple(w, 32)), max(nh, round_to_multiple(h, 32))
    ox, oy = (nw - w) // 2, (nh - h) // 2
    canvas = Image.new("RGB", (nw, nh), (127, 127, 127))
    canvas.paste(image, (ox, oy))
    mask = Image.new("L", (nw, nh), 255)
    ImageDraw.Draw(mask).rectangle(
        [ox + overlap, oy + overlap, ox + w - overlap, oy + h - overlap], fill=0)
    return canvas, mask, nw, nh


@_gpu_serial
def inpaint_run(background, mask, prompt, steps, denoise, seed):
    """Inpaint: regenere la zone blanche du masque selon le prompt
    (ZImageInpaintPipeline). background + mask = PIL (L: blanc = a changer)."""
    orig = background.convert("RGB")
    full_mask = mask
    # Diffusion bornee a ~1 MP (zone optimale du modele), puis recomposition pleine res.
    bg, work_mask, orig_size = _cap_work_res(orig, mask)
    w, h = bg.size
    pipe = get_pipe("inpaint")
    _log(f"inpaint: work {w}x{h} (orig {orig_size[0]}x{orig_size[1]}), {int(steps)} steps, "
         f"strength {float(denoise):.2f}, cfg {GUIDANCE:.1f} ...")
    _progress(0.1, "Inpainting...")
    _set_slicing(pipe, max(w, h))
    t0 = time.time()
    out = _qwen_call(pipe, prompt=prompt or "", image=bg, mask_image=work_mask,
                     strength=float(denoise), num_inference_steps=int(steps),
                     generator=_make_generator(seed), **_cfg(None)).images[0]
    # Recompose: hors-masque garde la pleine resolution; jointure fondue (feather).
    out = _composite_back(out, orig, full_mask, orig_size,
                          feather=max(2, int(min(orig_size) * 0.01)))
    _log(f"inpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# Resolution cible "zone optimale" du modele Z-Image (~1 MP, comme les ratios txt2img).
# Le reframe vise ce budget pour ne PAS exploser le nombre de pixels (sortie 2-3 MP qui
# sort de la zone d'entrainement -> lent et qualite degradee).
MODEL_TARGET_PX = 1024 * 1024


def _ratio_canvas(ratio_w, ratio_h, target_px=MODEL_TARGET_PX):
    """Dimensions (multiples de 32) d'un canevas au ratio donne, a ~target_px pixels."""
    r = float(ratio_w) / float(ratio_h)
    nh = (target_px / r) ** 0.5
    nw = nh * r
    return round_to_multiple(int(round(nw)), 32), round_to_multiple(int(round(nh)), 32)


def _cap_work_res(image, mask, max_px=MODEL_TARGET_PX):
    """Borne la resolution de travail pour la diffusion: si image > max_px, renvoie une
    version reduite (multiples de 32) de (image, mask) + la taille d'origine pour
    recomposer ensuite. Evite de faire tourner le modele tres au-dessus de sa zone
    optimale (~1 MP) -> plus rapide et meilleure qualite."""
    w, h = image.size
    if w * h > max_px:
        s = (max_px / (w * h)) ** 0.5
        ww, wh = round_to_multiple(int(w * s), 32), round_to_multiple(int(h * s), 32)
    else:
        ww, wh = round_to_multiple(w, 32), round_to_multiple(h, 32)
    img_w = image.resize((ww, wh), Image.LANCZOS) if (ww, wh) != image.size else image
    msk_w = mask.resize((ww, wh), Image.NEAREST) if mask.size != (ww, wh) else mask
    return img_w, msk_w, (w, h)


def _composite_back(result, original, mask, orig_size, feather=0):
    """Recompose a la resolution d'origine: la zone masquee (blanc) vient de `result`
    (re-agrandi a orig_size), le reste vient de `original` -> le hors-masque garde la
    pleine resolution de l'image de depart. `feather` (px) floute le masque pour fondre
    la jointure (transition progressive original <-> genere, plus de ligne dure)."""
    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)
    if original.size != orig_size:
        original = original.resize(orig_size, Image.LANCZOS)
    m = (mask.resize(orig_size, Image.NEAREST) if mask.size != orig_size else mask).convert("L")
    if feather and feather > 0:
        from PIL import ImageFilter
        m = m.filter(ImageFilter.GaussianBlur(float(feather)))
    return Image.composite(result, original.convert("RGB"), m)


def reframe(image, ratio_w, ratio_h, fit, prompt, steps, seed, strength=1.0):
    """Recadre l'image au ratio cible en bornant la sortie a la resolution optimale du
    modele (~1 MP) -> plus d'explosion du nombre de pixels.
      fit='contain' : l'image entiere rentre dans le canevas (sans l'agrandir), les bords
                      ajoutes sont remplis par Z-Image (outpaint).
      fit='cover'   : l'image remplit le canevas au ratio puis est recadree au centre
                      (pas d'outpaint, simple reframe/crop)."""
    from PIL import ImageDraw
    img = image.convert("RGB")
    w, h = img.size
    nw, nh = _ratio_canvas(ratio_w, ratio_h)
    if str(fit).lower() == "cover":
        scale = max(nw / w, nh / h)
        rw2, rh2 = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((rw2, rh2), Image.LANCZOS)
        left, top = (rw2 - nw) // 2, (rh2 - nh) // 2
        out = resized.crop((left, top, left + nw, top + nh))
        _log(f"reframe cover: {w}x{h} -> {nw}x{nh} (crop, no fill)")
        return out
    # contain -> on adapte l'original sans l'agrandir, puis on outpaint les bords.
    from PIL import ImageFilter
    scale = min(nw / w, nh / h, 1.0)
    rw2, rh2 = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((rw2, rh2), Image.LANCZOS) if (rw2, rh2) != (w, h) else img
    ox, oy = (nw - rw2) // 2, (nh - rh2) // 2
    # Bords = extension floue des couleurs du bord (blurred edge fill, comme l'outpaint)
    # plutot qu'un gris -> continuite d'exposition; transparait si strength < 1.0.
    arr = np.pad(np.array(resized), [[oy, nh - rh2 - oy], [ox, nw - rw2 - ox], [0, 0]],
                 mode="edge")
    canvas = Image.fromarray(np.ascontiguousarray(arr))
    overlap = 8
    mask = Image.new("L", (nw, nh), 255)
    ImageDraw.Draw(mask).rectangle(
        [ox + overlap, oy + overlap, ox + rw2 - overlap, oy + rh2 - overlap], fill=0)
    blur_r = max(8, int(min(nw, nh) * 0.03))
    canvas = Image.composite(canvas.filter(ImageFilter.GaussianBlur(blur_r)), canvas, mask)
    pipe = get_pipe("inpaint")
    _log(f"reframe contain (outpaint): {w}x{h} -> {nw}x{nh}, {int(steps)} steps, "
         f"strength {float(strength):.2f}, cfg {GUIDANCE:.1f} ...")
    _progress(0.1, f"Reframe -> {nw}x{nh}...")
    _set_slicing(pipe, max(nw, nh))
    t0 = time.time()
    out = _qwen_call(pipe, prompt=prompt or "", image=canvas, mask_image=mask,
                     strength=float(strength), num_inference_steps=int(steps),
                     generator=_make_generator(seed), **_cfg(None)).images[0]
    if out.size != (nw, nh):
        out = out.resize((nw, nh), Image.LANCZOS)
    _log(f"reframe done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


@_gpu_serial
def outpaint(image, ratio_w, ratio_h, prompt, steps, seed):
    """Compat (CLI --reframe et appels existants): reframe en mode 'contain' (outpaint),
    borne a la resolution optimale du modele."""
    return reframe(image, ratio_w, ratio_h, "contain", prompt, steps, seed)


def outpaint_directions(image, mask, directions, prompt, steps, seed, strength=1.0, expand=0.3):
    """Outpaint directionnel (facon Fooocus): agrandit l'image dans les directions
    choisies parmi left/right/top/bottom, chacune de `expand` (fraction de la dimension
    d'origine), en repliquant les pixels du bord (mode 'edge'), puis fait remplir les
    bandes ajoutees par Z-Image (ZImageInpaintPipeline). Un `mask` peint (L, blanc = a
    changer) est optionnel: il est conserve dans la zone d'origine et combine avec les
    bandes ajoutees (blanches)."""
    img = np.array(image.convert("RGB"))
    H, W = img.shape[:2]
    m = np.array(mask.convert("L")) if mask is not None else np.zeros((H, W), dtype=np.uint8)
    dirs = set(d.lower() for d in (directions or []))
    if "top" in dirs:
        p = int(H * expand)
        img = np.pad(img, [[p, 0], [0, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[p, 0], [0, 0]], mode="constant", constant_values=255)
    if "bottom" in dirs:
        p = int(H * expand)
        img = np.pad(img, [[0, p], [0, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[0, p], [0, 0]], mode="constant", constant_values=255)
    if "left" in dirs:
        p = int(W * expand)
        img = np.pad(img, [[0, 0], [p, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[0, 0], [p, 0]], mode="constant", constant_values=255)
    if "right" in dirs:
        p = int(W * expand)
        img = np.pad(img, [[0, 0], [0, p], [0, 0]], mode="edge")
        m = np.pad(m, [[0, 0], [0, p]], mode="constant", constant_values=255)
    canvas = Image.fromarray(np.ascontiguousarray(img))
    mask_img = Image.fromarray(np.ascontiguousarray(m))
    full_size = canvas.size
    # Dilate un peu la zone a generer vers l'interieur -> le modele regenere une fine
    # bande de transition qui se raccorde a l'original (evite la jointure franche).
    from PIL import ImageFilter
    k = max(3, (int(min(full_size) * 0.02) // 2) * 2 + 1)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(min(k, 15)))
    # "Blurred edge fill": on remplit la zone a generer avec une version FLOUE de
    # l'extension du bord (memes couleurs/tonalite que l'original) au lieu d'un bord
    # replique net. Avec strength < 1.0 ce flou transparait -> continuite d'exposition
    # (plus de bande plus claire) et le modele ajoute le detail par-dessus.
    blur_r = max(8, int(min(full_size) * 0.03))
    canvas = Image.composite(canvas.filter(ImageFilter.GaussianBlur(blur_r)), canvas, mask_img)
    # Diffusion bornee a ~1 MP (zone optimale), puis recomposition: le centre (image
    # d'origine) garde sa pleine resolution, seuls les bords ajoutes sont generes.
    work_img, work_mask, _ = _cap_work_res(canvas, mask_img)
    w2, h2 = work_img.size
    pipe = get_pipe("inpaint")
    _log(f"outpaint {sorted(dirs)}: {image.size[0]}x{image.size[1]} -> "
         f"{full_size[0]}x{full_size[1]} (work {w2}x{h2}), {int(steps)} steps, "
         f"cfg {GUIDANCE:.1f} ...")
    _progress(0.1, f"Outpaint -> {full_size[0]}x{full_size[1]}...")
    _set_slicing(pipe, max(w2, h2))
    t0 = time.time()
    out = _qwen_call(pipe, prompt=prompt or "", image=work_img, mask_image=work_mask,
                     strength=float(strength), num_inference_steps=int(steps),
                     generator=_make_generator(seed), **_cfg(None)).images[0]
    out = _composite_back(out, canvas, mask_img, full_size,
                          feather=max(4, int(min(full_size) * 0.015)))
    _log(f"outpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def _make_generator(seed):
    return torch.Generator(DEVICE).manual_seed(int(seed)) if int(seed) >= 0 else None


@_gpu_serial
def _refine_whole(pipe, image, denoise, steps, prompt, seed):
    """Passe Qwen-Image img2img sur l'image entiere (ou une tuile). Le slicing est pose
    selon la taille reelle traitee: tuile 1024 -> OFF (rapide), whole 2K+ -> ON.
    IMPORTANT: on passe width/height = taille de l'image (alignee sur 16). Sinon Qwen-Image
    img2img retombe sur son defaut (height = default_sample_size * vae_scale_factor = 1024)
    et REDIMENSIONNE l'entree en 1024x1024 -> le ratio est ecrase (bug). En forcant les
    dimensions de l'entree, le ratio d'origine est preserve en upscale/img2img."""
    _set_slicing(pipe, max(image.size))
    w = round_to_multiple(image.width, 16)
    h = round_to_multiple(image.height, 16)
    return _qwen_call(
        pipe,
        prompt=prompt or "",
        image=image,
        width=w, height=h,
        strength=float(denoise),
        num_inference_steps=int(steps),
        generator=_make_generator(seed),
        **_cfg(None),
    ).images[0]


def _feather_mask_np(th, tw, overlap, left, right, top, bottom):
    """Masque (th, tw, 1) a rampe lineaire sur les bords qui jouxtent une autre tuile."""
    mask = np.ones((th, tw, 1), dtype=np.float32)
    f = int(overlap)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        if left:
            mask[:, :f, 0] *= ramp[np.newaxis, :]
        if right:
            mask[:, tw - f:, 0] *= ramp[::-1][np.newaxis, :]
        if top:
            mask[:f, :, 0] *= ramp[:, np.newaxis]
        if bottom:
            mask[th - f:, :, 0] *= ramp[::-1][:, np.newaxis]
    return mask


def _refine_tiled(pipe, image, denoise, steps, prompt, seed, tile, overlap):
    """Passe Z-Image en tuiles avec recomposition feather (facon Ultimate SD Upscale).
    Plafonne le pic VRAM (une tuile a la fois) et permet le 4K+ sans coutures.
    Memes rampe lineaire + overlap-add que esrgan_upscale, mais a scale 1 sur PIL."""
    w, h = image.size
    tile = round_to_multiple(tile)                       # multiple de 16 pour le VAE
    overlap = max(0, min(int(overlap), tile - 16))
    if w <= tile and h <= tile:
        # Une seule tuile = image entiere -> pas de duplication possible: denoise demande.
        return _refine_whole(pipe, image, denoise, steps, prompt, seed)
    # Anti-duplication 1: prompt vide par tuile (le prompt global decrit toute la compo).
    prompt = _tile_prompt(prompt)
    if not (prompt or "").strip():
        _log("refine tiled: prompt vide par tuile (anti-duplication; regle refine_tile_prompt).")
    # Anti-duplication 2 (filet): a fort denoise chaque tuile peut encore deriver.
    denoise = float(denoise)
    if _TILE_DENOISE_CAP > 0 and denoise > _TILE_DENOISE_CAP:
        _log(f"refine tiled: denoise {denoise:.2f} > plafond {_TILE_DENOISE_CAP:.2f} -> "
             f"reduit a {_TILE_DENOISE_CAP:.2f} (regle refine_tile_denoise_cap).")
        denoise = _TILE_DENOISE_CAP

    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    step = max(16, tile - overlap)
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    total = len(ys) * len(xs)
    _log(f"refine: tiled {w}x{h}, tile {tile} overlap {overlap} -> {len(xs)}x{len(ys)} = {total} tiles")
    i = 0
    for y in ys:
        for x in xs:
            if _STOP:
                _log("refine tiled: stop requested")
                break
            i += 1
            x2, y2 = min(x + tile, w), min(y + tile, h)
            x1, y1 = max(x2 - tile, 0), max(y2 - tile, 0)
            cw, ch = x2 - x1, y2 - y1
            _progress(0.45 + 0.5 * (i - 1) / max(1, total), f"Refine tile {i}/{total}")
            crop = image.crop((x1, y1, x2, y2))
            _t_tile = time.time()
            out = _refine_whole(pipe, crop, denoise, steps, prompt, seed)
            _log(f"  tile {i}/{total} ({cw}x{ch}) in {time.time() - _t_tile:.1f}s{_vram_str()}")
            if out.size != (cw, ch):
                out = out.resize((cw, ch), Image.LANCZOS)
            out_arr = np.asarray(out.convert("RGB"), dtype=np.float32) / 255.0
            mask = _feather_mask_np(ch, cw, overlap,
                                    left=x1 > 0, right=x2 < w, top=y1 > 0, bottom=y2 < h)
            acc[y1:y2, x1:x2, :] += out_arr * mask
            weight[y1:y2, x1:x2, :] += mask

    out = acc / np.clip(weight, 1e-6, None)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


# ----------------------------------------------------------------------------
# Orchestration : process_one, batch txt2img (run/_gen_meta restent dans app.py
# car run emet des gr.Error pour l'UI).
# ----------------------------------------------------------------------------
@_gpu_serial
def process_one(image, esrgan_model, factor, denoise, steps, prompt, seed, tile, overlap,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                do_esrgan=True, refine_first=False, apply_force_ratio=False):
    """Pipeline sur une PIL Image, renvoie (image, timings_dict).
    do_esrgan=False -> img2img pur (saute l'etage ESRGAN, refine sur l'image native).
    refine_first=True -> refine PUIS ESRGAN (la diffusion tourne a la resolution
    native = bien plus rapide), au lieu de ESRGAN PUIS refine (detail en haute-def).
    apply_force_ratio=True + FORCE_RATIO defini -> amene l'ENTREE au ratio choisi avant
    traitement: FORCE_RATIO_MODE 'crop' = recadrage centre (facon Fooocus), 'extend' =
    outpaint (indisponible sur Krea 2 -> UnsupportedFeature). Sinon: ratio natif."""
    timings = {"esrgan": 0.0, "refine": 0.0}
    image = image.convert("RGB")
    if apply_force_ratio and FORCE_RATIO:
        r = _parse_ratio(FORCE_RATIO)
        if r:
            _before = image.size
            if FORCE_RATIO_MODE == "extend":
                image = _extend_to_ratio(image, r[0], r[1], prompt, max(6, int(steps)), seed)
                _verb = "extend (outpaint)"
            else:
                image = _crop_to_ratio(image, r[0], r[1])
                _verb = "crop"
            _log(f"force ratio {r[0]}:{r[1]} -> {_verb} {_before[0]}x{_before[1]} "
                 f"to {image.size[0]}x{image.size[1]}")
    w0, h0 = image.size
    use_esrgan = bool(do_esrgan and esrgan_model)
    do_refine = float(denoise) > 0.001
    _dbg(f"process_one in={w0}x{h0} factor={factor} denoise={denoise} steps={int(steps)} "
         f"do_esrgan={do_esrgan} refine_first={refine_first} esrgan={esrgan_model} "
         f"refine_tile={int(refine_tile)}")

    def _esrgan_stage(img):
        t0 = time.time()
        iw, ih = img.size
        _progress(0.15, f"ESRGAN upscale {iw}x{ih}...")
        model = load_esrgan(esrgan_model)
        _log(f"ESRGAN upscale: {iw}x{ih} (tile {int(tile)}) ...")
        up = esrgan_upscale(img, model, int(tile), int(overlap))
        # Cible = facteur applique a la taille d'origine (independant de l'ordre).
        target_w = round_to_multiple(w0 * factor)
        target_h = round_to_multiple(h0 * factor)
        up = up.resize((target_w, target_h), Image.LANCZOS)
        timings["esrgan"] += time.time() - t0
        _log(f"ESRGAN done in {timings['esrgan']:.1f}s -> {target_w}x{target_h}")
        return up

    def _refine_stage(img):
        t0 = time.time()
        pipe = load_pipe()
        rw, rh = img.size
        rt = int(refine_tile)
        # Garde-fou anti-crash: refine whole-image trop grand (4K+) -> auto-tuilage.
        if rt <= 0 and max(rw, rh) > _AUTO_TILE_ABOVE:
            rt = _pick_refine_tile(rw, rh, int(refine_overlap) or 64)
            _log(f"refine: image {rw}x{rh} > {_AUTO_TILE_ABOVE}px -> auto-tiling (tile {rt}) "
                 "to avoid the VRAM peak (settings: auto_refine_tile_above, auto_refine_tile)")
        if rt > 0:
            out = _refine_tiled(pipe, img, denoise, steps, prompt, seed,
                                rt, int(refine_overlap) or 64)
        else:
            _log(f"refine: whole image {rw}x{rh}, denoise {float(denoise):.2f}, "
                 f"{int(steps)} steps ...")
            _progress(0.5, f"Refine {rw}x{rh}...")
            out = _refine_whole(pipe, img, denoise, steps, prompt, seed)
        timings["refine"] += time.time() - t0
        return out

    result = image
    if refine_first:
        # refine sur l'image native (rapide) puis agrandissement ESRGAN.
        if do_refine:
            result = _refine_stage(result)
        if use_esrgan:
            result = _esrgan_stage(result)
    else:
        # ordre classique: ESRGAN (detailleur) puis refine a la resolution agrandie.
        if use_esrgan:
            result = _esrgan_stage(result)
        if do_refine:
            result = _refine_stage(result)

    if not use_esrgan and not do_refine:
        _log(f"process_one: nothing to do (no ESRGAN, denoise=0) on {w0}x{h0}")

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _progress(1.0, "Done")
    _log(f"process_one done | esrgan {timings['esrgan']:.1f}s + refine {timings['refine']:.1f}s "
         f"= {timings['esrgan'] + timings['refine']:.1f}s")
    return result, timings


@_gpu_serial
def txt2img_run(prompt, width, height, gen_steps, seed, negative_prompt="",
                upscale=False, esrgan_model=None, factor=2.0, denoise=0.30, steps=12,
                tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                refine_first=False):
    """Genere une image (txt2img Z-Image) puis, si upscale=True, la passe dans le
    pipeline ESRGAN + refine. Renvoie (image, timings_dict)."""
    timings = {"txt2img": 0.0, "esrgan": 0.0, "refine": 0.0}
    t0 = time.time()
    base = generate(prompt, width, height, gen_steps, seed, negative_prompt)
    timings["txt2img"] = time.time() - t0
    if not upscale:
        return base, timings
    result, t = process_one(base, esrgan_model, factor, denoise, steps, prompt, seed,
                            tile, overlap, refine_tile=refine_tile, refine_overlap=refine_overlap,
                            refine_first=refine_first)
    timings["esrgan"] = t.get("esrgan", 0.0)
    timings["refine"] = t.get("refine", 0.0)
    return result, timings


def _gen_meta(mode, prompt, negative="", seed=None, steps=None, guidance=None,
              size=None, model=None, styles=None, extra=None):
    """Construit le dict de metadonnees de generation (pour sidecar/PNG)."""
    m = {"app": "crispz-krea2", "mode": mode, "prompt": prompt or "",
         "negative": negative or "", "date": _now_stamp()}
    if seed is not None and int(seed) >= 0:
        m["seed"] = int(seed)
    if steps is not None:
        m["steps"] = int(steps)
    if guidance is not None:
        m["guidance"] = float(guidance)
    if size:
        m["size"] = f"{size[0]}x{size[1]}"
    # Noms de styles appliques (en plus des mots-cles deja injectes dans le prompt).
    _styles = [s for s in (styles or []) if s and s not in ("None", "none")]
    if _styles:
        m["styles"] = _styles
    m["sampler"] = f"{SAMPLER}/{SCHEDULE}"
    m["model"] = model or (ZIMAGE_TRANSFORMER or BASE_REPO)
    if LORAS:
        m["loras"] = [f"{os.path.basename(p)}@{w}" for p, w in LORAS]
    if extra:
        m.update(extra)
    return m
