"""crispz-krea2 - the Krea 2 core (diffusers, BF16 + fp8 quantization): loading the
txt2img pipeline, LoRAs / base model, generation, + the mutable runtime state.

Fork of crispz-qwen-edit. Krea 2 shares Qwen-Image's VAE (AutoencoderKLQwenImage) and text
encoder family (Qwen3-VL), hence the choice of that base.

  - base txt2img -> Krea2Pipeline (Krea2Transformer2DModel, 12.9B)

WHAT KREA 2 CANNOT DO (diffusers exposes NONE of these pipelines):
  - img2img  -> no Krea2Img2ImgPipeline  => refine / upscale-refine unavailable
  - inpaint  -> no Krea2InpaintPipeline  => inpaint / outpaint / reframe unavailable
  - instruction-based editing (the Omni tab)  => unavailable
Those capabilities are declared in CAPABILITIES below; cz_ui uses it to HIDE the
corresponding tabs/controls, and get_pipe() raises UnsupportedFeature as a safety net
(CLI, API, direct calls).

LOADING: from_pretrained on a diffusers repo/folder. Krea2Transformer2DModel has no
upstream from_single_file -> the Civitai .safetensors (bf16 as well as FP8/INT8 'scaled',
ConvRot included) go through OUR conversion to a diffusers folder, in a disk cache (see
_converted_folder), and the ComfyUI-GGUF .gguf files (arch 'krea2') follow the same path,
dequantized to bf16 by the gguf lib. See the README.
bf16 alone overflows a 32 GB card (35.6 GB at the peak, ~59 s/step) -> so it is quantized
on the fly through torchao (fp8 weight-only): 22.8 GB, ~2.4 s/step, equivalent quality.

CFG: the Krea 2 convention -> `guidance_scale` DIRECT (velocity = cond + g*(cond-uncond),
guidance active as soon as g > 0). This is NOT the Qwen convention (true_cfg_scale + a
distilled guidance_scale at 1.0). Turbo is distilled -> g = 0.0 and 8 steps.

The module's public API stays identical to upstream's (the same names, e.g.
ZIMAGE_TRANSFORMER, SAMPLER_CHOICES) so that neither cz_ui nor cz_cli breaks.

app reads the current state through cz_pipeline.NAME (BASE_REPO, ZIMAGE_TRANSFORMER, ...)
and sets cz_pipeline._PROGRESS / cz_pipeline._STOP from the UI handlers.
Depends only on cz_core / cz_esrgan / cz_imageio (never on app or gradio).

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

# The base Krea 2 model (txt2img). Turbo by default: distilled, 8 steps, guidance 0.
# Raw = the non-distilled mid-training checkpoint (28-52 steps + CFG) -> far slower, mostly
# meant for fine-tuning / LoRA training. GATED repos: the licence has to be accepted on
# huggingface.co WITH THE TOKEN'S ACCOUNT (see the README). Overridable through env
# ZIMAGE_MODEL (compat) or KREA_MODEL, or prefs.
DEFAULT_BASE_REPO = (os.environ.get("KREA_MODEL") or "krea/Krea-2-Turbo")
# No instruction-based edit model on Krea 2 (kept for API compat).
DEFAULT_OMNI_REPO = None

# ----------------------------------------------------------------------------
# Capabilities of THIS model family. cz_ui reads this dict to hide the tabs and controls
# with no pipeline behind them -> a single source of truth, no hardcoded list of hidden
# tabs inside the UI.
# ----------------------------------------------------------------------------
CAPABILITIES = {
    "txt2img": True,
    "img2img": False,   # no Krea2Img2ImgPipeline  -> refine, upscale-refine, harmonize
    "inpaint": False,   # no Krea2InpaintPipeline  -> inpaint, outpaint, reframe(contain)
    "omni": False,      # no instruction-based editing
    "lora": True,       # Krea2Transformer2DModel herite de PeftAdapterMixin
    "single_file": True,    # not through diffusers (no FromOriginalModelMixin) but through OUR
                            # conversion: a Civitai .safetensors (bf16 / FP8 / INT8 'scaled')
                            # and a ComfyUI-GGUF .gguf are listed and loaded, converted once
                            # to a diffusers folder (_converted_folder). SVDQuant/NVFP4 stay
                            # refused, with the reason.
    "esrgan": True,     # a pure ESRGAN upscale: independent of the diffusion model
}


class UnsupportedFeature(RuntimeError):
    """A feature this model family does not have. Raised by get_pipe() as a safety net:
    the UI already hides the controls concerned through CAPABILITIES."""


def supports(feature):
    """True when the current family exposes `feature` (a CAPABILITIES key)."""
    return bool(CAPABILITIES.get(feature, False))
from cz_esrgan import load_esrgan, esrgan_upscale
from cz_imageio import _now_stamp
import cz_hw

# Speed: allow TF32 (matmul/cudnn) on the GPU. A free win on Ampere+ for the residual
# fp32 operations; the weights stay BF16. No effect outside CUDA.
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# The current Z-Image model. An HF repo / diffusers folder -> BASE_REPO. A single-file
# checkpoint (a Civitai .safetensors) passed as the "model" -> a transformer override (the
# VAE and the Qwen3 encoder still come from the base repo).
_zmodel = os.environ.get("ZIMAGE_MODEL") or _prefs.get("zimage_model") or DEFAULT_BASE_REPO
ZIMAGE_TRANSFORMER = os.environ.get("ZIMAGE_TRANSFORMER") or _prefs.get("zimage_transformer") or None
if _is_single_file(_zmodel):
    ZIMAGE_TRANSFORMER = _zmodel
    BASE_REPO = DEFAULT_BASE_REPO
else:
    BASE_REPO = _zmodel

# Replacement text encoder (Models > Checkpoints > Text encoder). Empty = the one from
# the base repo, as before. Otherwise a FOLDER in transformers format (config.json +
# weights) or an HF repo ('owner/repo', 'owner/repo/subfolder') -- e.g. an "abliterated"
# Qwen3-VL-4B. Only the encoder changes: tokenizer, VAE and transformer stay those of the
# base repo, and the torchao quantization (QUANT_MODE) still touches the transformer only.
CFG_TEXT_ENCODER_KEY = "text_encoder"


def _resolve_text_encoder(env, prefs, config):
    """The encoder at startup: env > preferences > config. A key PRESENT in the
    preferences wins even when empty: that is the "Default" choice made in the UI, and a
    config.txt value must not undo it on the next start (a "" used to pass for absent)."""
    v = str(env.get("KREA2_TEXT_ENCODER") or "").strip()
    if v:
        return v
    if CFG_TEXT_ENCODER_KEY in prefs:
        return str(prefs.get(CFG_TEXT_ENCODER_KEY) or "").strip()
    return str(config.get(CFG_TEXT_ENCODER_KEY) or "").strip()


TEXT_ENCODER = _resolve_text_encoder(os.environ, _prefs, CONFIG)
# The one REALLY loaded ('' = the base repo's). Distinct from TEXT_ENCODER: an encoder
# that does not suit the current repo is dropped at load time, and the metadata says what
# ran, not what was asked for.
_TEXT_ENCODER_ACTIVE = ""
TEXT_ENCODERS_DIR = str(os.environ.get("TEXT_ENCODERS_DIR") or _prefs.get("text_encoders_dir")
                        or CONFIG.get("text_encoders_dir") or "").strip()

# Z-Image model folders: single-file checkpoints to switch between + LoRAs to apply.
CHECKPOINTS_DIR = (os.environ.get("CHECKPOINTS_DIR") or _prefs.get("checkpoints_dir")
                   or CONFIG.get("checkpoints_dir") or os.path.join(HERE, "checkpoints"))
# Additional checkpoints folder (optional) -> merged with CHECKPOINTS_DIR into the same
# checkpoint list. Empty by default; configurable through UI / prefs / config / env.
CHECKPOINTS_EXTRA_DIR = (os.environ.get("CHECKPOINTS_EXTRA_DIR") or _prefs.get("checkpoints_extra_dir")
                         or CONFIG.get("checkpoints_extra_dir") or "").strip()
LORAS_DIR = (os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
             or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))
# Active LoRAs: a list of (path, weight). Several LoRAs can be combined (multi-slot).
LORAS = []
LORA_WEIGHT = float(CONFIG.get("default_lora_weight", 1.0))  # the slots' default weight


def _lora_weight_range():
    """Bounds of the LoRA weight sliders (config 'lora_weight_min'/'lora_weight_max').
    Default -2..2: NEGATIVE weights are valid and useful (they invert the LoRA's
    effect). Defensive: unreadable values or min >= max -> fall back to the default."""
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
# The default weight has to stay inside the bounds (or the slider would be born out of range).
LORA_WEIGHT = min(LORA_WEIGHT_MAX, max(LORA_WEIGHT_MIN, LORA_WEIGHT))
# LoRAs applied AT STARTUP (e.g. Lightning 8-step). config 'default_loras' = a list of
# names (inside LORAS_DIR) or of [name, weight] pairs. Resolved to (path, weight).
for _spec in (CONFIG.get("default_loras") or []):
    _nm, _w = (_spec if isinstance(_spec, (list, tuple)) and len(_spec) == 2
               else (_spec, LORA_WEIGHT))
    if _nm and _nm not in ("None", "none"):
        _p = _nm if os.path.isabs(_nm) else os.path.join(LORAS_DIR, _nm)
        if os.path.isfile(_p):
            LORAS.append((_p, float(_w)))
# Omni/Edit model: none on Krea 2 (no Qwen-Image-Edit equivalent). It stays empty ->
# the corresponding tab does not appear (see omni_on in cz_ui). Kept as a variable because
# cz_ui / cz_cli read it.
OMNI_MODEL = (os.environ.get("ZIMAGE_OMNI_MODEL") or CONFIG.get("zimage_omni_model")
              or DEFAULT_OMNI_REPO or "").strip()

# Process-wide caches. A "base" pipeline (txt2img ZImagePipeline) owns the
# components; img2img / inpaint derive from it through from_pipe -> shared weights, no
# duplicate VRAM. Cache key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE, LORAS).
_BASE_PIPE = None
_DERIVED = {}
_LOADED_KEY = None
# LoRAs actually applied on _BASE_PIPE (a list of (path, weight)). Used to hot-swap the
# LoRAs without reloading the model: if it diverges from LORAS, _apply_loras resyncs.
_APPLIED_LORAS = []

# Step 2 (VRAM coexistence): CPU offload of the diffusion pass. none = everything in
# VRAM (the fastest). model = unloads per submodule (a good compromise). sequential = more
# aggressive, slower. This is NOT quantization: the weights stay BF16, they travel
# RAM <-> GPU. 'auto' (the default) = a free-VRAM test at load time (cz_hw): a model that
# overflows the VRAM does not crash, it spills into shared RAM (Windows Sysmem Fallback) and
# renders 50-100x slower WITH no error message -> 'none' is only promoted once the card has
# proven it has the room. Resolution order (the first one set wins):
# explicit UI/CLI choice > env CZ_OFFLOAD > config default_cpu_offload > auto.
OFFLOAD_CHOICES = ("auto", "none", "model", "sequential")
OFFLOAD_MODE = ((os.environ.get("CZ_OFFLOAD") or "").strip()
                or str(CONFIG.get("default_cpu_offload", "") or "").strip()).lower() or "auto"
if OFFLOAD_MODE not in OFFLOAD_CHOICES:
    _log(f"CZ_OFFLOAD/default_cpu_offload '{OFFLOAD_MODE}' unknown -> auto")
    OFFLOAD_MODE = "auto"
# The concrete mode resolved for 'auto' (set by _resolve_auto on the first load) and the
# flag of the runtime safety net (set by the VRAM callback during the denoise).
_AUTO_OFFLOAD = ""
_VRAM_DOWNGRADE = False

# Krea 2 guidance. A DIFFERENT convention from Qwen's: the UI slider drives
# `guidance_scale` directly, the velocity being cond + g*(cond-uncond); guidance is active as
# soon as g > 0 (equivalent to the usual CFG of scale 1+g). Turbo is DISTILLED -> 0.0 (no
# CFG, a single forward per step). Raw (non-distilled) wants ~4.5. So 0.0 IS a valid value
# here, hence no "or 4.0" fallback. Overridable through env KREA_CFG.
_g = os.environ.get("KREA_CFG")
if _g is None:
    _g = CONFIG.get("default_guidance")
GUIDANCE = float(_g) if _g not in (None, "") else 0.0

# Quantization of the transformer at load time (torchao). Krea 2 in bf16 weighs 26 GB
# and OVERFLOWS a 32 GB card (a 35.6 GB peak -> a spill into RAM over PCIe, ~59 s/step). In
# fp8 weight-only: 22.8 GB and ~2.4 s/step, for visually equivalent quality (the activations
# stay bf16). Measured on an RTX 5090 / 1024x1024 / 8 steps.
#   "float8_weight_only" (the default, native on sm_89+/Blackwell) | "int8_weight_only"
#   | "int4_weight_only" | "float8_dynamic" | "none" (raw bf16, needs >32 GB of VRAM)
QUANT_MODE = (os.environ.get("KREA_QUANT") or CONFIG.get("quantization")
              or "float8_weight_only").strip().lower()
QUANT_CHOICES = ("float8_weight_only", "int8_weight_only", "int4_weight_only",
                 "float8_dynamic", "none")
if QUANT_MODE not in QUANT_CHOICES:
    QUANT_MODE = "float8_weight_only"


def set_quant_mode(mode):
    """Changes the quantization scheme. Invalidates the pipe (a reload on the next run)."""
    global QUANT_MODE
    mode = (mode or "none").strip().lower()
    if mode not in QUANT_CHOICES or mode == QUANT_MODE:
        return
    # Both writes under the GPU lock, together. free_vram() takes it on its own and WAITS
    # for the render in progress (see _gpu_exclusive), but QUANT_MODE would already have
    # changed: a render grabbing the lock before the free would reuse the pipe built with
    # the OLD scheme while QUANT_MODE announces the new one -- one render quantised
    # differently from what the UI says. An explicit `with` rather than the decorator
    # because this function is defined before the lock is (a `with` resolves at call time).
    # The "waiting for the render" line is therefore not printed here; this dropdown only
    # changes exceptionally.
    with _GPU_LOCK:
        QUANT_MODE = mode
        free_vram()
    _log(f"quantization -> {QUANT_MODE} (reload on next run)")


def _quant_config():
    """A TorchAoConfig for QUANT_MODE, or None when disabled/unavailable.
    torchao >= 0.17 requires an AOBaseConfig object (strings are no longer accepted)."""
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

# Forced ratio (Fooocus-style) for upscale/img2img: when set, the INPUT image is
# center-cropped to that ratio before processing (crop to fit). Empty = the native ratio is
# preserved (the default). Format: 'W:H' or 'WxH' (e.g. '13:19', '832x1216'). Driven by the
# UI (checkbox + Aspect ratio dropdown) through set_force_ratio, or by config.txt
# 'force_upscale_ratio'.
FORCE_RATIO = (os.environ.get("CZ_FORCE_RATIO") or CONFIG.get("force_upscale_ratio") or "").strip()
# How to reach the forced ratio: 'crop' = a center crop (loses the edges, the
# default), 'extend' = outpaints the missing bands. Krea 2 note: no inpaint pipeline ->
# 'extend' raises the clear UnsupportedFeature message (the UI only offers Off/Crop).
FORCE_RATIO_MODE = (os.environ.get("CZ_FORCE_RATIO_MODE")
                    or CONFIG.get("force_ratio_mode") or "crop").strip().lower()
# Seam-blending pass of the extend mode (moot as long as Krea 2 has no inpaint, kept
# for config parity with the family). 0 = off.
try:
    EXTEND_DENOISE = float(CONFIG.get("force_ratio_extend_denoise", 0.22) or 0.0)
except Exception:
    EXTEND_DENOISE = 0.22

# Sampler / scheduler. The Z-Image pipeline imposes a custom `sigmas` schedule: only
# the schedulers whose set_timesteps accepts `sigmas` work. In practice -> Euler
# flow-matching (native, the default), UniPC (multistep) and LCM flow-matching (interesting
# on distilled/Turbo models: few steps, guidance ~0-1).
# diffusers' DPM++ 2M / DPM2a / DPM++ SDE (dpmpp_sde) do NOT take custom sigmas ->
# incompatible (DPMSolverSDEScheduler also requires torchsde). Not exposed.
SAMPLER_CHOICES = ("euler", "unipc", "lcm")
SAMPLER = (os.environ.get("ZIMAGE_SAMPLER") or CONFIG.get("default_sampler") or "euler").strip().lower()
if SAMPLER not in SAMPLER_CHOICES:
    SAMPLER = "euler"

# Sigma schedule (= the "scheduler" in ComfyUI terms). sgm_uniform = Z-Image's native
# one (linspace + dynamic shift). beta/karras/exponential = a sigma remapping applied ON TOP
# of the pipeline's schedule (FlowMatchEuler/UniPC: use_*_sigmas). beta -> scipy.
SCHEDULE_CHOICES = ("sgm_uniform", "beta", "karras", "exponential")
# 'simple' (ComfyUI) names EXACTLY the native schedule exposed here as 'sgm_uniform': the
# default sigmas the pipeline hands the scheduler are linspace(1, 1/n, n), which is what
# ComfyUI calls 'simple' on a flow-matching model. Accepted as input everywhere
# (config/env/CLI/XYZ) so a CivitAI recipe can be copied word for word, but normalised to
# the canonical name: metadata and presets only ever carry one name.
_SCHEDULE_ALIASES = {"simple": "sgm_uniform"}
SCHEDULE_INPUTS = SCHEDULE_CHOICES + tuple(_SCHEDULE_ALIASES)   # listes ouvertes (CLI/XYZ)


def _norm_schedule(name, default="sgm_uniform"):
    """Nom de schedule -> nom canonique (alias resolus). Inconnu -> `default`."""
    n = (name or "").strip().lower()
    n = _SCHEDULE_ALIASES.get(n, n)
    return n if n in SCHEDULE_CHOICES else default


SCHEDULE = _norm_schedule(os.environ.get("ZIMAGE_SCHEDULE") or CONFIG.get("default_schedule"))
_SCHEDULE_FLAG = {"beta": "use_beta_sigmas", "karras": "use_karras_sigmas",
                  "exponential": "use_exponential_sigmas"}  # sgm_uniform -> no flag (native)
# The model's own scheduler config (captured on the first load) -> the base every other
# sampler is built from (keeps shift/flow params whatever the current sampler is).
_BASE_SCHED_CONFIG = None

# UI progress hook (gradio gr.Progress). None outside the UI (CLI/server). Set by
# the handlers through cz_pipeline._PROGRESS = ...
_PROGRESS = None
# Fooocus-style Stop: a global flag plus the interruption of the diffusers pipelines. Set
# by the handlers through cz_pipeline._STOP = ... and by request_stop().
_STOP = False

# GPU lock: serialises EVERY generation. Gradio does not serialise the events of
# different LISTENERS (manual Generate vs Run queue vs the detailer): two threads can then
# call the SAME shared pipeline and step the SAME scheduler -> its index runs past the end
# ("IndexError: index 31 is out of bounds for dimension 0 with size 31",
# scheduling_flow_match_euler_discrete.step). RLock: one thread's nested calls
# (txt2img_run -> generate, process_one -> _refine_whole) stay free.
_GPU_LOCK = threading.RLock()


def _gpu_serial(fn):
    """Decorator: runs fn under _GPU_LOCK (a single GPU generation at a time)."""
    import functools

    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        with _GPU_LOCK:
            return fn(*args, **kwargs)
    return _locked


def _gpu_exclusive(fn):
    """Decorator for the setters that RELEASE the shared pipeline (free_vram and the three
    that call it). Doing that under a running denoise loop pulls the weights out from under
    it; the scheduler race showed what touching the shared pipe mid-render costs.

    Unlike the sampler, these cannot be DEFERRED: you pressed Free VRAM, or changed the
    encoder, to have it happen -- so they WAIT. The wait is announced, because a handler
    blocked for a whole render looks frozen otherwise. The try-acquire first keeps the
    common case silent.

    RLock -> a call from INSIDE a generation goes straight through: retry_on_oom and
    _consume_vram_downgrade both free the VRAM on the generation's own thread, and that
    thread is between two pipeline calls, not inside one.

    NOT applied to set_loras() nor set_zimage_transformer(): checked, they only write a
    global that the next _ensure_base reads under the lock, so a running render is not
    affected. A PAIR of setters is still two operations, though -- nothing makes
    set_zimage_transformer('') + set_zimage_model(x) atomic together.
    """
    import functools

    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        if not _GPU_LOCK.acquire(blocking=False):
            _log(f"{fn.__name__}: waiting for the render in progress (releasing the "
                 f"shared pipeline now would break it) ...")
            _GPU_LOCK.acquire()
        try:
            return fn(*args, **kwargs)
        finally:
            _GPU_LOCK.release()
    return _locked

# Seed handling (Fooocus-style):
#  _LAST_SEED         = the CONCRETE seed of the last render (a random -1 is resolved to a
#                       real value) -> the "Reuse last seed" button + honest metadata.
#  _NO_SEED_INCREMENT = True -> a whole batch uses the same seed (no +i per image).
_LAST_SEED = -1
_NO_SEED_INCREMENT = False
# True -> in txt2img+upscale, ALSO save the original txt2img image (before the upscale).
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
    """CFG kwargs for Krea 2. The model's convention: `guidance_scale` DIRECT, the
    velocity being cond + g*(cond-uncond), guidance active as soon as g > 0. No
    `true_cfg_scale` (that one was Qwen). At g = 0 (distilled Turbo) guidance is off and the
    pipeline ignores the negative prompt -> so it is not sent, which avoids a pointless
    forward."""
    g = float(GUIDANCE)
    kw = {"guidance_scale": g}
    if g > 0:
        kw["negative_prompt"] = (negative or None)
    return kw



def guidance_from_standard_cfg(cfg):
    """The Krea 2 guidance matching a STANDARD CFG (ComfyUI, A1111, CivitAI).

    A standard CFG is uncond + cfg*(cond-uncond): 1.0 = no guidance. Krea 2 writes
    cond + g*(cond-uncond) = uncond + (1+g)*(cond-uncond), so g = cfg - 1, and 0.0 cuts the
    guidance off (Krea2Pipeline's docstring says it: "scale 1 + guidance_scale"). Copying the
    cfg as is -- which is what the 'Apply CivitAI recommended settings' button did -- turned a
    community 'cfg 1' (no guidance) into g = 1.0: CFG ON, standard scale 2, two passes per
    step on a distilled Turbo. The opposite of what the community was using. None when
    unreadable.
"""
    try:
        return max(0.0, round(float(cfg) - 1.0, 2))
    except (TypeError, ValueError):
        return None

# --- Prompt embedding cache -------------------------------------------------------
# Encoding a prompt sends the text encoder through the GPU. Under 'model' offload that
# transfer is paid on EVERY pipeline call -- including the detailer's passes, which redo
# the SAME prompt once per face and per hand.
# Measured on crispz-klein (9B GGUF, model offload): prompt+setup 5.2-6.1 s per pass
# without the cache against 1.7-1.8 s with it, for 0.3 s of diffusion. 2.1x per hand.
# Without offload the win drops to ~8 % (the encoder is already resident, nothing to move).
#
# encode_prompt() short-circuits the encoder as soon as it is handed its embeddings. So the
# TUPLE it returns is memorised and handed back to __call__ through _EMBED_OUTS.
# The tensors are kept in RAM (a few MB): they hold no VRAM and survive the offload's moves.
# Krea2: encode_prompt -> (prompt_embeds, prompt_embeds_mask), like Qwen-Image.
_EMBED_OUTS = ("prompt_embeds", "prompt_embeds_mask")
_EMBED_CACHE = {}
_EMBED_CACHE_MAX = max(0, int(CONFIG.get("prompt_embed_cache", 8) or 0))


def _embed_cache_clear(why=""):
    """Empties the cache. Called as soon as the encoder may have changed (base repo, VRAM
    release): an embedding computed by another encoder is wrong."""
    if _EMBED_CACHE:
        _dbg(f"prompt embed cache cleared ({len(_EMBED_CACHE)} entries){why}")
    _EMBED_CACHE.clear()


def _cached_prompt_embeds(pipe, prompt, kw):
    """Embeddings of `prompt` for this pipeline, computed once then reused.

    Returns a kwargs dict for __call__, or None when the cache is off, when the
    pipeline does not expose the expected API, or when the encoding fails: in all
    those cases the caller passes the prompt as text and nothing changes. A cache must
    never break a render.
"""
    if not _EMBED_CACHE_MAX or not _EMBED_OUTS:
        return None
    try:
        enc = getattr(pipe, "text_encoder", None)
        if enc is None or not hasattr(pipe, "encode_prompt"):
            return None
        # The LoRAs are part of the key: some of them touch the text encoder, and an
        # embedding computed without them would be wrong.
        # So is the replacement encoder: id(enc) alone is not enough, CPython recycles
        # the id of a freed object -- and another encoder encodes differently.
        key = (BASE_REPO, _TEXT_ENCODER_ACTIVE, id(enc), prompt,
               kw.get("max_sequence_length"),
               tuple(sorted((p, float(w)) for p, w in _APPLIED_LORAS)))
        hit = _EMBED_CACHE.get(key)
        if hit is None:
            out = pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)
            if not isinstance(out, (tuple, list)):
                out = (out,)
            hit = tuple(v.detach().to("cpu") if hasattr(v, "detach") else v
                        for v in out[:len(_EMBED_OUTS)])
            if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
                _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))      # FIFO, borne simple
            _EMBED_CACHE[key] = hit
            _dbg(f"prompt embeds computed and cached ({len(_EMBED_CACHE)}/"
                 f"{_EMBED_CACHE_MAX}) for {prompt[:40]!r}")
        else:
            _dbg(f"prompt embeds reused (text encoder not touched) for {prompt[:40]!r}")
        dev = pipe._execution_device
        return {name: (v.to(dev) if hasattr(v, "to") else v)
                for name, v in zip(_EMBED_OUTS, hit) if name}
    except Exception as e:
        _dbg(f"prompt embed cache off for this call ({type(e).__name__}: {e})")
        return None


def _qwen_call(pipe, **kw):
    """Calls the pipeline, tolerating diffusers API variations: when the installed version
    does not know a guidance kwarg, it is dropped and the call is retried rather than crashing
    the generation. (The name is kept so as not to break the fork's internal API.)"""
    # Reuse the embeddings if this prompt has already been encoded (see _EMBED_CACHE).
    # Passing them skips the text encoder: that is the whole win.
    if isinstance(kw.get("prompt"), str) and not any(k in kw for k in _EMBED_OUTS):
        _emb = _cached_prompt_embeds(pipe, kw["prompt"], kw)
        if _emb:
            kw.update(_emb)
            kw["prompt"] = None

    try:
        return pipe(**kw)
    except TypeError as e:
        # callback_on_step_end = the optional VRAM guard (see _vram_guard_kwargs):
        # an old diffusers build that does not know it runs without the guard.
        if any(k in kw for k in ("negative_prompt", "callback_on_step_end")):
            for k in ("negative_prompt", "callback_on_step_end"):
                kw.pop(k, None)
            _dbg(f"krea2 call: retrying without the optional kwargs ({e})")
            return pipe(**kw)
        raise


# An explicit alias: the rest of the fork calls _qwen_call, but the name now lies.
_krea_call = _qwen_call


def _scheduler_accepts_sigmas(sched):
    """The Z-Image pipeline calls set_timesteps(..., sigmas=<custom schedule>). A scheduler
    whose set_timesteps does not accept `sigmas` crashes at generation time."""
    import inspect
    try:
        return "sigmas" in inspect.signature(sched.set_timesteps).parameters
    except Exception:
        return False


def _build_scheduler(sampler, schedule, config):
    """Builds the chosen scheduler (sampler x schedule) from the model's native config.
    schedule (sgm_uniform/beta/karras/exponential) = a sigma remapping (use_*_sigmas)."""
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
        # LCM flow-matching: takes the pipeline's custom sigmas AND the schedule flags.
        # Falls back to Euler when the installed diffusers does not expose it.
        try:
            from diffusers import FlowMatchLCMScheduler
            return FlowMatchLCMScheduler.from_config(config, **kw)
        except Exception as e:
            _log(f"sampler 'lcm' unavailable ({e}); falling back to euler")
    return FlowMatchEulerDiscreteScheduler.from_config(config, **kw)


def _apply_sampler(pipe):
    """Applies the current scheduler (SAMPLER x SCHEDULE) to a pipe. Checks compatibility
    (custom sigmas) and falls back to Euler/sgm_uniform when it fails -> never a crash at
    generation time."""
    if _BASE_SCHED_CONFIG is None:
        return
    from diffusers import FlowMatchEulerDiscreteScheduler
    try:
        sched = _build_scheduler(SAMPLER, SCHEDULE, _BASE_SCHED_CONFIG)
        if not _scheduler_accepts_sigmas(sched):
            raise ValueError(f"{type(sched).__name__} does not accept the custom sigmas of Z-Image")
        pipe.scheduler = sched
        _dbg(f"sampler applied: {SAMPLER}/{SCHEDULE} -> {type(pipe.scheduler).__name__}")
    except Exception as e:
        _log(f"sampler '{SAMPLER}/{SCHEDULE}' incompatible ({e}); fallback Euler/sgm_uniform")
        try:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(_BASE_SCHED_CONFIG)
        except Exception:
            pass


# Changing the scheduler is NOT a local change: it lives on the SHARED pipe. A denoise
# loop already running keeps its own `timesteps` list but steps whatever `pipe.scheduler`
# points at by then. A fresh scheduler knows nothing of those timesteps and has no
# begin_index, so diffusers looks the current timestep up and finds nothing:
#   IndexError: index 0 is out of bounds for dimension 0 with size 0
#   (scheduling_flow_match_euler_discrete._init_step_index -> index_for_timestep)
# Met on crispz-krea2 on 2026-09-28: checkpoint switched, then "Apply CivitAI recommended
# settings" (which sets the sampler AND the schedule), then Generate. Gradio does not
# serialise the events of DIFFERENT listeners -- the very hole _GPU_LOCK exists for, except
# that these two setters were never put under it.
# So the swap never lands under a running generation: free lock -> applied at once; held
# lock -> only recorded, and the next get_pipe() applies it. Every generation path goes
# through get_pipe() with the lock held.
_SAMPLER_DIRTY = False


def _reapply_sampler_all():
    """Re-applies the current scheduler to every cached pipe (base + derived). Returns
    False when a generation holds the GPU: the change is recorded, not applied."""
    global _SAMPLER_DIRTY
    # A try-acquire, not a wait: blocking here would freeze the dropdown handler for the
    # whole render (up to a 30-image batch). RLock -> a call from a thread that ALREADY
    # holds the lock (the job queue restoring a snapshot between two jobs) goes through and
    # applies at once, which is correct: that thread is between two generations.
    if not _GPU_LOCK.acquire(blocking=False):
        _SAMPLER_DIRTY = True
        _log(f"sampler/schedule {SAMPLER}/{SCHEDULE}: applied on the NEXT run "
             f"(a generation is running; swapping it now would crash that render)")
        return False
    try:
        _SAMPLER_DIRTY = False
        for p in [_BASE_PIPE] + list(_DERIVED.values()):
            if p is not None:
                _apply_sampler(p)
    finally:
        _GPU_LOCK.release()
    return True


def _apply_sampler_if_dirty():
    """Applies a sampler/schedule change that arrived while a generation was running.
    Called by get_pipe(), i.e. by every generation path, with _GPU_LOCK held."""
    global _SAMPLER_DIRTY
    if not _SAMPLER_DIRTY:
        return
    _SAMPLER_DIRTY = False
    _dbg(f"applying the deferred sampler/schedule {SAMPLER}/{SCHEDULE}")
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            _apply_sampler(p)


def _sampler_status():
    """The label shown next to the two dropdowns. Says so when the change is only
    recorded: telling the user it is active while the render still uses the old one is
    exactly the confusion to avoid."""
    return (f"Sampler: {SAMPLER} / {SCHEDULE}"
            + (" — on the next run" if _SAMPLER_DIRTY else ""))


def set_sampler(name):
    """Changes the sampler (euler/unipc) and re-applies it to the cached pipes (no
    reload). No effect on the Omni pipe (its own scheduler)."""
    global SAMPLER
    name = (name or "euler").strip().lower()
    if name not in SAMPLER_CHOICES:
        name = "euler"
    if name != SAMPLER:
        SAMPLER = name
        _log(f"sampler -> {SAMPLER}")
        _reapply_sampler_all()
    return _sampler_status()


def set_schedule(name):
    """Changes the sigma schedule (sgm_uniform/beta/karras/exponential, alias 'simple'
    = sgm_uniform) and re-applies it to the cached pipes."""
    global SCHEDULE
    name = _norm_schedule(name)
    if name != SCHEDULE:
        SCHEDULE = name
        _log(f"schedule -> {SCHEDULE}")
        _reapply_sampler_all()
    return _sampler_status()


def _progress(frac, desc=""):
    if _PROGRESS is not None:
        try:
            _PROGRESS(min(1.0, max(0.0, float(frac))), desc)
        except Exception:
            pass


# ---- Model loading feedback (terminal + UI) ----
# from_pretrained is blocking and silent (the first load downloads from HF -> several
# minutes). So the load runs in a thread and every ~2s a terminal line plus the Gradio bar
# are refreshed (elapsed time + allocated VRAM). Config block "load_progress";
# enabled=false -> a direct load (no thread, zero cost).
_LOAD_CFG = CONFIG.get("load_progress") if isinstance(CONFIG.get("load_progress"), dict) else {}
LOAD_PROGRESS_ENABLED = bool(_LOAD_CFG.get("enabled", True))
_LOAD_TARGET_GB = float(_LOAD_CFG.get("target_vram_gb", 14.0))
_LOAD_HEARTBEAT = float(_LOAD_CFG.get("heartbeat_s", 2.0))


def _fmt_load(label, elapsed, vram_gb):
    """Loading progress text (pure, testable). VRAM > 0 -> the loading-into-memory
    phase; otherwise the download/disk-read phase."""
    if vram_gb > 0.05:
        return f"{label}... {elapsed:.0f}s | {vram_gb:.1f} GB in VRAM"
    return f"{label}... {elapsed:.0f}s (downloading / reading, first run only)"


def _load_pct(elapsed, vram_gb, target_gb=None):
    """An honest %: based on the allocated VRAM / target once the load into memory has
    started (capped at 0.95); during the download (VRAM~0) a small time-based bar."""
    target_gb = target_gb or _LOAD_TARGET_GB
    if vram_gb <= 0.05:
        return min(0.12, elapsed / 600.0)
    return min(0.95, vram_gb / max(1.0, float(target_gb)))


def _load_monitor(label, fn):
    """Runs fn() (a blocking load) in a thread and refreshes terminal + UI (time +
    VRAM) every ~2s. Returns fn's result (re-raises its exception)."""
    if not LOAD_PROGRESS_ENABLED:
        return fn()
    box = {}

    def _work():
        try:
            box["v"] = fn()
        except BaseException as e:   # noqa: BLE001 - it is re-raised in the main thread
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
    """Asks for a stop: halts the running denoise loop (pipe._interrupt) and the
    batch/tile loops (_STOP). Near-immediate (it stops at the next step)."""
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


@_gpu_exclusive
def set_zimage_model(repo_or_path):
    """Changes the Krea 2 model. An HF repo / diffusers folder -> BASE_REPO.
    A single-file checkpoint (a Civitai .safetensors, a .gguf) -> a transformer override."""
    global BASE_REPO, ZIMAGE_TRANSFORMER
    if not repo_or_path:
        return
    if _is_single_file(repo_or_path):
        # A transformer-only change: NO free_vram -> _ensure_base will swap the
        # transformer alone (VAE + text encoder kept in VRAM).
        if repo_or_path != ZIMAGE_TRANSFORMER:
            ZIMAGE_TRANSFORMER = repo_or_path
            _log("Krea 2 transformer (single-file) changed -> transformer swap on next run")
    elif repo_or_path != BASE_REPO:
        # The base repo changes: the VAE/encoder/tokenizer change too -> a full reload.
        BASE_REPO = repo_or_path
        free_vram()
        _log("Krea 2 base repo changed -> will reload")


def set_zimage_transformer(path):
    """Sets (or removes with '' / None) the single-file transformer.

    Does NOT release the pipeline: with the same base repo, _ensure_base will only reload
    the transformer (_swap_transformer) and keep VAE + text encoder in VRAM.
"""
    global ZIMAGE_TRANSFORMER
    path = path or None
    if path != ZIMAGE_TRANSFORMER:
        ZIMAGE_TRANSFORMER = path
        _log(f"Krea 2 transformer -> {path or '(base repo)'} "
             "-> transformer swap on next run (base components kept)")


# --- Replacement text encoder ---------------------------------------------------------
# Krea2Pipeline.get_text_hidden_states calls the encoder with output_hidden_states=True and
# stacks outputs.hidden_states[i] for the FIXED indices of text_encoder_select_layers
# ((2, 5, ..., 35) in model_index.json, as many as transformer.config.num_text_layers, that
# is 12), each text_hidden_dim (2560) wide. So an encoder only fits if it has the same family
# (qwen3_vl), the same width AND the same number of layers as the base repo's: fewer layers
# and the indices overflow on the first prompt, more and the transformer reads depths other
# than the ones it was trained on, with no error. An "abliterated" or fine-tuned Qwen3-VL-4B
# of the same size plugs in as is. That is checked on the config, BEFORE reading 8 GB.
# Extensions of a single-file checkpoint (klein keeps them in cz_core).
_TE_SINGLE_FILE_EXTS = (".safetensors", ".ckpt", ".pt", ".sft", ".gguf")


def _looks_single_file(p):
    """True when the NAME is that of a single-file checkpoint, whether it exists or not."""
    return bool(p) and str(p).lower().endswith(_TE_SINGLE_FILE_EXTS)


def _split_hf_src(src):
    """'owner/repo/sub/folder' -> ('owner/repo', 'sub/folder'). The weights of an encoder
    published on HF often sit in a subfolder of the repo."""
    parts = [p for p in str(src).replace("\\", "/").split("/") if p]
    if len(parts) > 2:
        return "/".join(parts[:2]), "/".join(parts[2:])
    return str(src), None


def _enc_dims(cfg):
    """(width, layers, family) of a transformers config. The VLs (Qwen3-VL here) keep the
    text part under 'text_config'; T5 says d_model / num_layers."""
    c = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    h = c.get("hidden_size") or c.get("d_model")
    n = c.get("num_hidden_layers") or c.get("num_layers")
    return (int(h) if h else None, int(n) if n else None, cfg.get("model_type"))


def _base_text_encoder_config(base=None):
    """The config.json of the base repo's encoder, or None when unreadable."""
    base = (base or BASE_REPO or "").strip()
    try:
        cfg = os.path.join(base, "text_encoder", "config.json")
        if not os.path.isfile(cfg):
            from huggingface_hub import hf_hub_download
            try:
                cfg = hf_hub_download(base, "text_encoder/config.json", local_files_only=True)
            except Exception:
                cfg = hf_hub_download(base, "text_encoder/config.json")
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _dbg(f"cannot read {base}'s text encoder config: {e}")
        return None


def _text_encoder_source(src):
    """Locates the encoder `src`: (config, folder or repo, subfolder) or None.
    A local folder: config.json at the root or inside text_encoder/. An HF repo: the same,
    or the subfolder named in the id."""
    src = (src or "").strip()
    if not src:
        return None
    if os.path.isdir(src):
        for sub in (None, "text_encoder"):
            p = os.path.join(src, sub, "config.json") if sub else os.path.join(src, "config.json")
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        return json.load(f), src, sub
                except Exception:
                    return None
        return None
    if os.path.exists(src) or _looks_single_file(src) or "\\" in src or os.path.isabs(src):
        return None
    repo, sub0 = _split_hf_src(src)
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    for sub in ([sub0] if sub0 else [None, "text_encoder"]):
        rel = f"{sub}/config.json" if sub else "config.json"
        for local in (True, False):          # the cache first: works offline
            try:
                p = hf_hub_download(repo, rel, local_files_only=local)
                with open(p, encoding="utf-8") as f:
                    return json.load(f), repo, sub
            except Exception:
                continue
    return None


def _encoder_label(src):
    """A readable name for an encoder: the FOLDER's name -- never the path, which would
    end up in shared PNGs along with the Windows session name -- or the HF repo id."""
    src = (src or "").strip()
    if not src:
        return ""
    if os.path.isabs(src) or os.path.exists(src) or "\\" in src:
        parts = [p for p in src.replace("\\", "/").split("/") if p]
        if len(parts) >= 2 and parts[-1] == "text_encoder":
            return parts[-2]
        return parts[-1] if parts else src
    return src


def _text_encoder_problem(src, base=None):
    """Why `src` should be refused as the encoder of repo `base`, or None when it fits."""
    src = (src or "").strip()
    if not src:
        return None
    if src.lower().endswith(".gguf"):
        return ("a GGUF text encoder is a ComfyUI / llama.cpp file; this app loads the "
                "transformers folder (config.json + .safetensors)")
    if os.path.isfile(src) or _looks_single_file(src):
        return ("a single file carries no config.json; point to the FOLDER that holds "
                "config.json and the weights")
    found = _text_encoder_source(src)
    if found is None:
        return ("no config.json found, neither at its root nor in text_encoder/"
                if os.path.isdir(src) else
                "neither a folder on this machine nor a readable Hugging Face repo")
    ref_cfg = _base_text_encoder_config(base)
    if ref_cfg is None:
        return None                      # nothing to compare: the load will decide
    (h, n, t), (rh, rn, rt) = _enc_dims(found[0]), _enc_dims(ref_cfg)
    b = (base or BASE_REPO)
    if t and rt and t != rt:
        return f"a '{t}' model, and {b} uses a '{rt}' text encoder"
    if h and rh and h != rh:
        return (f"hidden size {h}, and {b}'s encoder is {rh} wide: the transformer "
                f"cannot read its embeddings")
    if n and rn and n != rn:
        return (f"{n} layers, and {b}'s encoder has {rn}: Krea 2 reads fixed layers of "
                f"it (text_encoder_select_layers)")
    return None


def _encoder_class(base=None):
    """The encoder's transformers class, read from the base repo's model_index.json
    (Qwen3VLModel here): the same one diffusers would have loaded."""
    base = (base or BASE_REPO or "").strip()
    try:
        p = os.path.join(base, "model_index.json")
        if not os.path.isfile(p):
            from huggingface_hub import hf_hub_download
            try:
                p = hf_hub_download(base, "model_index.json", local_files_only=True)
            except Exception:
                p = hf_hub_download(base, "model_index.json")
        with open(p, encoding="utf-8") as f:
            lib, cls = json.load(f)["text_encoder"]
        import importlib
        return getattr(importlib.import_module(lib), cls)
    except Exception as e:
        raise RuntimeError(f"cannot tell which class {base}'s text encoder uses "
                           f"({type(e).__name__}: {e})") from e


def _load_text_encoder(src, base=None):
    """Loads the encoder `src` in DTYPE, with the base repo's class. No quantization:
    torchao (QUANT_MODE) only touches the transformer, as before."""
    found = _text_encoder_source(src)
    if found is None:
        raise RuntimeError(f"{src}: no config.json")
    _cfg, where, sub = found
    kw = {"torch_dtype": DTYPE}
    if sub:
        kw["subfolder"] = sub
    return _encoder_class(base).from_pretrained(where, **kw)


def list_text_encoders():
    """Encoder folders offered in the Models tab: the subfolders with a config.json inside
    `text_encoders_dir`, or inside text_encoders / text_encoder / clip next to the
    checkpoints folder or its parent (ComfyUI and Forge conventions)."""
    roots = [TEXT_ENCODERS_DIR] if TEXT_ENCODERS_DIR else []
    here = os.path.abspath(CHECKPOINTS_DIR or ".")
    for up in (os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        roots += [os.path.join(up, n) for n in ("text_encoders", "text_encoder", "clip")]
    out = []
    for r in roots:
        try:
            names = sorted(os.listdir(r))
        except OSError:
            continue
        for d in names:
            p = os.path.join(r, d)
            if p in out or not os.path.isdir(p):
                continue
            if (os.path.isfile(os.path.join(p, "config.json"))
                    or os.path.isfile(os.path.join(p, "text_encoder", "config.json"))):
                out.append(p)
    return out



def _hf_cache_dir():
    """The Hugging Face cache folder (follows HF_HUB_CACHE / HF_HOME), or None."""
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except Exception:
        return None


def _scan_cached_encoders():
    """[(HF id, config)] from the Hugging Face cache: repos that are NOT diffusers
    pipelines (no model_index.json), one of whose configs -- at the root or in a subfolder --
    has its weights next to it (the most recent revision)."""
    root = _hf_cache_dir()
    if not root or not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        if not d.startswith("models--"):
            continue
        repo = d[len("models--"):].replace("--", "/", 1)
        snaps = os.path.join(root, d, "snapshots")
        try:
            revs = sorted(os.listdir(snaps),
                          key=lambda r: os.path.getmtime(os.path.join(snaps, r)), reverse=True)
        except OSError:
            continue
        if not revs:
            continue
        snap = os.path.join(snaps, revs[0])
        if os.path.isfile(os.path.join(snap, "model_index.json")):
            continue                                 # a diffusers pipeline, not an encoder
        try:
            subs = [""] + sorted(s for s in os.listdir(snap) if os.path.isdir(os.path.join(snap, s)))
        except OSError:
            continue
        for s in subs:
            p = os.path.join(snap, s) if s else snap
            try:
                with open(os.path.join(p, "config.json"), encoding="utf-8") as f:
                    cfg = json.load(f)
                if not isinstance(cfg, dict):
                    continue
                if not any(fn.endswith(".safetensors") for fn in os.listdir(p)):
                    continue                         # the config alone, the weights are not downloaded
            except Exception:
                continue
            out.append((f"{repo}/{s}" if s else repo, cfg))
    return out


def list_cached_text_encoders(base=None):
    """COMPATIBLE encoders already downloaded in the Hugging Face cache, as (name, HF id).
    An encoder downloaded from HF lives in that cache, not in a text_encoders folder:
    without this sweep the Models tab list did not show it (caught on klein on 2026-09-10).
    Compatible = the same family, width and number of layers as the base repo's encoder. The
    value is the HF id: readable in the metadata."""
    ref_cfg = _base_text_encoder_config(base)
    if not ref_cfg:
        return []
    ref = _enc_dims(ref_cfg)
    return [(hid, hid) for hid, cfg in _scan_cached_encoders() if _enc_dims(cfg) == ref]


def cached_text_encoder_mismatches(base=None):
    """Encoders in the HF cache of the SAME family but of another size than the base
    repo's: hidden from the list (they would be refused), named next to it so one knows why.
    ([(HF id, width)], expected width)."""
    ref_cfg = _base_text_encoder_config(base)
    if not ref_cfg:
        return [], None
    rh, rn, rt = _enc_dims(ref_cfg)
    out = []
    for hid, cfg in _scan_cached_encoders():
        h, n, t = _enc_dims(cfg)
        if t == rt and (h, n) != (rh, rn):
            out.append((hid, h))
    return out, rh


@_gpu_exclusive
def set_text_encoder(src):
    """Picks the text encoder ('' = the base repo's). A change RELEASES the pipeline --
    the encoder loads with it, no hot swap under the offload hooks -- and free_vram empties
    the embedding cache, which the previous one computed."""
    global TEXT_ENCODER
    src = (src or "").strip()
    if src == TEXT_ENCODER:
        return
    TEXT_ENCODER = src
    free_vram()
    _log(f"text encoder -> {_encoder_label(src) or '(base repo)'} -> full reload on next run")


# ----------------------------------------------------------------------------
# LyCORIS (LoKr / LoHa). The update there is factorised as a Kronecker product (LoKr) or a
# Hadamard one (LoHa). It is NOT a LoRA and diffusers converts neither -- not one occurrence
# of 'lokr' in loaders/lora_conversion_utils.
# Without detection, a LoKr slips past EVERY guard: its ai-toolkit keys are called
# 'diffusion_model.<module>.lokr_w1', neither '.lora_A/B' nor 'lora_unet_'.
# So it is taken for a checkpoint in one folder, and handed as is to peft in the other --
# where peft applies NOTHING and says nothing. The LoKr is now supported by MERGING into the
# weights (see _merge_lokr); the LoHa stays refused by name.
# ----------------------------------------------------------------------------
_LYCORIS_SUFFIXES = ("lokr_", "hada_")
_LYCORIS_CACHE = {}


def _lycoris_header(path):
    """The JSON header of a .safetensors (a short read), {} when unreadable."""
    try:
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(min(n, 3_000_000)).decode("utf-8", "ignore"))
    except Exception:
        return {}


def _lycoris_algo_from_header(hdr):
    """'LoKr', 'LoHa' ou None."""
    suf = [k.rsplit(".", 1)[-1] for k in hdr if k != "__metadata__"]
    if sum(1 for s in suf if s.startswith("hada_")) >= 4:
        return "LoHa"
    if sum(1 for s in suf if s.startswith("lokr_")) >= 4:
        return "LoKr"
    return None


def _lycoris_algo(path):
    """The LyCORIS algorithm of a file, from the header alone, memoised on
    (path, size, mtime): _apply_loras runs ON EVERY generation, and re-reading a gigabyte's
    header on a network drive for every image would be paying dearly for an answer that does
    not change."""
    try:
        st = os.stat(path)
        k = (os.path.abspath(path), st.st_size, int(st.st_mtime))
    except OSError:
        return None
    if k not in _LYCORIS_CACHE:
        _LYCORIS_CACHE[k] = _lycoris_algo_from_header(_lycoris_header(path))
    return _LYCORIS_CACHE[k]


def _lycoris_reason(hdr):
    """The named refusal for a LyCORIS found where it does not belong."""
    algo = _lycoris_algo_from_header(hdr) or "adapter"
    if algo == "LoKr":
        return ("LyCORIS LoKr, not a checkpoint - it IS supported, but as an adapter: "
                "move it to the LoRA folder and pick it in Models > LoRA, where it is "
                "merged into the weights at load")
    return (f"LyCORIS {algo}, neither a LoRA nor a checkpoint - only LoKr is supported "
            f"here. Use a version already merged into a base model, or merge it "
            f"yourself with LyCORIS/sd-scripts first")


def _lora_unsupported(path):
    """A reason (str) when this file cannot be applied as a PEFT adapter, otherwise None.
    A LoKr returns None: it IS supported, by merging, and it is removed upstream from the set
    handed to peft."""
    algo = _lycoris_algo(path)
    if algo and algo != "LoKr":
        return (f"LyCORIS {algo} - only LoKr is supported here; peft recognises none "
                f"of its Hadamard factors and would apply nothing, silently")
    return None


def _safetensors_unsupported(path):
    """Returns a reason (str) when the .safetensors is NOT loadable by diffusers,
    otherwise None. Only reads the header (fast). Two unsupported cases:
      - FP8 (F8_E4M3 / F8_E5M2) -> "FP8"
      - ComfyUI / SVDQuant-Nunchaku style INT8/INT4 quantization (I8/U8 tensors +
        'weight_scale' factors) -> "INT8/INT4 quantized". diffusers does not dequantize that
        scheme.
      - SVDQuant / Nunchaku (tensors named '*.qweight') -> "SVDQuant/Nunchaku INT4".
        That scheme does NOT use 'weight_scale', hence a dedicated detection.
    Take the unquantized BF16/FP16 build, or a .gguf.
"""
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
        # A LyCORIS filed with the checkpoints: none of the guards below sees it
        # (neither '.lora_A/B', nor a 'lora_unet_' prefix, and its dtypes are normal bf16).
        if _lycoris_algo_from_header(hdr):
            return _lycoris_reason(hdr)
        # A LoRA file filed in the checkpoints folder (a classic mistake): loading it
        # as a transformer sends diffusers looking for a default config (SD1.5) -> a 404
        # 'stable-diffusion-v1-5 does not appear to have a file named config.json'.
        if lora_keys >= 4:
            return "LoRA file, not a checkpoint - move it to the LoRA folder and pick it in Models > LoRA"
        # '*.qweight' = pre-quantized weights (SVDQuant/Nunchaku, GPTQ-like). A clear
        # signal: a normal BF16/FP16 checkpoint never has a 'qweight'. Not dequantizable by
        # our path (a grouped INT4 scheme), unlike the ComfyUI 'scaled' FP8/INT8 which DO go
        # through the conversion (_converted_folder).
        if has_qweight:
            return "SVDQuant/Nunchaku INT4"
    except Exception:
        pass
    return None


def _safetensors_is_fp8(path):
    """Compat: the old FP8-only predicate. Prefer _safetensors_unsupported()."""
    return _safetensors_unsupported(path) == "FP8"


# Inherited from crispz-qwen-edit and UNUSED here (as is the 'gguf_arch' config key it
# reads): list_checkpoints() filters no .gguf on its architecture, and _read_gguf_state_dict
# dequantizes whatever Krea 2 export it is given. A diffusion GGUF declares its architecture
# in 'general.architecture' ('flux', 'qwen_image', 'krea2'...), which _gguf_arch() can still
# read from a header when a filter is wanted one day.
GGUF_ARCH = str(CONFIG.get("gguf_arch") or "qwen_image").strip().lower()

_GGUF_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _gguf_skip(f, t):
    """Skips past a GGUF value in the stream without reading it (strings and arrays
    included)."""
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
    """The 'general.architecture' of a .gguf -- reads the header only (a few KB), never
    the weights. Returns 'qwen_image' / 'flux' / 'krea2' / 'llama'... or None when
    unreadable (in that case nothing is filtered: better to try than to discard a valid
    model)."""
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
    """Folders to scan for single-file checkpoints: the main one + the extra one (when
    set), with no duplicate path."""
    dirs = [CHECKPOINTS_DIR]
    if CHECKPOINTS_EXTRA_DIR and CHECKPOINTS_EXTRA_DIR not in dirs:
        dirs.append(CHECKPOINTS_EXTRA_DIR)
    return dirs


def list_checkpoints():
    """Local diffusers folders usable as a Krea 2 base model.

    Single-file .safetensors (Civitai bf16 and 'scaled' FP8/INT8) are offered, as are the
    ComfyUI-GGUF .gguf files (arch 'krea2'): everything goes through the CONVERSION to a
    diffusers folder on the first load (a disk cache, see _converted_folder; the GGUF is
    dequantized to bf16 - it only saves the download, not the VRAM). These stay discarded:
    stray LoRAs and SVDQuant (an INT4 scheme that cannot be dequantized), with the reason in
    the console.

    A folder is only kept when it holds a model_index.json (the diffusers layout).
    The official HF repos are added by the UI (ZIMAGE_BASE_REPOS), not here.
"""
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
    """Absolute path of a single-file checkpoint from its file name, looked up in the
    checkpoints folders (main then extra). Returns name as is when it is already absolute;
    falls back to the main folder when not found."""
    if not name or os.path.isabs(name):
        return name
    for d in _checkpoint_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(CHECKPOINTS_DIR, name)


def list_loras():
    """LoRAs (.safetensors / .ckpt / .pt) from the loras folder, RECURSIVELY (subfolders
    included). Returns paths RELATIVE to LORAS_DIR with '/' (e.g.
    'subfolder/my_lora.safetensors') -> set_loras / resolve resolve them through
    os.path.join(LORAS_DIR, name)."""
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
    """Sets (or clears with '' / None) the additional checkpoints folder."""
    global CHECKPOINTS_EXTRA_DIR
    CHECKPOINTS_EXTRA_DIR = (path or "").strip()


def set_loras_dir(path):
    global LORAS_DIR
    if path:
        LORAS_DIR = path


def _read_safetensors_metadata(path):
    """Reads the JSON header (__metadata__) of a .safetensors WITHOUT loading the weights."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
    return (json.loads(header.decode("utf-8")) or {}).get("__metadata__", {}) or {}


def lora_keywords(path):
    """Extracts a LoRA's keywords / trigger words from its metadata: explicit trigger
    fields + the top training tags (ss_tag_frequency)."""
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
    """Sets the active LoRAs. slots = a list of (name_or_None, weight). Resolves the
    names to paths, ignores the Nones.

    Does NOT reload the model: the LoRAs are hot-swapped on the transformer already in
    VRAM (_apply_loras, called by _ensure_base on the next run).
"""
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
    """Sets the Omni/Edit model (an HF repo or a folder). Invalidates the omni pipe."""
    global OMNI_MODEL
    repo = (repo or "").strip()
    if repo != OMNI_MODEL:
        OMNI_MODEL = repo
        _DERIVED.pop("omni", None)
        _log(f"Omni model -> {repo or '(none)'}")


def check_omni_available():
    """The Edit tab = Qwen-Image-Edit (an instruction-based edit model). Checks that the
    configured edit repo exists on Hugging Face (public API).

    Krea 2 has NO edit model (CAPABILITIES['omni'] False, DEFAULT_OMNI_REPO None): a clear
    answer instead of an AttributeError on None.strip() should the button be reached anyway.
"""
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


@_gpu_exclusive
def set_offload_mode(mode):
    """Changes the CPU offload mode. Invalidates the pipe (the hooks are set at load
    time). An unknown value -> 'auto' (never 'none': the fallback must be the SAFE mode)."""
    global OFFLOAD_MODE, _AUTO_OFFLOAD
    mode = str(mode or "").strip().lower()
    mode = mode if mode in OFFLOAD_CHOICES else "auto"
    if mode != OFFLOAD_MODE:
        OFFLOAD_MODE = mode
        _AUTO_OFFLOAD = ""   # 'auto' runs the VRAM test again on the next load
        free_vram()
        _log(f"offload -> {OFFLOAD_MODE}: pipeline invalidated -> will reload")


# ---- Offload 'auto': a VRAM test at load time + a runtime safety net (cz_hw) ----

def _hw_profile_path():
    """Profile of the VRAM test's verdicts (JSON), next to the other caches."""
    return os.path.join(HERE, "cache", "hw_profile.json")


def _model_footprint_gb():
    """VRAM footprint (GB) of the whole pipeline under 'none' offload (weights in VRAM,
    activations excluded). Krea 2 (12.9B) is quantized to fp8 at load time: a measured peak of
    ~22.8 GB all-in-VRAM, and renders sustain ~32 GB from 1.5 MPx on -> 26 GB by default
    (margin included). In raw bf16 (quant 'none') it weighs 35.6 GB and fits in no 32 GB card.
    Overridable through config 'model_footprint_gb'."""
    try:
        v = float(CONFIG.get("model_footprint_gb", 0) or 0)
        if v > 0:
            return v
    except Exception:
        pass
    return 26.0


def _resolve_auto(retest=False):
    """The concrete mode for 'auto' (memoised for the process). The verdict is cached in
    cache/hw_profile.json per (GPU, torch/cuda build, model, quant): the test only costs one
    mem_get_info per combination, then a JSON read."""
    global _AUTO_OFFLOAD
    if _AUTO_OFFLOAD and not retest:
        return _AUTO_OFFLOAD
    mode, why = cz_hw.resolve(
        "auto", footprint_gb=_model_footprint_gb(),
        model_id=(ZIMAGE_TRANSFORMER or BASE_REPO), dtype=str(QUANT_MODE or "bf16"),
        profile_path=_hw_profile_path(), retest=retest)
    _AUTO_OFFLOAD = mode
    _log(f"offload auto -> {mode} ({why})")
    return mode


def offload_status():
    """Status line for the UI: the requested mode + the 'auto' resolution when relevant."""
    if OFFLOAD_MODE != "auto":
        return f"offload: {OFFLOAD_MODE} (explicit)"
    if not _AUTO_OFFLOAD:
        return "offload: auto (resolves at the next model load)"
    return f"offload: auto -> {_AUTO_OFFLOAD}"


def retest_offload():
    """The UI's 'Re-test VRAM' button: runs the test again, ignoring the profile (another
    app closed/opened, a driver change...). Invalidates the pipe when the verdict changes."""
    if OFFLOAD_MODE != "auto":
        return f"Offload is '{OFFLOAD_MODE}' (explicit) - select 'auto' to use the VRAM test."
    old = _AUTO_OFFLOAD
    mode = _resolve_auto(retest=True)
    if old and mode != old:
        free_vram()
        return f"auto -> {mode} (was {old}; the pipeline will reload)"
    return f"auto -> {mode}"


def _vram_guard_kwargs():
    """Runtime safety net: a callback_on_step_end that checks AFTER the first denoise step
    in effective mode 'none' that the VRAM is not saturated (the load-time test estimates; a
    third-party process may have arrived since, or the requested resolution exceeds the
    margin). Saturated -> the flag + an interruption of the denoise; the caller switches to
    'model' and replays the job ONCE.
    {} when the guard is pointless (offload already on, no CUDA).
"""
    if DEVICE != "cuda" or _effective_offload() != "none":
        return {}

    def _cb(pipe, i, t, cb_kwargs):
        global _VRAM_DOWNGRADE
        if i == 0 and cz_hw.vram_saturated():
            _VRAM_DOWNGRADE = True
            pipe._interrupt = True
        return cb_kwargs
    return {"callback_on_step_end": _cb}


def _consume_vram_downgrade():
    """When the guard has fired: applies the downgrade to 'model', records it in the
    profile (the next boot starts in 'model' directly) and releases the pipe. True -> the
    caller replays the job once."""
    global _VRAM_DOWNGRADE, _AUTO_OFFLOAD
    if not _VRAM_DOWNGRADE:
        return False
    _VRAM_DOWNGRADE = False
    _log("WARNING: VRAM saturated after the first denoise step in offload 'none' "
         "-> the render would spill to shared RAM (50-100x slower, no error). "
         "Switching to 'model' and retrying the job once.")
    cz_hw.record_downgrade(_hw_profile_path(), ZIMAGE_TRANSFORMER or BASE_REPO,
                           str(QUANT_MODE or "bf16"), "model",
                           "VRAM saturated after denoise step 1")
    if OFFLOAD_MODE == "auto":
        _AUTO_OFFLOAD = "model"
        free_vram()
    else:
        set_offload_mode("model")
    return True


@_gpu_exclusive
def free_vram():
    """Releases the base pipeline + the derived pipelines and gives the VRAM back
    (step 3: unload on idle or the /unload endpoint). Lazy reload."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _APPLIED_LORAS, _APPLIED_LOKRS
    global _TEXT_ENCODER_ACTIVE
    _BASE_PIPE = None
    _DERIVED = {}
    _LOADED_KEY = None
    _APPLIED_LORAS = []      # no pipe any more -> no adapter applied either
    _APPLIED_LOKRS = []      # ... nor of weights where a LoKr would be merged
    _TEXT_ENCODER_ACTIVE = ""  # ... nor of a replacement encoder loaded
    _embed_cache_clear(" (VRAM freed)")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def is_oom(e):
    """True when `e` is a lack of VRAM, in either of its two forms: torch's allocator one
    ("CUDA out of memory. Tried to allocate ...") and a direct CUDA call's one ("CUDA error:
    out of memory"). The second happens once torch's cache has reserved everything: a kernel
    loaded on demand finds nothing left and cannot claim anything back from that cache
    (carried over from crispz-klein 1.36.4)."""
    s = str(e).lower()
    return "out of memory" in s or "alloc_failed" in s


def release_vram(offload=False, why=""):
    """Gives the driver back the VRAM that torch's cache holds in reserve, unloading
    nothing.

    torch only empties its cache when ITS allocator fails; the other consumers fail without
    being able to reclaim it. offload=True also puts back on the CPU the models an
    interrupted call left on the GPU under 'model' offload, for EVERY pipeline loaded with
    its hooks (the base one, and the Omni pipeline when it is loaded separately). On
    crispz-klein, a transformer half-moved by an OOM stayed on the GPU: 10.8 GB stuck, and
    every later render failed until a restart. `why` logs the VRAM state afterwards.
"""
    if offload:
        seen = set()
        for p in [_BASE_PIPE, *_DERIVED.values()]:
            if p is None or id(p) in seen or not getattr(p, "_all_hooks", None):
                continue
            seen.add(id(p))
            try:
                p.maybe_free_model_hooks()   # diffusers: everything on the CPU, hooks put back
            except Exception as e:
                _dbg(f"release_vram: offload failed ({e})")
    gc.collect()
    if DEVICE != "cuda":
        return
    try:
        torch.cuda.empty_cache()
        if why:
            free, total = torch.cuda.mem_get_info()
            _log(f"VRAM released ({why}): {free / 1024 ** 3:.1f} GB free of "
                 f"{total / 1024 ** 3:.1f}, torch holds "
                 f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB "
                 f"(reserved {torch.cuda.memory_reserved() / 1024 ** 3:.1f})")
    except Exception as e:
        _dbg(f"release_vram: {e}")


# ----------------------------------------------------------------------------
# Repairing the LoRA state_dict for Krea 2.
#
# diffusers' Krea 2 converter (_convert_non_diffusers_krea2_lora_to_diffusers) walks the
# keys matching `\.lora_[AB]\.weight$`, maps them (wq/wk/wv/wo/gate -> to_q/to_k/to_v/
# to_out.0/to_gate) and pops them -- but it NEVER pops the `.alpha` keys, and never applies
# them either. Any LoRA carrying alphas therefore dies at the end of the conversion on
# "`state_dict` should be empty at this point but has ...attn.gate.alpha".
# Folding alpha into the up weight removes the orphan keys AND applies the scaling the
# converter was dropping: with no `.alpha` left, diffusers sets lora_alpha = rank
# (get_peft_kwargs), i.e. a runtime scale of 1.0, so the factor has to be in the weights.
# ----------------------------------------------------------------------------
_LORA_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lora.down.weight")
_LORA_UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight", ".lora.up.weight")


def fold_lora_alpha(sd):
    """Folds every '.alpha' into the matching up weight (x alpha/rank) and drops the key.
    Returns (state_dict, number folded). rank = down.shape[0]."""
    sd = dict(sd)
    folded = 0
    for ak in [k for k in sd if k.endswith(".alpha")]:
        base = ak[: -len(".alpha")]
        down = next((sd[base + s] for s in _LORA_DOWN_SUFFIXES if base + s in sd), None)
        up_key = next((base + s for s in _LORA_UP_SUFFIXES if base + s in sd), None)
        if down is None or up_key is None:
            # An alpha without its pair: dropping it is what the converter would have done.
            sd.pop(ak)
            _dbg(f"LoRA alpha without weights, dropped: {ak}")
            continue
        rank = int(down.shape[0]) or 1
        scale = float(sd[ak].item()) / rank
        up = sd[up_key]
        sd[up_key] = (up.float() * scale).to(up.dtype)
        sd.pop(ak)
        folded += 1
    return sd, folded


def _lora_needs_repair(sd):
    """True for a Krea 2 LoRA whose '.alpha' keys the converter will leave behind: the
    trainer layout (blocks.N.attn|mlp.<module>) in the lora_A/lora_B form, WITH alphas.
    A LoKr file never qualifies -- those go through _apply_lokrs_to, not this path."""
    import re
    if not any(k.endswith(".alpha") for k in sd):
        return False
    return any(re.search(r"(?:^|\.)(?:blocks|transformer_blocks)\.\d+\."
                         r"(?:attn|mlp)\.\w+\.lora_[AB]\.weight$", k) for k in sd)


def _lora_source(path):
    """What to hand load_lora_weights for `path`: (source, extra kwargs).

    By default the FOLDER + weight_name -- diffusers refuses a full path offline
    (HF_HUB_OFFLINE: "must specify a weight_name"), and that route is the tested one. A file
    whose alphas the Krea 2 converter would leave behind is repaired in memory first and
    passed as a dict.
"""
    try:
        from safetensors.torch import load_file
        sd = load_file(path)
    except Exception as e:                      # not a safetensors, unreadable: as before
        _dbg(f"LoRA pre-read skipped for {os.path.basename(path)}: {e}")
        return (os.path.dirname(path) or "."), {"weight_name": os.path.basename(path)}
    if not _lora_needs_repair(sd):
        return (os.path.dirname(path) or "."), {"weight_name": os.path.basename(path)}
    sd, folded = fold_lora_alpha(sd)
    _log(f"LoRA {os.path.basename(path)} repaired for diffusers: {folded} alpha folded "
         f"into the up weights (the Krea 2 converter leaves them behind)")
    return sd, {}


def _load_lora(pipe, *args, **kwargs):
    """pipe.load_lora_weights with REAL tensors (low_cpu_mem_usage=False).

    A diffusers/peft build that does not know that parameter refuses it with a TypeError:
    the call is then retried without it rather than failing the application (the default
    creates the layers on 'meta' there -- see _apply_loras).
"""
    try:
        return pipe.load_lora_weights(*args, low_cpu_mem_usage=False, **kwargs)
    except TypeError as e:
        if "low_cpu_mem_usage" not in str(e):
            raise
        _dbg(f"load_lora_weights without low_cpu_mem_usage ({e})")
        return pipe.load_lora_weights(*args, **kwargs)


def _offload_hooks(pipe):
    """Number of 'model' offload hooks diffusers has set on this pipe (0 = none)."""
    return len(getattr(pipe, "_all_hooks", None) or [])


def restore_offload(pipe, why=""):
    """Puts the pipe back in its EFFECTIVE offload state when it has been left on the CPU.

    diffusers REMOVES the offload hooks before applying a LoRA and puts them back after.
    When the load fails in between, nobody puts them back: the pipe stays on the CPU, its
    `_execution_device` becomes cpu, and EVERY later render fails on "Cannot generate a cpu
    tensor from a generator of type cuda" -- until the app is restarted. Caught on 2026-09-23
    with two DoRA LoRAs (crispz-klein 1.36.6). Returns True when the state has been restored.
"""
    if DEVICE != "cuda" or pipe is None:
        return False
    try:
        dev = pipe._execution_device
    except Exception:
        return False
    if str(getattr(dev, "type", dev)) == "cuda":
        return False
    off = _effective_offload()
    try:
        if off == "model":
            pipe.enable_model_cpu_offload()
        elif off == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(DEVICE)
    except Exception as e:
        _log(f"pipeline left on the CPU and NOT restored ({e}): restart crispz-krea2")
        return False
    _log(f"pipeline was left on the CPU{' after ' + why if why else ''} -> offload "
         f"'{off}' restored in place (no reload)")
    return True


def retry_on_oom(what, fn, *args, **kwargs):
    """Calls fn(*args, **kwargs); on a lack of VRAM, gives the VRAM back (torch's cache,
    models left on the GPU) and retries ONCE. A second failure gives the VRAM back again
    before re-raising: the process stays usable for the next render."""
    err = None
    for attempt in (1, 2):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not is_oom(e):
                raise
            # The traceback holds the frames, so their tensors on the GPU: it is
            # dropped BEFORE emptying the cache, or empty_cache reclaims nothing.
            err = e.with_traceback(None)
            err.__context__ = err.__cause__ = None
        if attempt == 1:
            _log(f"{what}: out of VRAM ({str(err).strip().splitlines()[0]}), "
                 f"freeing it and retrying once")
        release_vram(offload=True, why=what)
    raise err


# Beyond this side (px) attention slicing is turned on (whole-image 2K+ -> avoids the
# 32 GB VRAM spill). Below it (1024 tiles, 1024/1536 txt2img) -> slicing OFF = native SDPA
# attention = FAST (like ComfyUI). Tunable through config attention_slice_above.
_SLICE_ABOVE = int(CONFIG.get("attention_slice_above", 1664))

# Guard rail: beyond this side (px), a "whole image" refine (refine_tile=0) is
# auto-tiled (1024 tile). The default = the slicing threshold: beyond it a whole-image pass
# would be sliced (slow: ~120s at 2K) AND risks the VRAM spill (4K -> a crash). Tiling is
# faster AND safe.
_AUTO_TILE_ABOVE = int(CONFIG.get("auto_refine_tile_above", _SLICE_ABOVE))

# Size of the tile this auto-tiling uses. "auto" (the default) = computed by
# _pick_refine_tile; an integer freezes the size (the old behaviour: 1024).
# Measured (RTX 5090, 4096x4096 output, denoise 0.40, overlap 64): the cost per pixel is
# FLAT from 768 to 1024 (1.78 / 1.83 / 1.79 us/px) and only climbs beyond that (2.41 at
# 1536, 3.00 at 2048). So the time follows the TILED AREA (n x tile^2), not the tile size.
# But at 1024 the grid overflows: 960 does not divide 4096 -> the last tile is pulled back
# and overlaps the previous one by 832px instead of 64, that is 1.56x the image's area. At
# 896 the step lands right (1.20x) -> 36.7s instead of 46.9s on the same image, with an
# IDENTICAL number of tiles (25) and seams (8).
# Bounds [768, 1024]: below them tiles and seams multiply and each tile sees less context
# (the render drifts - a blurred background rebuilds differently, checked visually); above
# them the attention becomes superlinear.
_AUTO_TILE_MIN = int(CONFIG.get("auto_refine_tile_min", 768))
_AUTO_TILE_MAX = int(CONFIG.get("auto_refine_tile_max", 1024))
_AUTO_TILE_SIZE = str(CONFIG.get("auto_refine_tile", "auto")).strip().lower()


def _pick_refine_tile(w, h, overlap):
    """The tile that minimises the tiled area needed to cover w x h (= the pass's real
    cost).

    At equal area the LARGEST tile wins: fewer seams and more context per tile. An integer
    in auto_refine_tile short-circuits the computation (a frozen size).
"""
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
        cands.append((n * t * t, -t, t))       # the smallest area, then the largest tile
    return min(cands)[2]

# Denoise ceiling for the TILED refine. In tiles, each tile is re-diffused with the
# global prompt -> at a high denoise the diffusion rebuilds the subject (the cup, say) IN
# every tile = duplications. So the per-tile denoise is capped (the existing content then
# guides the diffusion, Ultimate SD Upscale style). The "whole image" refine keeps the
# requested denoise (no duplication is possible: a single pass over the whole composition).
# Tunable through config refine_tile_denoise_cap (0 = no cap).
_TILE_DENOISE_CAP = float(CONFIG.get("refine_tile_denoise_cap", 0.40))

# Prompt used for the TILED refine. The global prompt describes the WHOLE composition
# (not the tile) -> handing it to every tile pushes the diffusion to recreate the subject
# (the cup) in tiles that are nothing but background. So an EMPTY prompt is passed by
# default: each tile just refines the local detail. config refine_tile_prompt values:
#   "" (the default) = an empty prompt per tile
#   "global"/"scene" = reuses the scene's prompt (the old behaviour)
#   any other text = a generic prompt applied to every tile (e.g. "high detail, sharp")
_TILE_PROMPT = str(CONFIG.get("refine_tile_prompt", ""))


def _tile_prompt(scene_prompt):
    """The prompt to use per tile according to the config (empty by default,
    anti-duplication)."""
    if _TILE_PROMPT.strip().lower() in ("global", "scene"):
        return scene_prompt or ""
    return _TILE_PROMPT


def _set_slicing(pipe, longest_side):
    """Turns attention slicing on/off according to the largest side to process. Called
    before EVERY diffusion pass (txt2img/refine/tile/inpaint/outpaint/omni)."""
    try:
        if int(longest_side) > _SLICE_ABOVE:
            pipe.enable_attention_slicing()
        else:
            pipe.disable_attention_slicing()
    except Exception:
        pass


def _vram_str():
    """PyTorch's peak reserved VRAM / the total (to spot saturation -> a spill into
    Windows' shared RAM = extreme slowness, and TDR/'CUDA unknown error'). Does NOT see the
    other processes' VRAM (ComfyUI, etc.) -> use nvidia-smi for the real total."""
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
# img2img / inpaint derived through from_pipe (shared weights, no duplicate VRAM).
# ----------------------------------------------------------------------------
def _is_gguf_path(p):
    return bool(p) and str(p).lower().endswith(".gguf")


def _effective_offload(tpath=None):
    """The offload REALLY applied. A quantized GGUF transformer does not move onto the GPU
    through .to(cuda) nor in sequential -> only enable_model_cpu_offload places it on the GPU
    during the forward. So 'model' is forced for a GGUF base, whatever the setting says."""
    off = OFFLOAD_MODE
    if off == "auto":
        off = _resolve_auto()   # test VRAM (memoise + profil cache) -> mode concret
    t = ZIMAGE_TRANSFORMER if tpath is None else tpath
    if DEVICE == "cuda" and _is_gguf_path(t) and off != "model":
        off = "model"
    return off


# ----------------------------------------------------------------------------
# Single-file (Comfy/Civitai) -> diffusers folder conversion, in a disk cache.
#
# Krea2Transformer2DModel has no from_single_file in diffusers: the key mapping
# table does not exist upstream. So it is implemented HERE: the Comfy checkpoint
# is a 1:1 mirror of the diffusers model (430 keys on both sides, checked on the
# official re-export) - pure renaming, plus ONE reshape
# (mod.lin (36864,) -> scale_shift_table (6, 6144)). The ComfyUI 'scaled'
# FP8/INT8 variants (ConvRot included) are dequantized to bf16 during the
# conversion (the same path as crispz-studio, validated on Z-Image).
#
# The result is written ONCE as a diffusers folder (cache/krea2_convert/...)
# then reloaded through the EXISTING from_pretrained path (torchao quantization
# included): the first selection = a conversion (minutes, ~26 GB written), the
# later ones = a normal load.
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
    """A Comfy name -> (the diffusers name, reshape_rows|None). None for an unknown key."""
    if k in _KREA2_TOP_MAP:
        return _KREA2_TOP_MAP[k], None
    m = _re.match(r"^blocks\.(\d+)\.(.+)$", k)
    if m:
        if m.group(2) == "mod.lin":
            # (6*dim,) -> (6, dim): the modulation table is stored flat
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


# Range of each 8-bit format: the largest stored value a QUANTIZED weight
# (weight / scale) can reach.
_QUANT_RANGE = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0, torch.int8: 127.0}


def _stored_at_scale(t, s, qdtype, cfg=None):
    """True when the stored weights are ALREADY at their real scale, a weight_scale being
    supplied on top -- not to be applied. Carried over from crispz-klein 1.34.1.

    A normal 'scaled' FP8 stores weight / scale: it FILLS the format's range (448 in E4M3)
    and max|stored| / (scale x range) is 1 / scale (71 to 1,691 across the 16 FP8/INT8 files
    of the library). kleinFinalcutFP16FP8_comfyQuant stores its weights as they are (0.375
    out of 448) and supplies amax / 448 anyway: ratio 1.03. Applying the scale made every
    weight 1,200 to 1,700 times too small, and the image came out as noise. MX scales
    (uint8 = an E8M0 exponent) are never concerned.
"""
    rng = _QUANT_RANGE.get(qdtype)
    fmt = str((cfg or {}).get("format", "")).lower()
    if rng is None or s.dtype == torch.uint8 or fmt.startswith("mx"):
        return False
    smax = float(s.detach().float().abs().max())
    if smax <= 0.0:
        return False
    amax = float(t.detach().float().abs().max())
    # 1. the range is barely used (a normal file fills it)...
    if amax >= rng / 4:
        return False
        # 2. ... AND the scale describes exactly the stored values: ratio ~1.
    ratio = amax / (smax * rng)
    return 0.5 <= ratio < 2.0


def _hadamard_ortho(n):
    """The comfy-quants ConvRot 'regular hadamard' matrix -- CAREFUL, this is NOT
    Sylvester's construction: a precise H4 base, extended by Kronecker products up to n (a
    power of 4), normalised by 1/sqrt(n). Orthonormal AND symmetric -> the reconstruction
    multiplies by the same matrix again. (Carried over from crispz-studio, checked against
    comfy_quants/formats/convrot.py.)"""
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]])
    H = h4
    while H.shape[0] < n:
        H = torch.kron(H, h4)
    if H.shape[0] != n:
        raise ValueError(f"convrot groupsize {n} is not a power of 4")
    return H / (float(n) ** 0.5)


def _read_comfy_state_dict(path):
    """Reads a Krea 2 single-file (Comfy/Civitai) into RAM, dequantized to DTYPE:
      - an AIO bundle: only the 'model.diffusion_model.*' keys are kept;
      - X.weight (F8/I8) * X.weight_scale / X.scale_weight -> bf16;
      - an X.comfy_quant blob declaring 'convrot' -> the Hadamard rotation is UNDONE
        after the descale (otherwise the weights are pure noise);
      - the quantization keys are consumed and dropped.
    SEQUENTIAL read in physical order (data_offsets): a HDD collapses on random access
    (measured on crispz-studio: 349 s -> bound by the disk's throughput).
"""
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
    # Architecture guard: without a Krea 2 marker, a clear refusal (a stray
    # FLUX/Z-Image checkpoint would load noise).
    if not any(k[len(prefix):].startswith(("txtfusion.", "blocks.0.attn.wq"))
               for k, _ in entries):
        raise RuntimeError(
            f"{os.path.basename(path)}: not a Krea 2 checkpoint (no txtfusion/"
            f"blocks markers) - this build only converts Krea 2 single files.")
    entries.sort(key=lambda kv: kv[1].get("data_offsets", [0])[0])
    raw, qcfg = {}, {}
    # comfy-quants declares the scheme either in PER-TENSOR blobs (X.comfy_quant), or
    # CENTRALLY in __metadata__._quantization_metadata (StableYogi INT8:
    # {"layers": {"blocks.0.attn.gate": {"format": "int8_tensorwise", "convrot": true,
    # "convrot_groupsize": 256}}}). Ignoring that variant renders the weights as PURE NOISE
    # (the rotation never undone) - observed on realismByStableYogi_v25INT8Turbo. The
    # per-tensor blobs win (written afterwards, they override the metadata entry of the same
    # layer).
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
    # The dequantization work (fp32 cast + scales + un-rotation) is memory-bandwidth
    # bound on the CPU (~9 min measured on a 12.9B INT8): so it runs on the GPU when one is
    # available, tensor by tensor (~400 MB of VRAM at most), with the bf16 coming back to RAM.
    # convert_device: auto (the default, cuda when present) | cpu.
    dev = "cpu"
    try:
        if (torch.cuda.is_available()
                and str(CONFIG.get("convert_device", "auto")).lower() != "cpu"):
            dev = "cuda"
    except Exception:
        pass
    _had, sd = {}, {}
    n_dq = n_rot = n_pre = 0
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
            qdt = t.dtype
            t = t.to(dev).to(torch.float32)
            cfg = qcfg.get(k[:-len(".weight")]) if k.endswith(".weight") else None
            if s is not None and _stored_at_scale(t, s, qdt, cfg):
                s = None                     # already at scale: see _stored_at_scale
                n_pre += 1
            if s is not None:
                t = t * s.to(dev).to(torch.float32)
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
             + (f", {n_pre} already stored at scale: weight_scale NOT applied" if n_pre else "")
             + (f" on {dev}" if n_dq else "")
             + f" in {time.time() - t0:.1f}s")
    return sd


def _read_gguf_state_dict(path):
    """Reads a Krea 2 .gguf (a ComfyUI-GGUF export, arch 'krea2') and dequantizes
    EVERYTHING to DTYPE through the gguf lib (Q8_0/Q6_K/... -> float32 -> bf16). The tensor
    names are the SAME as the Comfy safetensors layout (checked: 430 identical keys) -> the
    same renaming table applies afterwards, and gguf.quants.dequantize already returns the
    torch orientation (checked on blocks.0.attn.wk: (1536, 6144)).

    NB: unlike crispz-studio (Z-Image), the GGUF can NOT stay quantized in VRAM here (no
    from_single_file for this architecture): it is converted ONCE to bf16 (a disk cache), and
    the runtime quantization stays torchao (float8_weight_only). So Q8_0 only saves the
    download, not the VRAM.
"""
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
    """(config_dict, {key: shape}) of the base repo's transformer, WITHOUT loading the
    weights (init_empty_weights). The config.json comes from the base repo's local HF cache ->
    the official model has to have been loaded at least once."""
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
    """The conversion cache folder. Config convert_cache: 'auto' (the default) =
    <app>/cache/krea2_convert, a path = a custom folder, 'off' = no cache (the conversion is
    then refused: 26 GB per entry, a throwaway tmp makes no sense)."""
    mode = str(CONFIG.get("convert_cache", "auto") or "auto")
    if mode.lower() == "off":
        return None
    if mode.lower() == "auto":
        return os.path.join(cz_core.HERE, "cache", "krea2_convert")
    return mode


def _prune_convert_cache(keep):
    """Evicts the least recently used conversions beyond convert_cache_max_gb
    (0 = unlimited). `keep` = a folder never to evict."""
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
    """The diffusers folder of the single-file `path`: converted on the first request
    (cached by (path, size, mtime)), reused afterwards."""
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
    """Loads ONLY the current transformer (without the rest of the pipeline), from a
    diffusers repo, quantized on the fly according to QUANT_MODE.

    A single-file (a Civitai .safetensors bf16/FP8/INT8 'scaled', or a ComfyUI-GGUF .gguf)
    goes through the CONVERSION to a diffusers folder (_converted_folder, a disk cache) then
    through this same from_pretrained path.

    Used both on a full load AND for the hot swap (_swap_transformer).
"""
    from diffusers import Krea2Transformer2DModel
    repo = ZIMAGE_TRANSFORMER or BASE_REPO
    single = bool(ZIMAGE_TRANSFORMER and _is_single_file(ZIMAGE_TRANSFORMER))
    if single:
        bad = None if ZIMAGE_TRANSFORMER.lower().endswith(".gguf") \
            else _safetensors_unsupported(ZIMAGE_TRANSFORMER)
        if bad:  # a stray LoRA / an SVDQuant; a .gguf is validated by its arch
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
        # The (expensive) conversion is only triggered HERE, so only on a real MISS of
        # the quantization cache.
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
# QUANTIZATION cache: the torchao form (~13 GB) is serialised ONCE then reloaded
# directly -> no more reading the ~26 GB of bf16 nor quantizing on every load (app
# startup included). A pickle .bin is imposed: torchao tensors do not serialise to
# safetensors.
# ----------------------------------------------------------------------------
def _quant_cache_dir():
    """Config quant_cache: 'auto' (the default) = <app>/cache/krea2_quant, a path = a
    custom folder, 'off' = disabled (a direct load)."""
    # OFF BY DEFAULT since 23/08: on torch 2.8 + this torchao, the UNPICKLED tensors
    # lose their fast kernels and renders are ~3.3x slower (an A/B measurement: 316 s direct
    # vs 1060 s through the pickle, same settings), while the load does drop to ~4 s. The
    # regression was invisible to the tests (pixel corr 1.0000: the maths stay right, only
    # the speed breaks). Kept as opt-in so it can be re-tested on a future torch/torchao pair
    # where the serialisation keeps the kernels.
    mode = str(CONFIG.get("quant_cache", "off") or "off")
    if mode.lower() == "off":
        return None
    if mode.lower() == "auto":
        return os.path.join(cz_core.HERE, "cache", "krea2_quant")
    return mode


def _prune_quant_cache(keep):
    """Evicts the least recently used quantized forms beyond quant_cache_max_gb
    (0 = unlimited)."""
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
    """Stable identity of the transformer's SOURCE, in order:
    the original single-file FILE -> path + size + mtime OF THE FILE (the key thus survives
    the deletion of the conversion cache - a lesson learned: the historical key pointed at
    the converted folder, and purging krea2_convert invalidated every quantized form); a
    diffusers folder -> its weights; an HF repo -> its identifier."""
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
    """A quant-cache entry is usable: the config + at least one shard."""
    if not os.path.isfile(os.path.join(d, "config.json")):
        return False
    try:
        return any(f.startswith("diffusion_pytorch_model") and f.endswith(".bin")
                   for f in os.listdir(d))
    except OSError:
        return False


def _load_transformer_quant_cached(sig_source, repo_loader, kw, label):
    """Loads the transformer through the quantization cache.

    sig_source = the original FILE (single-file) or the repo id: the key does NOT depend on
    the conversion cache. repo_loader = a callable -> the folder/repo to load on a MISS (it is
    THAT call which triggers the conversion, if any: on a HIT no conversion happens at all,
    even when krea2_convert has been purged).

    HIT: a direct from_pretrained of the pickled torchao form (~13 GB, ~4 s).
    MISS: repo_loader() -> a normal load (bf16 + quantize) THEN save_pretrained for every
    later time (a failure there is non-fatal).
    Entries with the HISTORICAL KEY (the converted folder) are migrated by stem: renamed and
    re-stamped, not reserialised.
"""
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
    # Migration of the historical entries (the key = the converted folder, destroyed
    # by a legitimate cleanup of the conversion cache): the same stem => the same model.
    # The most recent complete one is adopted, the others with that stem are dropped.
    if os.path.isdir(root) and not os.path.isdir(dst):
        cands = [os.path.join(root, d) for d in os.listdir(root)
                 if d.startswith(base[:24]) and os.path.join(root, d) != dst]
        def _adoptable(c):
            # only adopt a HISTORICAL entry (a key built on the converted folder) of
            # the SAME quantization scheme. Two refusals:
            #  - another scheme: a float8 pickle served after a move to int8 would be
            #    wrong;
            #  - a new-generation key of the SAME file (a sig starting with its path): if
            #    we are here then the source has CHANGED, the entry is stale and must be
            #    reconverted, not adopted.
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


def _meta_params(model, limit=8):
    """Names of `model`'s parameters/buffers left on the 'meta' device (declared, no data).
    Capped at `limit`: we only need to know THAT some exist, plus a few names for the log.

    One meta parameter is terminal. pipe.to(DEVICE) raises "Cannot copy out of meta tensor;
    no data!", and peft builds an adapter on the device of the layer it wraps, so every
    later LoRA load inherits meta and loops on "copying from a non-meta parameter in the
    checkpoint to a meta parameter in the current model, which is a no-op". Nothing can
    repair it in place: the model has to be reloaded from disk.
"""
    if model is None:
        return []
    found = []
    try:
        for gen in (model.named_parameters(), model.named_buffers()):
            for name, t in gen:
                if getattr(getattr(t, "device", None), "type", "") == "meta":
                    found.append(name)
                    if len(found) >= limit:
                        return found
    except Exception as e:
        _dbg(f"_meta_params: {e}")
    return found


def _clear_loras(pipe):
    """Removes EVERY LoRA adapter from the pipe to start from a clean state.

    unload_lora_weights() alone leaves, depending on the diffusers/peft versions, a residual
    peft_config on the transformer -> the next load warns ('Already found a peft_config')
    and, since the same adapter names are reused (cz_lora_i), the old adapter can stay in
    place (the wrong LoRA applied). So the remaining adapters are deleted explicitly by name
    after the unload.
"""
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


# ----------------------------------------------------------------------------
# Merging a LoKr into the weights: dW = w1 (x) w2, added to W.
#
# Neither peft nor diffusers knows how to apply a LoKr to a pipeline. But nothing stops us
# from MATERIALISING its update and adding it: that is what a merge does, except it happens
# here, locally, without depending on a checkpoint merged by someone else -- and those can
# be broken (measured on the published klein SNOFS: 145 times further from its base than
# its own adapter can explain).
#
# The keys go through _krea2_rename, the fork's table -- the very one used to convert a
# Comfy single-file into a diffusers folder. So it cannot diverge from the model's load.
# Checked on the real file before writing this: all 256 modules of snofs_krea_v1_4 land on
# an existing weight, shape included.
#
# An accepted trade-off, and an announced one: a merge is not an adapter. Changing the
# LoKr or its weight demands a transformer reload.
# ----------------------------------------------------------------------------
_LOKR_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")
# LoKrs MERGED into the current transformer. _apply_loras compares this set with the
# requested one and forces a full reload as soon as it changes.
_APPLIED_LOKRS = []


def _strip_lokr_prefix(name):
    for p in _LOKR_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def _lokr_factor(mod, which):
    """(matrix, rank) of a LoKr factor: the FULL matrix when it is there (rank None,
    there is none), otherwise the product of its two reduced-rank factors."""
    full = mod.get(f"lokr_{which}")
    if full is not None:
        return full.to(torch.float32), None
    a, b = mod.get(f"lokr_{which}_a"), mod.get(f"lokr_{which}_b")
    if a is None or b is None:
        return None, None
    return a.to(torch.float32) @ b.to(torch.float32), int(a.shape[1])


def _lokr_scale(mod, rank):
    """The LyCORIS scale factor. When w1 AND w2 are full there is no rank: LyCORIS
    applies no scalar, and ai-toolkit then writes alpha = lora_dim (measured: 1e10 on SNOFS),
    so alpha/rank is 1.0 too -- the two conventions agree. Otherwise alpha / rank, like
    peft."""
    if rank is None:
        return 1.0
    alpha = mod.get("alpha")
    return 1.0 if alpha is None else float(alpha) / float(rank)


def _lokr_delta(mod):
    """A LoKr module's float32 dW."""
    if "lokr_t2" in mod:
        raise ValueError("lokr_t2 (convolution factor) is not supported here")
    w1, r1 = _lokr_factor(mod, "w1")
    w2, r2 = _lokr_factor(mod, "w2")
    if w1 is None or w2 is None:
        raise ValueError("incomplete LoKr factors")
    return torch.kron(w1, w2) * _lokr_scale(mod, r1 if r1 is not None else r2)


def _merge_lokr(transformer, path, weight):
    """Merges a LoKr into the transformer's weights: W += weight * dW.

    MODULE BY MODULE: the complete delta weighs what the layers it touches weigh (tens of
    GB), and materialising it in one go would overflow the RAM for nothing; one layer at a
    time caps at a few hundred MB.

    Returns (n_merged, [what could not be]). Nothing is skipped silently.
"""
    from safetensors.torch import load_file
    sd = load_file(path)
    mods = {}
    for k, v in sd.items():
        base, _, suf = k.rpartition(".")
        if suf == "alpha" or suf.startswith(_LYCORIS_SUFFIXES):
            mods.setdefault(_strip_lokr_prefix(base), {})[suf] = v
    params = dict(transformer.named_parameters())
    hit, problems = 0, []
    for name in sorted(mods):
        r = _krea2_rename(name + ".weight")
        if not r or not r[0]:
            problems.append(f"{name}: unknown module (no mapping)")
            continue
        key, reshape_rows = r
        if reshape_rows is not None:
            problems.append(f"{name}: reshaped weight, not a plain linear")
            continue
        p = params.get(key)
        if p is None:
            problems.append(f"{key}: no such weight in the transformer")
            continue
        try:
            d = _lokr_delta(mods[name])
        except Exception as e:
            problems.append(f"{name}: {e}")
            continue
        if tuple(p.shape) != tuple(d.shape):
            problems.append(f"{key}: delta {tuple(d.shape)} vs weight {tuple(p.shape)}")
        else:
            with torch.no_grad():
                # float32 for the addition: adding a small delta to a bf16 weight
                # INSIDE bf16 loses the delta's low-order bits.
                p.copy_((p.float() + d.to(p.device).float() * float(weight)).to(p.dtype))
            hit += 1
        del d
    return hit, problems


def _lokr_set(loras):
    """The LoKr subset of a list of (path, weight): merged, not applied."""
    return [pw for pw in loras if _lycoris_algo(pw[0]) == "LoKr"]


def _peft_set(loras):
    """Everything that is not a LoKr, so what goes to peft. A LoHa deliberately STAYS in
    it: it is refused there BY ITS NAME, whereas filtering it silently here would make it
    disappear without a word."""
    return [pw for pw in loras if _lycoris_algo(pw[0]) != "LoKr"]


def _apply_lokrs_to(transformer):
    """Merges the LoKrs of LORAS into this transformer. To be called right after the load
    and BEFORE the offload: the weights are still on the CPU, whole, with no accelerate hook
    on them."""
    global _APPLIED_LOKRS
    _APPLIED_LOKRS = []
    for p, w in _lokr_set(LORAS):
        t0 = time.time()
        try:
            n, problems = _merge_lokr(transformer, p, w)
        except Exception as e:
            _log(f"LoKr NOT merged, {os.path.basename(p)}: {type(e).__name__}: {e}")
            continue
        if problems:
            _log(f"LoKr {os.path.basename(p)}: {len(problems)} tensor(s) NOT merged, "
                 f"first: {problems[0]}")
        if not n:
            _log(f"LoKr {os.path.basename(p)}: nothing merged - the render will look "
                 f"exactly as if it were not selected")
            continue
        _log(f"LoKr merged into the weights: {os.path.basename(p)} @ {w} "
             f"-> {n} tensor(s) in {time.time() - t0:.1f}s")
        _APPLIED_LOKRS.append((p, w))


def _apply_loras(pipe, force=False):
    """Synchronises the pipe's LoRA adapters with LORAS, WITHOUT reloading the model.

    The transformer stays in VRAM; only the PEFT adapters move:
      - same files, different weights -> set_adapters (immediate)
      - a different LoRA set          -> unload_lora_weights + reloading the LoRAs (~1s)
    The derived pipes (from_pipe) share that transformer -> they follow automatically.
    Returns True when applied, False on a failure (the caller falls back to a full reload).
"""
    global _APPLIED_LORAS
    # A LoKr is MERGED into the weights: it can be neither removed nor re-weighted
    # without starting again from the original transformer. As soon as the requested set
    # differs from the one already inside, we say so and hand over to a full reload.
    want_lokr = _lokr_set(LORAS)
    if not force and want_lokr != _APPLIED_LOKRS:
        _log("LoKr selection changed -> full reload. A LoKr is merged INTO the weights "
             "(it is not a PEFT adapter), so it cannot be swapped or re-weighted in "
             "place: " + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in want_lokr)
                          or "none") + " wanted, "
             + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in _APPLIED_LOKRS)
                or "none") + " in the weights")
        return False
    wanted = _peft_set(LORAS)
    if not force and _APPLIED_LORAS == wanted:
        return True
    # low_cpu_mem_usage=False on EVERY load (see _load_lora): diffusers' default
    # creates the adapter's layers on 'meta' then copies the weights into them. A DoRA whose
    # 'dora_scale' keys diffusers filters out then leaves parameters without data, and the
    # first move raises "Cannot copy out of meta tensor". With real tensors, a missing key
    # keeps its initial value.
    had_hooks = _offload_hooks(pipe)
    old_paths = [p for p, _ in _APPLIED_LORAS]
    new_paths = [p for p, _ in wanted]
    try:
        if not force and old_paths and old_paths == new_paths:
            # Only the weights change -> an instant re-weighting.
            pipe.set_adapters(_lora_names(wanted), [float(w) for _, w in wanted])
            _APPLIED_LORAS = list(wanted)
            _log("LoRA weights updated in place (no reload): "
                 + ", ".join(f"{os.path.basename(p)}@{w}" for p, w in wanted))
            return True
        if old_paths or force:
            _clear_loras(pipe)
        names, weights = [], []
        for i, (p, w) in enumerate(wanted):
            if os.path.isfile(p):
                # A format peft cannot apply (a LoHa): it is named and we move on.
                # Letting it slip would be worse than an error -- the render would come out
                # without it, identical to a render where it had never been chosen.
                why = _lora_unsupported(p)
                if why:
                    _log(f"LoRA SKIPPED, {os.path.basename(p)}: {why}")
                    continue
                an = f"cz_lora_{i}"
                _log(f"applying LoRA: {os.path.basename(p)} (weight {w})")
                # _lora_source gives the folder + weight_name (diffusers offline
                # refuses a full path: "must specify a weight_name"), or a repaired
                # state_dict when the Krea 2 converter would choke on the file's alphas.
                src, src_kw = _lora_source(p)
                _load_lora(pipe, src, adapter_name=an, **src_kw)
                names.append(an)
                weights.append(float(w))
            else:
                _log(f"LoRA file not found, ignored: {p}")
        if names:
            pipe.set_adapters(names, weights)
        _APPLIED_LORAS = list(wanted)
        if not force:
            _log("LoRAs hot-swapped (no model reload)")
        return True
    except Exception as e:
        _log(f"LoRA hot-swap failed ({e}); falling back to a full reload")
        # The adapters are left half-injected: reusing that state would apply the wrong
        # LoRA (the cz_lora_i names are reused) or copy from a 'meta' parameter. Wipe it
        # BEFORE restoring the offload, while diffusers still has its hooks off.
        _clear_loras(pipe)
        # diffusers removed the offload hooks before loading and did not get to put
        # them back: without this, the pipe stays on the CPU and EVERY later render fails,
        # including the ones that have nothing to do with this LoRA.
        if had_hooks and not _offload_hooks(pipe):
            restore_offload(pipe, "a failed LoRA load")
        _APPLIED_LORAS = []
        return False


def _swap_transformer(pipe):
    """Replaces ONLY the transformer of the already cached pipeline: the VAE, the text
    encoder, the tokenizer and the scheduler stay in VRAM (they are most of the load time).
    Valid only with an identical base repo + EFFECTIVE offload.

    Returns True when the swap succeeded, False -> the caller does a full reload.
"""
    global _APPLIED_LORAS, _DERIVED
    t0 = time.time()
    old_t = _LOADED_KEY[1] if _LOADED_KEY else None
    # Switching to/from a GGUF changes the EFFECTIVE offload (a GGUF forces 'model')
    # -> the accelerate hooks and the placement differ: no tinkering, we reload.
    if _effective_offload(old_t) != _effective_offload(ZIMAGE_TRANSFORMER):
        _log("transformer swap skipped (GGUF changes the effective offload) -> full reload")
        return False
    try:
        _log(f"switching Krea 2 transformer -> {ZIMAGE_TRANSFORMER or BASE_REPO} "
             "(keeping VAE + text encoder in VRAM)")
        new_t = _load_transformer()
        # A new transformer = new weights: the LoKrs merged into the old one are not
        # in it. So they are merged again BEFORE the placement, while it is on the CPU.
        _apply_lokrs_to(new_t)
        old = getattr(pipe, "transformer", None)
        off = _effective_offload()
        # Offload: the accelerate hooks are set on the components. They have to be
        # removed before the swap, or the new transformer has none and the old one keeps
        # its own.
        if DEVICE == "cuda" and off in ("model", "sequential"):
            try:
                pipe.remove_all_hooks()
            except Exception as e:
                _dbg(f"remove_all_hooks: {e}")
        try:
            pipe.register_modules(transformer=new_t)   # API diffusers (met a jour le config)
        except Exception:
            pipe.transformer = new_t
        # Free the OLD transformer BEFORE putting the new one on the GPU: otherwise
        # old + new + VAE/encoders exceed the VRAM -> a spill into shared RAM that never
        # recovers (measured on a multi-checkpoint XYZ grid on the studio side: 1.7 s/step
        # -> 300-600 s/step, then a crash). The derived pipes (from_pipe) point at the old
        # one too -> purge them first, or `del old` frees nothing (from_pipe is free, it
        # will be rebuilt).
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
                new_t.to(DEVICE)       # never a GGUF here (offload forced to 'model')
        # The LoRA adapters were applied on the old transformer -> to be reapplied.
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
    """Loads (when needed) the base txt2img pipeline. Handles the single-file/GGUF
    transformer and the offload. Cached by (repo, transformer, offload).

    Two hot swaps avoid a full reload (transformer + VAE + text encoder, tens of seconds):
      - different LoRAs            -> _apply_loras (the PEFT adapters alone)
      - a different transformer, same base repo + offload -> _swap_transformer.
"""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _BASE_SCHED_CONFIG, _APPLIED_LORAS
    global _TEXT_ENCODER_ACTIVE
    key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE)
    _dbg(f"_ensure_base key={key} cached={_LOADED_KEY}")
    if _BASE_PIPE is not None and _LOADED_KEY == key:
        # Safety net: a pipe left on the CPU by an earlier failure (LoRA, offload)
        # would make THIS render fail on "Cannot generate a cpu tensor from a generator of
        # type cuda", and every later one too.
        restore_offload(_BASE_PIPE, "an earlier failure")
        if _apply_loras(_BASE_PIPE):
            # A LoRA load can report success and still have left parameters on 'meta'
            # (see _meta_params). Reusing the pipe would fail on the first .to() and
            # contaminate every later adapter -> reload from disk instead.
            _meta = _meta_params(getattr(_BASE_PIPE, "transformer", None))
            if not _meta:
                _dbg("base pipeline: reusing cached (no reload)")
                return _BASE_PIPE
            _log(f"meta parameters on the cached transformer ({len(_meta)}, e.g. "
                 f"{_meta[0]}) -> forced reload from disk")
            free_vram()
        else:
            _dbg("base pipeline: LoRA hot-swap failed -> free + reload")
            free_vram()
    elif _BASE_PIPE is not None:
        # Only the transformer changes (same base repo + same offload)? -> reload the
        # transformer ONLY and keep VAE + text encoder in VRAM.
        if (_LOADED_KEY and _LOADED_KEY[0] == BASE_REPO and _LOADED_KEY[2] == OFFLOAD_MODE
                and _swap_transformer(_BASE_PIPE)):
            _LOADED_KEY = key
            return _BASE_PIPE
        _dbg("base pipeline: key changed -> free + reload")
        free_vram()
    from diffusers import Krea2Pipeline
    t0 = time.time()
    # The transformer is ALWAYS loaded separately (unlike the other forks): that is
    # the only way to apply the torchao quantization to it, which is essential to fit in
    # 32 GB. Without an explicit override, the base repo's is the one quantized.
    kwargs = {"transformer": _load_transformer()}
    # A replacement encoder: checked on the config then loaded with the repo's class,
    # in DTYPE (torchao stays reserved for the transformer above). An encoder that does not
    # suit (the base repo changed since the choice, a moved folder, an unreadable config,
    # weights that will not load) is dropped WITH a log line: the repo's encoder runs, the
    # render happens, and the metadata says so.
    _TEXT_ENCODER_ACTIVE = ""
    if TEXT_ENCODER:
        try:
            _why = _text_encoder_problem(TEXT_ENCODER)
            if not _why:
                kwargs["text_encoder"] = _load_monitor(
                    f"text encoder {_encoder_label(TEXT_ENCODER)}",
                    lambda: _load_text_encoder(TEXT_ENCODER))
                _TEXT_ENCODER_ACTIVE = TEXT_ENCODER
        except Exception as e:
            kwargs.pop("text_encoder", None)
            _why = f"it could not be loaded ({type(e).__name__}: {e})"
        if _why:
            _log(f"text encoder {_encoder_label(TEXT_ENCODER)} NOT used: {_why}. "
                 f"{BASE_REPO}'s own encoder runs instead; the image metadata says so "
                 f"(text_encoder_not_applied).")
        else:
            _log(f"text encoder: {_encoder_label(TEXT_ENCODER)} replaces {BASE_REPO}'s "
                 f"own (tokenizer, VAE and transformer unchanged)")
    _off_label = (f"auto->{_resolve_auto()}" if OFFLOAD_MODE == "auto" else OFFLOAD_MODE)
    _log(f"loading Krea 2 base: {BASE_REPO} (offload={_off_label}, dtype=bf16, "
         f"quant={QUANT_MODE}) ... first time downloads ~35.7 GB from HF (gated), then cached")
    def _fresh_base(fresh_transformer=False):
        """Builds the base pipe. fresh_transformer=True also reloads the transformer
        OVERRIDE: a 'meta' parameter lives in that module, so handing the same instance
        back to from_pretrained would carry the problem over. Dropping it from kwargs
        first lets the broken one be collected."""
        if fresh_transformer and kwargs.get("transformer") is not None:
            kwargs.pop("transformer", None)
            gc.collect()
            kwargs["transformer"] = _load_transformer()
        return Krea2Pipeline.from_pretrained(BASE_REPO, torch_dtype=DTYPE, **kwargs)

    pipe = _load_monitor(f"Krea 2 base {BASE_REPO}", _fresh_base)
    # Capture the scheduler's native (flow-matching) config -> the base for building
    # the other samplers (euler/dpm2a/dpmpp2m) without losing shift/flow params.
    try:
        _BASE_SCHED_CONFIG = dict(pipe.scheduler.config)
    except Exception:
        _BASE_SCHED_CONFIG = None
    # LoRAs (on the base's transformer -> shared by the derived pipes).
    # force=True: a new pipe, no adapter applied -> we (re)apply everything.
    _APPLIED_LORAS = []
    # The LoKrs BEFORE any move/offload: the transformer is still on the CPU, in one
    # piece, with no accelerate hook -- the only window where merging into the weights is
    # simple and safe.
    _apply_lokrs_to(pipe.transformer)
    if LORAS:
        _apply_loras(pipe, force=True)
        # The return value used to be ignored: a half-injected adapter went straight to the
        # .to(DEVICE) / enable_*_cpu_offload below and raised "Cannot copy out of meta
        # tensor; no data!". A meta parameter cannot be repaired in place -> the model is
        # reloaded from disk, without any adapter (the render then runs LoRA-free rather
        # than not at all, and the log says so).
        _meta = _meta_params(getattr(pipe, "transformer", None))
        if _meta:
            _log(f"meta parameters left after the LoRA load ({len(_meta)}, e.g. "
                 f"{_meta[0]}) -> reloading {BASE_REPO} from disk WITHOUT any adapter")
            del pipe
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            _APPLIED_LORAS = []
            pipe = _load_monitor(f"Krea 2 base {BASE_REPO} (reload, no LoRA)",
                                 lambda: _fresh_base(fresh_transformer=True))
    # Attention slicing: SET PER CALL through _set_slicing (according to the
    # resolution processed), NOT at load time. In tiles/at 1024 -> slicing OFF = native SDPA
    # attention, fast (like ComfyUI). Whole-image 2K+ -> slicing ON to avoid the 32 GB VRAM
    # spill.
    # enable_*_cpu_offload handles the device itself -> do NOT call .to(cuda) then.
    # IMPORTANT: a quantized GGUF transformer does NOT move onto the GPU through .to(cuda)
    # (offload=none) nor in sequential -> it stays on the CPU = ULTRA slow (empty VRAM,
    # ~500s/step). Only enable_model_cpu_offload (accelerate) places it properly on the GPU
    # during the forward. So 'model' is forced for a GGUF base, whatever the UI/config says.
    _off = _effective_offload()
    _base_off = _resolve_auto() if OFFLOAD_MODE == "auto" else OFFLOAD_MODE
    if _off != _base_off:
        _log(f"quantized base: offload '{_base_off}' forced to '{_off}' (a quantized "
             f"transformer does not run on GPU in none/sequential -> would stay on CPU, "
             f"~500s/step)")
    if DEVICE == "cuda" and _off == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and _off == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    # VAE tiling/slicing: essential for img2img/upscale. Qwen-Image is big (~20B
    # transformer + text encoder) -> without tiling the VAE can overflow the VRAM (a spill
    # into shared RAM = very slow). Tiling the VAE caps that peak (like ComfyUI's "tiled
    # decode"). The VAE is shared by the derived pipes.
    try:
        pipe.vae.config.force_upcast = False   # the VAE in bf16 (fp32 is slow on Blackwell) -- ALWAYS
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception as e:
        _dbg(f"VAE tiling not available: {e}")
    _apply_sampler(pipe)   # applies the chosen sampler (euler by default) to the base pipe
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
    """Returns the requested pipeline. Krea 2 only exposes txt2img.

    Safety net: the UI already hides the img2img/inpaint/omni controls through CAPABILITIES,
    but the CLI and direct calls still come through here -> UnsupportedFeature is raised with
    an actionable message rather than leaving a diffusers ImportError or a silent fallback to
    txt2img (which would produce an image unrelated to the input).
"""
    # The refusal comes BEFORE the load: supports() only reads CAPABILITIES (which is
    # static), whereas _ensure_base() downloads and mounts the transformer. Asking for
    # inpaint or omni on this family therefore cost ~26 GB of download only to end on an
    # UnsupportedFeature - the refusal is flat, it does not have to be expensive.
    if kind in _UNSUPPORTED_MSG and not supports(kind):
        raise UnsupportedFeature(_UNSUPPORTED_MSG[kind])
    base = _ensure_base()
    # A sampler/schedule change asked for DURING a render was only recorded
    # (see _reapply_sampler_all): this is where it lands, between two
    # generations, with _GPU_LOCK held by the caller.
    _apply_sampler_if_dirty()
    if kind in _DERIVED:
        _dbg(f"get_pipe('{kind}'): reuse derived")
        return _DERIVED[kind]
    cls = None
    if cls is None:
        return base
    _log(f"deriving {kind} pipeline (shared weights, no extra VRAM)")
    # A GGUF transformer is QUANTIZED: it cannot be recast to a dtype (.to(DTYPE)
    # raises "Casting a quantized model is unsupported"). So the bf16 recast is skipped in
    # that case (the compute_dtype is bf16 already). Otherwise (full bf16): a defensive
    # Blackwell recast (some from_pipe calls upcast to float32 -> very slow without fp32
    # tensor cores).
    quantized = bool(ZIMAGE_TRANSFORMER) and ZIMAGE_TRANSFORMER.lower().endswith(".gguf")
    try:
        # A quantized GGUF: torch_dtype=None EXPLICITLY -> otherwise from_pipe puts
        # float32 by default and casts the quantized model -> ValueError "Casting a
        # quantized model".
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
    _apply_sampler(p)   # the same sampler as the base (in case from_pipe recreates the scheduler)
    # Speed diagnosis: if the derived pipe is NOT on cuda -> img2img/refine runs on the
    # CPU = ultra slow. So it is forced onto DEVICE in full-VRAM mode (offload handles
    # itself).
    # NB: the EFFECTIVE offload (a GGUF base forces 'model' even when the UI says 'none'):
    # under offload, a transformer "on the CPU" is normal -> a .to(cuda) would break the
    # hooks.
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
    """Krea 2 has no instruction-based edit model (no Qwen-Image-Edit equivalent at
    Krea). Kept for API compat: the corresponding tab is hidden by the UI through
    CAPABILITIES['omni'] = False."""
    raise UnsupportedFeature(_UNSUPPORTED_MSG["omni"])


@_gpu_serial
def generate_omni(refs, prompt, negative, width, height, steps, seed):
    """Qwen-Image-Edit instruction-based editing: edits one (or several, through 2509)
    input image(s) according to the instruction prompt. Keeps upstream's signature (cz_ui).
    width/height are ignored: editing preserves the input image's dimensions."""
    refs = [r.convert("RGB") for r in (refs or []) if r is not None]
    if not refs:
        raise ValueError("Edit needs at least one input image.")
    pipe = get_pipe("omni")
    _log(f"edit: {len(refs)} image(s), {int(steps)} steps, cfg {GUIDANCE:.1f} ...")
    _progress(0.1, f"Editing ({len(refs)} image(s))...")
    _set_slicing(pipe, max(max(r.size) for r in refs))
    t0 = time.time()
    # 2509/Plus takes a list of images; the base revision takes a single one.
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
    """Krea 2 txt2img: generates an image from a prompt. A direct guidance_scale (= the
    guidance slider, ~4.0), ~30-50 steps advised. The negative prompt works thanks to the
    real CFG (see _cfg)."""
    pipe = get_pipe("txt2img")
    w = round_to_multiple(int(width))
    h = round_to_multiple(int(height))
    _log(f"txt2img: {w}x{h}, {int(steps)} steps, cfg {GUIDANCE:.1f} ...")
    _dbg(f"txt2img seed={seed} dtype=bf16 device={DEVICE} offload={OFFLOAD_MODE} "
         f"transformer={'single-file' if ZIMAGE_TRANSFORMER else 'repo'}")
    if DEVICE == "cuda":
        _dbg(f"VRAM before: alloc={torch.cuda.memory_allocated()/1024**3:.2f} Go")
    _progress(0.1, f"Generating {w}x{h} ({int(steps)} steps)...")
    t0 = time.time()
    # Two attempts at most: when the VRAM guard fires at the first step ('none' mode
    # too optimistic), _consume_vram_downgrade switches to 'model' and we replay.
    for _attempt in (0, 1):
        _set_slicing(pipe, max(w, h))
        img = _qwen_call(
            pipe,
            prompt=prompt or "",
            width=w, height=h,
            num_inference_steps=int(steps),
            generator=_make_generator(seed),
            **_cfg(negative_prompt),
            **_vram_guard_kwargs(),
        ).images[0]
        if not _consume_vram_downgrade():
            break
        pipe = get_pipe("txt2img")   # reload with the downgraded offload
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
    """Sets the forced ratio for upscale/img2img: 'W:H' / 'WxH' (e.g. '13:19',
    '832x1216') or '' to turn it off (the native ratio is preserved). Driven by the UI
    radio."""
    global FORCE_RATIO
    FORCE_RATIO = (spec or "").strip()
    _log(f"force ratio -> {FORCE_RATIO or '(off, ratio natif preserve)'}")


def set_force_ratio_mode(mode):
    """'crop' (recadrage centre) ou 'extend' (outpaint -- indisponible sur Krea 2)."""
    global FORCE_RATIO_MODE
    FORCE_RATIO_MODE = "extend" if str(mode or "").strip().lower() == "extend" else "crop"
    _log(f"force ratio mode -> {FORCE_RATIO_MODE}")


def _parse_ratio(spec):
    """(w, h) from 'W:H', 'WxH', or a '832 x 1216 | 13:19' label; otherwise None."""
    import re
    if not spec:
        return None
    m = re.search(r"(\d+)\s*[:xX×]\s*(\d+)", str(spec))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if a > 0 and b > 0 else None


def _crop_to_ratio(image, ratio_w, ratio_h):
    """Centre-crops the image to the ratio_w:ratio_h ratio, keeping the largest area."""
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur > target:                       # too wide -> cut the sides
        nw = max(1, int(round(h * target)))
        x0 = (w - nw) // 2
        return image.crop((x0, 0, x0 + nw, h))
    nh = max(1, int(round(w / target)))    # trop haut -> couper haut/bas
    y0 = (h - nh) // 2
    return image.crop((0, y0, w, y0 + nh))


def _extend_to_ratio(image, ratio_w, ratio_h, prompt, steps, seed):
    """Brings the image to the target ratio by EXTENDING it (outpaint) instead of
    cropping. On Krea 2 there is NO inpaint pipeline: outpaint_directions raises the clear
    UnsupportedFeature message. Kept for code parity with the family (the UI does not offer
    this mode here; only a force_ratio_mode='extend' in the config leads to it)."""
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur < target:                       # too narrow -> widen left + right
        pad = target * h - w
        return outpaint_directions(image, None, ["left", "right"], prompt, steps, seed,
                                   expand=pad / (2.0 * w))
    pad = w / target - h                   # trop large -> etendre haut + bas
    return outpaint_directions(image, None, ["top", "bottom"], prompt, steps, seed,
                               expand=pad / (2.0 * h))


def _reframe_canvas(image, ratio_w, ratio_h, overlap=8):
    """Places the image in a larger canvas at the target ratio (expansion on 1 axis),
    + a mask (white = to fill, black = to keep, with a small overlap)."""
    from PIL import ImageDraw
    image = image.convert("RGB")
    w, h = image.size
    r = ratio_w / ratio_h
    # Aligned on 32 (patch 2 x VAE 16): avoids conv errors (no engine).
    if w / h < r:  # trop etroit -> elargir
        nw, nh = round_to_multiple(int(round(h * r)), 32), round_to_multiple(h, 32)
    else:          # too wide -> grow in height
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
    """Inpaint: regenerates the white area of the mask according to the prompt
    (ZImageInpaintPipeline). background + mask = PIL (L: white = to change)."""
    orig = background.convert("RGB")
    full_mask = mask
    # Diffusion bounded to ~1 MP (the model's sweet spot), then recomposed at full
    # resolution.
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
    # Recompose: outside the mask keeps the full resolution; the join is feathered.
    out = _composite_back(out, orig, full_mask, orig_size,
                          feather=max(2, int(min(orig_size) * 0.01)))
    _log(f"inpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# The Z-Image model's "sweet spot" target resolution (~1 MP, like the txt2img ratios).
# The reframe aims at that budget so as NOT to blow up the pixel count (a 2-3 MP output that
# leaves the training zone -> slow and degraded quality).
MODEL_TARGET_PX = 1024 * 1024


def _ratio_canvas(ratio_w, ratio_h, target_px=MODEL_TARGET_PX):
    """A canvas's size (multiples of 32) at the given ratio, around target_px pixels."""
    r = float(ratio_w) / float(ratio_h)
    nh = (target_px / r) ** 0.5
    nw = nh * r
    return round_to_multiple(int(round(nw)), 32), round_to_multiple(int(round(nh)), 32)


def _cap_work_res(image, mask, max_px=MODEL_TARGET_PX):
    """Bounds the working resolution for the diffusion: when image > max_px, returns a
    reduced version (multiples of 32) of (image, mask) + the original size to recompose
    afterwards. Avoids running the model far above its sweet spot (~1 MP) -> faster and
    better quality."""
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
    """Recomposes at the original resolution: the masked area (white) comes from `result`
    (scaled back up to orig_size), the rest from `original` -> everything outside the mask
    keeps the starting image's full resolution. `feather` (px) blurs the mask to blend the
    join (a gradual original <-> generated transition, no hard line)."""
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
    """Crops the image to the target ratio while bounding the output to the model's sweet
    spot (~1 MP) -> no more pixel-count explosion.
      fit='contain' : the whole image fits inside the canvas (without enlarging it), and the
                      added edges are filled by Z-Image (outpaint).
      fit='cover'   : the image fills the canvas at the ratio then is center-cropped (no
                      outpaint, a plain reframe/crop).
"""
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
    # contain -> the original is fitted without being enlarged, then the edges are
    # outpainted.
    from PIL import ImageFilter
    scale = min(nw / w, nh / h, 1.0)
    rw2, rh2 = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((rw2, rh2), Image.LANCZOS) if (rw2, rh2) != (w, h) else img
    ox, oy = (nw - rw2) // 2, (nh - rh2) // 2
    # Edges = a blurred extension of the edge colours (blurred edge fill, like the
    # outpaint) rather than a grey -> exposure continuity; it shows through when
    # strength < 1.0.
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
    """Compat (CLI --reframe and existing calls): a reframe in 'contain' mode (outpaint),
    bounded to the model's sweet spot."""
    return reframe(image, ratio_w, ratio_h, "contain", prompt, steps, seed)


def outpaint_directions(image, mask, directions, prompt, steps, seed, strength=1.0, expand=0.3):
    """Directional outpaint (Fooocus-style): enlarges the image in the chosen directions
    among left/right/top/bottom, each by `expand` (a fraction of the original dimension), by
    replicating the edge pixels (mode 'edge'), then has the added bands filled by Z-Image
    (ZImageInpaintPipeline). A painted `mask` (L, white = to change) is optional: it is kept
    over the original area and combined with the added bands (white)."""
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
    # Dilate the area to generate a little towards the inside -> the model regenerates
    # a thin transition band that joins up with the original (avoids a hard seam).
    from PIL import ImageFilter
    k = max(3, (int(min(full_size) * 0.02) // 2) * 2 + 1)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(min(k, 15)))
    # "Blurred edge fill": the area to generate is filled with a BLURRED version of
    # the edge extension (the same colours/tone as the original) instead of a sharp
    # replicated edge. With strength < 1.0 that blur shows through -> exposure continuity
    # (no lighter band any more) and the model adds the detail on top.
    blur_r = max(8, int(min(full_size) * 0.03))
    canvas = Image.composite(canvas.filter(ImageFilter.GaussianBlur(blur_r)), canvas, mask_img)
    # Diffusion bounded to ~1 MP (the sweet spot), then recomposed: the centre (the
    # original image) keeps its full resolution, only the added edges are generated.
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
    """A Qwen-Image img2img pass over the whole image (or over one tile). The slicing is
    set according to the size really processed: a 1024 tile -> OFF (fast), whole 2K+ -> ON.
    IMPORTANT: width/height = the image's size (aligned on 16) are passed. Otherwise
    Qwen-Image img2img falls back to its default (height = default_sample_size *
    vae_scale_factor = 1024) and RESIZES the input to 1024x1024 -> the ratio is crushed (a
    bug). Forcing the input's dimensions preserves the original ratio in upscale/img2img."""
    w = round_to_multiple(image.width, 16)
    h = round_to_multiple(image.height, 16)
    # Two attempts at most: the VRAM guard at the first step (see generate), then a retry in 'model'.
    for _attempt in (0, 1):
        _set_slicing(pipe, max(image.size))   # a reposer sur le pipe recharge du retry
        out = _qwen_call(
            pipe,
            prompt=prompt or "",
            image=image,
            width=w, height=h,
            strength=float(denoise),
            num_inference_steps=int(steps),
            generator=_make_generator(seed),
            **_cfg(None),
            **_vram_guard_kwargs(),
        ).images[0]
        if not _consume_vram_downgrade():
            return out
        pipe = get_pipe("img2img")   # reload with the downgraded offload
    return out


def _feather_mask_np(th, tw, overlap, left, right, top, bottom):
    """A (th, tw, 1) mask with a linear ramp on the edges that adjoin another tile."""
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
    """A Z-Image pass in tiles with feathered recomposition (Ultimate SD Upscale style).
    Caps the VRAM peak (one tile at a time) and makes 4K+ possible without seams.
    The same linear ramp + overlap-add as esrgan_upscale, but at scale 1 on PIL."""
    w, h = image.size
    tile = round_to_multiple(tile)                       # a multiple of 16 for the VAE
    overlap = max(0, min(int(overlap), tile - 16))
    if w <= tile and h <= tile:
        # A single tile = the whole image -> no duplication possible: the requested
        # denoise.
        return _refine_whole(pipe, image, denoise, steps, prompt, seed)
    # Anti-duplication 1: an empty prompt per tile (the global prompt describes the
    # whole composition).
    prompt = _tile_prompt(prompt)
    if not (prompt or "").strip():
        _log("refine tiled: empty prompt per tile (anti-duplication; rule refine_tile_prompt).")
    # Anti-duplication 2 (a safety net): at a high denoise each tile can still drift.
    denoise = float(denoise)
    if _TILE_DENOISE_CAP > 0 and denoise > _TILE_DENOISE_CAP:
        _log(f"refine tiled: denoise {denoise:.2f} > the cap {_TILE_DENOISE_CAP:.2f} -> "
             f"lowered to {_TILE_DENOISE_CAP:.2f} (refine_tile_denoise_cap rule).")
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
# Orchestration: process_one, the txt2img batch (run/_gen_meta stay in app.py because
# run emits gr.Error for the UI).
# ----------------------------------------------------------------------------
@_gpu_serial
def process_one(image, esrgan_model, factor, denoise, steps, prompt, seed, tile, overlap,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                do_esrgan=True, refine_first=False, apply_force_ratio=False):
    """Pipeline over one PIL Image, returns (image, timings_dict).
    do_esrgan=False -> pure img2img (skips the ESRGAN stage, refines the native image).
    refine_first=True -> refine THEN ESRGAN (the diffusion runs at the native resolution =
    far faster), instead of ESRGAN THEN refine (detail at high resolution).
    apply_force_ratio=True + FORCE_RATIO set -> brings the INPUT to the chosen ratio before
    processing: FORCE_RATIO_MODE 'crop' = a center crop (Fooocus-style), 'extend' = an
    outpaint (unavailable on Krea 2 -> UnsupportedFeature). Otherwise: the native ratio.
"""
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
        # The target = the factor applied to the original size (order-independent).
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
        # Anti-crash guard rail: a whole-image refine that is too large (4K+) -> auto-tiling.
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
        # refine on the native image (fast) then the ESRGAN enlargement.
        if do_refine:
            result = _refine_stage(result)
        if use_esrgan:
            result = _esrgan_stage(result)
    else:
        # the classic order: ESRGAN (the detailer) then refine at the enlarged resolution.
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
    """Generates an image (Z-Image txt2img) then, when upscale=True, runs it through the
    ESRGAN + refine pipeline. Returns (image, timings_dict)."""
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


# ----------------------------------------------------------------------------
# INPUT image(s) in the metadata. An img2img, an inpaint or an edit is defined as much by
# its input as by its prompt: without it, the file does not reproduce from itself. Carried
# over from crispz-klein 1.32.0.
#
# The NAME by default, not the path. The PNG travels (Civitai, forums, a client) whereas the
# sidecar stays local: a full path would export the disk's tree and the Windows session name
# with it. And on the UI side it would be worth nothing anyway -- Gradio drops uploads in a
# temporary folder where only the BASE NAME carries the file's original name. 'full' only
# makes sense for inputs taken from a folder.
# ----------------------------------------------------------------------------
METADATA_SOURCE = str(CONFIG.get("metadata_source", "name") or "name").strip().lower()


def _source_path_of(x, _depth=0):
    """File path of an image input, or None when it cannot be known.

    Accepts a path, a PIL opened from a file (.filename), or a gr.ImageEditor value
    ({background, composite, layers}). The background is tried BEFORE the composite:
    after a crop the composite is a brand new image, with no name.
"""
    if not x or _depth > 2:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("path", "name", "background", "composite", "image"):
            p = _source_path_of(x.get(k), _depth + 1)
            if p:
                return p
        return None
    p = getattr(x, "filename", None)
    return p if isinstance(p, str) and p else None


def source_meta(items, key="source"):
    """A metadata fragment naming the input image(s), or {} when it is not known.
    Nothing rather than an invented name: a wrong piece of metadata is worse than an absent
    one."""
    if METADATA_SOURCE in ("off", "none", "no", "0", "false"):
        return {}
    full = METADATA_SOURCE in ("full", "path", "abs")
    vals = []
    for it in (items if isinstance(items, (list, tuple)) else [items]):
        p = _source_path_of(it)
        if p:
            vals.append(os.path.abspath(p) if full else os.path.basename(p))
    if not vals:
        return {}
    return {key: vals[0] if len(vals) == 1 else vals}


def _gen_meta(mode, prompt, negative="", seed=None, steps=None, guidance=None,
              size=None, model=None, styles=None, extra=None):
    """Builds the generation metadata dict (for the sidecar/PNG)."""
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
    # Names of the applied styles (on top of the keywords already injected into the
    # prompt).
    _styles = [s for s in (styles or []) if s and s not in ("None", "none")]
    if _styles:
        m["styles"] = _styles
    m["sampler"] = f"{SAMPLER}/{SCHEDULE}"
    m["model"] = model or (ZIMAGE_TRANSFORMER or BASE_REPO)
    # A single-file only replaces the TRANSFORMER: the VAE, the text encoder and the
    # architecture config come from the base repo. Without it, the image is not reproducible
    # from its own file.
    if ZIMAGE_TRANSFORMER:
        m["base_repo"] = BASE_REPO
    # A replacement encoder: the one that REALLY ran, by its folder name -- with or
    # without a transformer override, independently of base_repo above. Requested but
    # dropped at load time = the image comes from the base repo's encoder, and the one that
    # did not serve is named separately.
    if _TEXT_ENCODER_ACTIVE:
        m["text_encoder"] = _encoder_label(_TEXT_ENCODER_ACTIVE)
    elif TEXT_ENCODER:
        m["text_encoder_not_applied"] = _encoder_label(TEXT_ENCODER)
    # What was REALLY applied, not what was asked for: a LoRA can be dropped along the
    # way (a missing file, a refused format), and signing an image with a LoRA it does not
    # carry is a quiet lie -- the worst kind.
    # A LoKr is MERGED into the weights: it appears in no PEFT adapter, so without
    # _APPLIED_LOKRS it vanished from the metadata entirely.
    applied = list(_APPLIED_LORAS) + list(_APPLIED_LOKRS)
    if applied:
        m["loras"] = [f"{os.path.basename(p)}@{w}" for p, w in applied]
    missing = [pw for pw in LORAS if pw not in applied]
    if missing:
        m["loras_not_applied"] = [f"{os.path.basename(p)}@{w}" for p, w in missing]
    if extra:
        m.update(extra)
    return m
