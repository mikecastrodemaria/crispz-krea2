# crispz-krea2 — fork Krea 2 de crispz-studio

Fork **texte → image** basé sur Krea 2 (Krea.ai, 12,9B). Parti de
**crispz-qwen-edit** plutôt que de crispz-studio, parce que Krea 2 partage avec
Qwen-Image le VAE (`AutoencoderKLQwenImage`), la famille d'encodeur texte (Qwen3-VL
contre Qwen2.5-VL) et l'alignement des dimensions sur 16.

- Modèle : `krea/Krea-2-Turbo` (défaut) et `krea/Krea-2-Raw`. **Repos gated.**
- Remotes : `origin` = crispz-krea2, `upstream` = crispz-studio, `qwen` = crispz-qwen-edit.

## Ce que ce fork ne sait PAS faire

diffusers n'expose qu'un seul pipeline Krea 2 : `Krea2Pipeline` (txt2img). Il n'y a
ni `Krea2Img2ImgPipeline` ni `Krea2InpaintPipeline`.

| Fonction | État |
|---|---|
| `generate` (txt2img) | OK |
| `process_one`, `txt2img_run(upscale=True)`, `_refine_whole`, `_refine_tiled`, `load_pipe` | `UnsupportedFeature` (img2img) |
| `inpaint_run`, `reframe(contain)`, `outpaint`, `outpaint_directions` | `UnsupportedFeature` (inpaint) |
| `generate_omni` | `UnsupportedFeature` (pas de modèle d'édition) |
| ESRGAN (`cz_esrgan`) | OK — indépendant de la diffusion |
| LoRA | OK — `Krea2Transformer2DModel` hérite de `PeftAdapterMixin` |

L'UI **masque** les onglets et contrôles correspondants. La source de vérité est
`cz_pipeline.CAPABILITIES`, lu par `cz_ui` (`HAS_IMG2IMG` / `HAS_INPAINT` /
`HAS_OMNI`). **Ne pas coder en dur une liste d'onglets cachés dans l'UI** : ajouter
la capacité au dict.

Les composants Gradio concernés sont **construits puis masqués** (`visible=False`),
pas supprimés : ça évite de recâbler des dizaines de handlers. Leurs valeurs sont
forcées à un no-op (refine décoché, denoise 0).

## Ce qui diverge de l'amont — à ne PAS écraser lors d'un merge

**1. Pas de single-file, du tout.** `Krea2Transformer2DModel` n'hérite pas de
`FromOriginalModelMixin` → `from_single_file` n'existe pas. Conséquences :
- les `.safetensors` Civitai (Krea 2 Turbo fp8, Raw fp8 **et** bf16) sont inchargeables ;
- les GGUF communautaires aussi (le chemin GGUF passe par `from_single_file`) ;
- `list_checkpoints()` ne liste que des **dossiers diffusers** (`model_index.json`) ;
- tout le bloc GGUF hérité de qwen-edit (`GGUF_ARCH`, `_gguf_arch`, `GGUFQuantizationConfig`)
  est mort. Ne pas le réactiver depuis l'amont.

**2. La quantification torchao n'est pas optionnelle.** Mesures RTX 5090, 1024×1024, 8 steps :

| mode | VRAM pic | par step | une image |
|---|---|---|---|
| `none` (bf16) | **35,63 Go** — déborde, spill PCIe | 59,17 s | 473 s |
| `float8_weight_only` | **22,82 Go** | 2,4 s | 19,5 s |

`_load_transformer()` charge donc **toujours** le transformer séparément, pour pouvoir
lui appliquer `TorchAoConfig`. Ne pas revenir au `from_pretrained` global de l'amont.

**3. Convention CFG opposée à Qwen.** Krea 2 : `guidance_scale` **direct**
(velocity = `cond + g*(cond-uncond)`, guidance active dès `g > 0`). Qwen :
`true_cfg_scale` + un `guidance_scale` distillé à 1.0. `_cfg()` diverge donc
totalement — c'est un des points les plus faciles à casser en reprenant l'amont.
À `g = 0` (Turbo), le negative prompt n'est pas transmis.

**4. `transformers` 5.x, pas `<5`.** La contrainte s'est inversée :
`diffusers` (main) veut `huggingface-hub >=1.23`, `transformers` 4.x veut `<1.0`.
Ne pas recopier la borne `transformers>=4.51,<5` des autres forks.

**5. Profils Turbo / Raw.** `turbo` est testé **avant** `krea` dans `MODEL_PROFILES`,
sinon `Krea-2-Turbo` tomberait dans le profil générique.

## Workflow de merge

```bash
git fetch upstream && git merge upstream/main      # ameliorations generiques
git fetch qwen     && git merge qwen/main          # le frere le plus proche
```

Résolution des conflits :

| Fichier | Stratégie |
|---|---|
| `cz_pipeline.py` | **ours** + porter à la main les améliorations génériques |
| `cz_ui.py`, `cz_core.py`, `config-sample.txt` | **theirs** + réappliquer le delta du fork |

À porter depuis l'amont : `_load_monitor`, `_apply_loras` / `_APPLIED_LORAS`,
`_lora_weight_range`, `_LAST_SEED` / `_NO_SEED_INCREMENT` / `_SAVE_PRE_UPSCALE`,
job queue, XYZ grid, asset browser.

À **ne pas** porter : tout `QwenImage*` / `ZImage*`, le bloc GGUF, `from_single_file`,
`true_cfg_scale`, la borne `transformers<5`.

## Piège du clonage entre forks

`config.txt` et `preferences.json` sont gitignorés et **suivent le clone**. Au premier
portage ils pointaient encore `Qwen/Qwen-Image` et un `.gguf` — quatre échecs de
validation venant de la config, pas du code. Après un clone, vérifier :
`zimage_model`, `zimage_transformer` (doit être absent), `checkpoints_extra_dir`,
`model_profiles`, `default_guidance`, `zimage_omni_model`, `gguf_arch`.

## Checklist post-merge

```bash
.venv\Scripts\python -m py_compile app.py cz_pipeline.py cz_ui.py cz_core.py
.venv\Scripts\python -c "import cz_ui; cz_ui.build_ui()"
.venv\Scripts\python -c "import cz_pipeline as p; assert p.round_to_multiple(100)==96"
.venv\Scripts\python -c "import cz_pipeline as p; assert p.CAPABILITIES['img2img'] is False"
.venv\Scripts\python -c "import cz_pipeline as p; p.GUIDANCE=0.0; assert 'true_cfg_scale' not in p._cfg('x')"
```

Puis une génération réelle : `cz_pipeline.generate(prompt=..., width=1024, height=1024,
steps=8, seed=7)` → attendu ~19,5 s et ~22,8 Go après chargement.

## Licence

Krea 2 est sous **Krea 2 Community License** : usage commercial plafonné à
1 M$ de CA annuel, **filtrage de contenu obligatoire pour tout déploiement servant
des tiers**, licence révocable avec préavis de 30 jours. Voir [`NOTICE`](NOTICE) et
la section « Krea 2 — gated access & licensing » du README. C'est la seule famille
de la lignée crispz dont la licence contraint la façon de distribuer l'app.
