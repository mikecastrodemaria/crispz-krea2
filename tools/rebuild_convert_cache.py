"""Reconstruit le cache de conversion (cache/krea2_convert) pour TOUS les
checkpoints single-file du dossier de modeles.

Usage:
    .venv/Scripts/python tools/rebuild_convert_cache.py [--list] [--cpu]
    (ou double-clic sur rebuild_cache.bat a la racine)

- REPRISE GRATUITE: un checkpoint deja converti (cache a jour) est saute en
  une seconde -> relancable a volonte, y compris apres une coupure.
- --list : montre ce qui serait fait, sans rien convertir.
- --cpu  : force la dequantification FP8/INT8 sur CPU (par defaut: GPU si
  libre - la dequant prend ~1 Go de VRAM par tenseur, mais si un rendu tourne
  en meme temps, prefere --cpu ou attends).
- Duree indicative: ~15 min par BF16 (borne par le disque des modeles),
  ~2-3 min par FP8/INT8/GGUF avec la dequant GPU.
- Verifie que convert_cache_max_gb (config.txt) couvre le total (~26 Go par
  checkpoint), sinon les premieres conversions seraient evincees par les
  dernieres.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as czp  # noqa: E402

if "--cpu" in sys.argv:
    czp.CONFIG["convert_device"] = "cpu"

files = []
for d in czp._checkpoint_dirs():
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if not os.path.isfile(p):
            continue
        low = f.lower()
        if low.endswith(".safetensors"):
            bad = czp._safetensors_unsupported(p)
            if bad:
                print(f"SKIP {f}: {bad}")
                continue
            files.append(p)
        elif low.endswith(".gguf"):
            files.append(p)

if not files:
    print("Aucun checkpoint single-file trouve dans:", czp._checkpoint_dirs())
    sys.exit(0)

cap = float(czp.CONFIG.get("convert_cache_max_gb", 80) or 0)
need = len(files) * 26
print(f"{len(files)} checkpoint(s) a couvrir (~{need} Go de cache; "
      f"plafond convert_cache_max_gb = {cap:.0f} Go"
      + (", 0 = illimite)" if cap == 0 else ")"))
if 0 < cap < need:
    print(f"ATTENTION: plafond {cap:.0f} Go < ~{need} Go necessaires -> les "
          f"conversions les plus anciennes seraient evincees par les dernieres."
          f" Monte convert_cache_max_gb dans config.txt avant de continuer.")
    if "--list" not in sys.argv:
        sys.exit(1)

if "--list" in sys.argv:
    for p in files:
        dst = None
        try:
            root = czp._convert_cache_dir()
            done = "?"
            if root:
                # meme calcul de cle que _converted_folder, sans convertir
                import hashlib
                import re
                st = os.stat(p)
                sig = f"{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}"
                key = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
                stem = re.sub(r"[^A-Za-z0-9_-]+", "_",
                              os.path.splitext(os.path.basename(p))[0])[:40]
                dst = os.path.join(root, f"{stem}_{key}")
                done = "DEJA CONVERTI" if os.path.isfile(
                    os.path.join(dst, "source.json")) else "a convertir"
        except Exception as e:
            done = f"? ({e})"
        print(f"  {os.path.basename(p)}: {done}")
    sys.exit(0)

t_all = time.time()
ok = fail = skipped = 0
for i, p in enumerate(files, 1):
    name = os.path.basename(p)
    t0 = time.time()
    try:
        folder = czp._converted_folder(p)
        dt = time.time() - t0
        if dt < 5:
            skipped += 1
            print(f"[{i}/{len(files)}] SKIP {name} (deja en cache)")
        else:
            ok += 1
            print(f"[{i}/{len(files)}] OK {name} en {dt / 60:.1f} min")
    except Exception as e:
        fail += 1
        print(f"[{i}/{len(files)}] FAIL {name}: {type(e).__name__}: {e}")

print(f"\nTermine en {(time.time() - t_all) / 60:.0f} min: "
      f"{ok} converti(s), {skipped} deja en cache, {fail} echec(s).")
print("Relancable a volonte: tout ce qui est fait est saute.")
