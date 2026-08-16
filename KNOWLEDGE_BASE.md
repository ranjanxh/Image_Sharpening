# Knowledge Base — Image Sharpening via Knowledge Distillation

Reference document for this repository's actual state. Written from direct code/test inspection, not from README/report claims. Update this file when the underlying code changes — don't let it drift into aspirational documentation.

**Last updated**: 2026-08-16, after the Phase 0 (bug fixes) + Phase 1 (package refactor) hardening pass. Everything in this revision has been verified by actually running it in this environment (CPU-only, no GPU available here) — see §6 for the exact proof.

---

## 1. What this project is

Teacher-student knowledge distillation for image sharpening/deblurring:
- **Teacher**: pretrained **Restormer** (transformer-based image restoration, motion deblurring weights), frozen.
- **Student**: custom **U-Net + Squeeze-Excitation blocks**, trained to mimic the teacher and match ground truth.
- **Target use case** (per project report): real-time video-conferencing image sharpening under low bandwidth.
- **Dataset**: GoPro deblurring dataset (Nah et al., CVPR 2017), restructured into `train/{blurred,sharp}` and `test/{blurred,sharp}`.

As of this revision, the pipeline is a real Python package under `src/`, not a single script. The original monolith is preserved at `legacy/main_sharpening_script.py` for historical reference only (with its two fatal bugs fixed, nothing else changed) — see §5.

## 2. Current file/directory map

```
Image_Sharpening/
├── src/                               # the actual package — see §3
│   ├── config.py                      # YAML config loading, dataclasses, path resolution
│   ├── models/{teacher.py,student.py}
│   ├── data/dataset.py                # PairedImageDataset — dataset-agnostic
│   ├── losses.py                      # CombinedLoss, all 5 terms independently toggleable
│   ├── train.py                       # training loop, EMA, cosine annealing, metrics logging
│   ├── evaluate.py                    # PSNR/SSIM evaluation + tiled inference
│   └── benchmark.py                   # FPS measurement + ONNX export helper (not yet run at scale)
├── configs/default.yaml               # single source of truth for paths/hyperparams — no
│                                       # hardcoded absolute paths anywhere in src/
├── scripts/{train.sh,evaluate.sh}
├── tests/                             # pytest; 14 tests, all passing (see §6)
├── legacy/main_sharpening_script.py   # original monolith, superseded, kept for reference
├── requirements.txt, requirements-dev.txt
├── pyproject.toml                     # pytest + ruff config
├── .gitignore                         # NEW — does not retroactively untrack already-committed
│                                       # large files (see §8)
├── README.md                          # STILL THE OLD ONE — not yet rewritten (waiting on a
│                                       # real training run; see §9)
├── Project_Report_Intel_Summer_Training_compressed.pdf
├── Restormer/                         # git submodule, NOW INITIALIZED (was empty before)
├── GoPro_dataset/train,test/{blurred,sharp}/   # 702 / 186 pairs, unchanged
├── model_checkpoints/student_model_ema_final_epoch_050.pth   # SEE §4 — NOT a valid student checkpoint
├── inference_samples/sharpened_test_outputs/   # 186 images, unchanged, provenance still unclear
└── temp_fps_files/                    # leftover from an old run, harmless
```

## 3. Architecture — as implemented in `src/`

### Teacher (`src/models/teacher.py`, `RestormerTeacherWrapper`)
- Loads the real Restormer class straight from the vendored submodule source via `runpy.run_path` (no `basicsr` package install needed — the arch file only imports `torch`/`einops`).
- Verified: loading `motion_deblurring.pth` into the instantiated architecture gives **zero missing, zero unexpected keys** (strict `load_state_dict`). This is a real, correctly-configured Restormer, not the identity-passthrough fallback.
- **Fixed bug**: the original hooked a nonexistent `encoder_level4` attribute. The real architecture only has `encoder_level1/2/3` followed directly by the `latent` bottleneck — there is no 4th encoder stage. That hook silently never fired in every prior version of this code, so "deep encoder" feature distillation never actually happened, ever. This version hooks `encoder_level3` instead (the real deepest encoder stage) with a correctly-sized projection conv (`dim*4` input channels, not the incorrect `dim*8` the original assumed).
- Wrapped in `@torch.no_grad()` — the teacher can never accidentally require/accumulate gradients regardless of caller context.

### Student (`src/models/student.py`, `StudentModel`)
- Same 4-level U-Net + SE architecture as before, `base_channels=256` (config-driven).
- **Fixed latent bug**: `SEBlock` used `channel // reduction` unguarded — with small channel counts (e.g. a 16-channel test model with reduction=16) this floors to 0, creating a zero-width bottleneck (`nn.Linear(16, 0)`), which PyTorch accepts silently but is dead weight. Now `max(1, channel // reduction)`. Doesn't affect the production config (channels there are always ≥256), but was a real landmine for anyone reusing this block at smaller widths.

## 4. Checkpoint provenance — critical finding

**`model_checkpoints/student_model_ema_final_epoch_050.pth` is byte-for-byte identical (MD5 `bb3ee694...`) to the real Restormer teacher weights (`motion_deblurring.pth`, downloaded fresh from the official release and independently verified to match).**

This is not a trained student model. It is either a copy-paste/upload mistake, or a placeholder that was never replaced with a real trained checkpoint. Consequences:
- It cannot even be loaded into `StudentModel` — the architectures are completely different (a Restormer transformer vs. a U-Net), so `load_state_dict` would raise immediately on a key mismatch.
- Any historical claim that this checkpoint achieved 0.91 SSIM / 28.4 dB PSNR / 45 FPS is therefore **impossible** — those numbers, if measured at all, were not measured on this file.
- There is currently **no valid trained student checkpoint anywhere in this repository.** A real one will only exist after a real training run (see §9).

## 5. Bugs fixed this pass

All three were confirmed present in the original `main_sharpening_script.py` and are now fixed in both `legacy/main_sharpening_script.py` (minimally, for historical accuracy) and properly in `src/`:

1. **Missing `import sys`** — used at `sys.path.insert(...)` but never imported. Immediate `NameError`.
2. **Wrong variable names in `RestormerTeacherWrapper.forward()`** — computed `teacher_features_e2_raw` etc. but referenced undefined `teacher_e2_feat_raw` etc. Immediate `NameError` on the teacher's first forward pass.
3. **`encoder_level4` hook never fires** (see §3) — silent, not crashing, but meant one of three intended feature-distillation taps was always dead code. Retargeted to `encoder_level3` in `src/models/teacher.py`.

Additionally initialized the previously-empty `Restormer/` git submodule (`git submodule update --init --recursive`) and downloaded the real `motion_deblurring.pth` from `https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth` (confirmed working direct-download URL, ~104.7MB, loads with an exact key match).

## 6. Verification performed (what was actually run, in this CPU-only environment)

- Real Restormer instantiated + real weights loaded, strict `load_state_dict`, zero missing/unexpected keys.
- Full `RestormerTeacherWrapper` forward pass on a real tensor, all three feature taps returning correctly-shaped tensors (e2, deep/e3, bottleneck).
- Full `StudentModel` forward pass, real backward pass through all 5 real loss terms (L1, VGG19 perceptual, feature-distillation L1, KL-divergence, SSIM) computed together as `combined_loss`, gradients confirmed reaching student parameters.
- **`python -m src.train`** run end-to-end for 1 real epoch on a 6-image train / 3-image test subset of the real GoPro dataset (64×64 resolution, small student width — chosen only to keep CPU wall-clock short; the code path is identical to the full-scale config), producing real logged metrics to both CSV and JSONL, and real checkpoint files. Output included, e.g.: `val_ssim: 0.1600, val_psnr: 8.61` (meaningless as a quality number on 1 epoch / 6 images / 64px, but proves the full loop — teacher forward, student forward/backward, EMA update, scheduler step, checkpoint save, metrics logging — genuinely executes without error).
- `pytest tests/` — **14/14 passed**, covering: student forward-pass shapes (square + non-square input), teacher forward-pass shapes with real weights, teacher is frozen, all 5 loss terms individually (each enabled/disabled path, numeric match against hand-computed reference), checkpoint save/load round-trip (bit-exact output before/after reload).
- `ruff check src/ tests/` — clean, 0 issues (3 unused imports were caught and fixed).
- Import-only smoke test confirms no module in `src/` has side effects on import (training/data-loading only happens inside `main()` / explicit function calls).

No GPU is available in this environment (`nvidia-smi` cannot find a driver), so none of the above numbers are from a real, full-scale, meaningful training run — that step still needs to happen on a GPU machine (see §9).

## 7. Loss functions, training config — unchanged in substance, now config-driven

Same 5 terms and same defaults as originally designed (L1=1.0, perceptual=0.01, feature-distillation=1.0, KL-div=0.01 @ T=4.0, SSIM=10.0; Adam lr=2e-4; CosineAnnealingWarmRestarts T_0=10/T_mult=2/eta_min=1e-6; EMA decay=0.999; grad accumulation ×4; 50 epochs) — all now live in `configs/default.yaml` rather than hardcoded, and each loss term has an `enabled: true/false` flag (`src/losses.py`) for the ablation study this project still needs to run.

## 8. What's still outstanding

- [ ] **Real training run** on a GPU (this environment has none) — full 50-epoch run on the full 702/186 GoPro split, with real logged metrics.
- [ ] **Ablation study** (teacher alone / no-distillation / full distillation / minus-KL / minus-feature-distillation), same data split and seed, PSNR/SSIM/params/FPS table.
- [ ] **Real FPS benchmarking chain** (teacher → student → ONNX-exported → fp16-quantized), replacing all prior unverified numbers.
- [ ] **README rewrite** with the real ablation table, real benchmark chain, real sample images, and an explicit Limitations section — deferred until the above produce real numbers to report.
- [ ] **CI** (ruff + pytest on push) — trivial to add, deferred to bundle with the README rewrite/final packaging pass.
- [ ] **Untrack large binaries from git** — `.gitignore` now prevents *new* large files from being tracked, but `model_checkpoints/student_model_ema_final_epoch_050.pth` (the mislabeled teacher-weights copy, see §4) and the full `GoPro_dataset/` are already committed to git history. Removing them from tracking (`git rm --cached`) or rewriting history is a decision for whoever owns the remote — not done automatically here since it changes what's pushed to `origin`.
- [ ] Decide what to do with `model_checkpoints/student_model_ema_final_epoch_050.pth` — it should probably be deleted or clearly relabeled once a real checkpoint exists, to stop anyone mistaking it for a trained model again.

## 9. Recommended next steps (handoff)

1. On a machine with a GPU: `pip install -r requirements.txt` (swap the CPU torch index URL for your CUDA version — see the comment at the top of `requirements.txt`), `git submodule update --init --recursive`, download `motion_deblurring.pth` per the URL in §5.
2. `python -m src.train --config configs/default.yaml` — trains for the full 50 epochs on the full dataset, logs to `logs/student_kd_metrics.{csv,jsonl}`, saves both EMA and raw checkpoints to `model_checkpoints/`.
3. `python -m src.evaluate --config configs/default.yaml --checkpoint model_checkpoints/student_kd_ema_final_epoch_050.pth --save-images` for real, final PSNR/SSIM numbers on the held-out test set.
4. Come back for the ablation study (toggle `losses.enabled.*` in the config per run) and the benchmark chain (`src/benchmark.py`) once real checkpoints exist.

## 10. Dataset-agnostic vs. GoPro-specific — for a future document/receipt-data swap

**Dataset-agnostic** (no changes needed): `src/models/`, `src/losses.py`, `src/train.py`, `src/evaluate.py`, `src/benchmark.py` — none of these reference GoPro by name or assume anything about image content.

**Config-only change** (edit `configs/default.yaml`, no code touched): dataset root and subdirectory paths (`data.*_subdir`), image size, batch size, filename stem-suffix stripping rules (`data.blurry_stem_suffixes` / `sharp_stem_suffixes` — GoPro's `_blurred`/`_sharp` naming quirk; set to `[]` for datasets with exact-matching filenames).

**Would need actual new code**: `src/data/dataset.py`'s `PairedImageDataset` assumes the new dataset also ships as paired blurry/sharp image directories with matchable filenames — if document/receipt data instead needs synthetic blur generation, different augmentation (e.g. no random crop for documents where layout matters), or a different pairing scheme, that's new code, not a config change. No document-specific logic exists anywhere in this codebase yet, by design.

---
*Source: direct inspection and execution of the code in this repository, in a CPU-only sandboxed environment, 2026-08-16. Every claim in this file that isn't marked "unverified"/"claimed" was reproduced by an actual command in this environment.*
