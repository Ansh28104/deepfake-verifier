# Deepfake Verifier

Real-time deepfake-aware identity verification with dual-mode inference (Light/Heavy), blink-based liveness support, and a fusion verdict engine.

## MVP Highlights (Implemented)

- Hybrid liveness: passive blink/motion scoring plus active challenge prompts.
- Challenge prompts in UI: blink twice, turn head left/right, open mouth.
- Configurable fusion weighting: deepfake and liveness contributions are tunable.
- Smoothed risk scoring to reduce noisy verdict jumps.

## Project Structure

- `main.py`: Entry point and orchestrator loop.
- `config.py`: Runtime settings (mode, thresholds, durations).
- `core/`: Detection, inference, fusion, and camera modules.
- `display/`: Renderer and reusable visual components.
- `models/`: Fine-tuned model weights (`.pth`) to be added.
- `dataset/`: Training and evaluation assets.
- `tests/`: Unit and integration tests.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run:
   - `python main.py`

## Add Fine-Tuned Weights

Place your checkpoints in `models/` with these filenames:

- `models/efficientnet_b4_finetuned.pth`
- `models/vit_b16_finetuned.pth`

If files are missing, the app falls back to base pretrained backbones and logs a startup message.

To verify checkpoint files:

- `python scripts/verify_checkpoints.py`

Download from Hugging Face Hub:

- `python scripts/download_hf_checkpoints.py --light-repo <repo_id> --light-file <file.pth> --heavy-repo <repo_id> --heavy-file <file.pth>`

Note: `InferenceClient.image_classification(...)` calls hosted inference and does not download `.pth` checkpoints.

To force app startup only when fine-tuned weights are loaded, set in `config.py`:

- `require_finetuned_weights = True`

## Continue Fine-Tuning

Train Light model:

- `python scripts/train_finetune.py --mode light --epochs 5 --batch-size 8`

Train Heavy model:

- `python scripts/train_finetune.py --mode heavy --epochs 5 --batch-size 4`

Train with multiple datasets (FaceForensics + Celeb-DF v2):

- `python scripts/train_finetune.py --mode light --data-root dataset/faceforensics --data-root "D:/datasets/Celeb-DF-v2" --epochs 5 --batch-size 8`

Celeb-DF v2 folder support:

- Real videos: `Celeb-real/` and `YouTube-real/`
- Fake videos: `Celeb-synthesis/`
- Detection is recursive, so nested folders are supported.

Both commands save best checkpoints directly into `models/` with app-compatible filenames.

Training now reports per-class metrics each epoch:

- `val_acc`
- `macro_f1`
- real/fake class recall and F1

Saved checkpoints include metadata (`mode`, `epoch`, best metrics, class counts, dataset roots):

- Binary weights: `models/*.pth` (contains `state_dict` + `metadata`)
- Sidecar metadata: `models/*.meta.json`

Optional checkpoint quality gates:

- `--checkpoint-min-val-acc`
- `--checkpoint-min-macro-f1`

## Runtime Controls

- `Q`: Quit
- `M`: Toggle model mode (`light` <-> `heavy`)
- `R`: Reset current verification session

## Benchmark Latency

Run model-only latency benchmark:

- `python scripts/benchmark_latency.py --mode light --frames 120`

Run full pipeline benchmark (model + face + blink + fusion) on a video:

- `python scripts/benchmark_latency.py --mode heavy --video dataset/test/real/sample.mp4 --frames 150 --include-face`

Sample output:

```text
Latency Benchmark
- mode: light
- include_face: False
- samples: 110
- avg_ms: 18.42
- median_ms: 17.90
- p95_ms: 24.63
- p99_ms: 28.11
- est_fps: 54.29
```

## Replay Attack Harness

Use the sample manifest at `docs/replay_manifest.sample.json` and replace video paths with your local files.

Run harness:

- `python scripts/replay_attack_harness.py --manifest docs/replay_manifest.sample.json --mode light --fail-on-mismatch`

Harness writes a JSON report by default to `docs/replay_harness_report.json`.

Sample output:

```text
[PASS] real_user_baseline: expected=VERIFIED actual=VERIFIED risk=21.8 frames=187
[PASS] replay_attack_screen_capture: expected=REJECTED actual=REJECTED risk=82.4 frames=203
[PASS] deepfake_clip: expected=REJECTED actual=REJECTED risk=79.1 frames=198
Replay Harness Summary
- total: 3
- pass: 3
- fail: 0
- skipped: 0
```

## CI Quality Gate

Checkpoint acceptance gate script:

- `python scripts/ci_gate_checkpoints.py --min-val-acc 0.90 --min-macro-f1 0.88`

Behavior:

- Reads `models/*.meta.json` metadata files.
- Fails non-zero if any checkpoint is below thresholds.

GitHub Actions workflow is included at `.github/workflows/quality-gate.yml`:

- runs test suite on push/PR,
- runs checkpoint gate when metadata files are present.

## Liveness & Fusion Tuning

`config.py` includes MVP controls for challenge flow and score fusion:

- `liveness_challenge_enabled`
- `challenge_timeout`
- `challenge_required_successes`
- `challenge_weight`
- `deepfake_weight`
- `blink_weight`
- `risk_smoothing_alpha`

For lower-latency demos, reduce `analysis_duration` and `frame_buffer_size` while keeping `risk_smoothing_alpha` above 0 to avoid unstable risk swings.

## Roadmap Alignment

Current scaffold aligns with the architecture roadmap:

- Separation of concerns by module.
- Config-driven mode switching (`light` / `heavy`).
- Fusion engine with state machine and weighted risk score.
- Display layer separated from core logic.

## Notes

- Model preprocessing and forward pass are implemented. If a loaded model has a non-binary head (for example ImageNet classes), the engine intentionally returns a neutral score (`0.5`) until fine-tuned deepfake weights are connected.
- If `mediapipe` or `torchvision` are unavailable, the app stays in safe fallback behavior instead of crashing.
