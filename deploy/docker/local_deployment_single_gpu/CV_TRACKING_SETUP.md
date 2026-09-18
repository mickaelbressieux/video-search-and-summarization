# Enabling CV Detection + Tracking (Set-of-Marks) in VSS 2.4.1

**Deployment:** `deploy/docker/local_deployment_single_gpu` (docker compose)
**Backend:** `:8100`  · **Frontend:** `:9100`
**Box:** 2× H100 NVL (95 GB each). VSS pinned to **GPU 1** (`NVIDIA_VISIBLE_DEVICES=1`).
**VLM:** remote `nvidia/cosmos3-nano-reasoner` via openai-compat at `:38011` (not a local vLLM).
**Last updated:** 2026-09-18

---

## Goal

Enable object **detection + tracking** so each person gets a stable **tracking ID** that is
drawn onto the frames sent to the VLM (Set-of-Marks / SoM overlay). Intended use: let the
model distinguish individual people across a clip (e.g. "which person did the sequence fastest").

Decisions made:
- GDINO detector model: **auto-download from NGC** (uses `NGC_API_KEY` in `.env`).
- Detect classes: **`person`** only (CV prompt = `person`).
- Keep VSS on **GPU 1 only**.

---

## How it works (mental model)

1. CV pipeline = **GroundingDINO** (open-vocab detector) + **NvDCF tracker** (+ ReID, + SAM2 build).
2. Tracker assigns each detected person a stable **object ID**.
3. At caption time, the frame getter overlays those IDs onto the frames given to the VLM
   (`src/vlm_pipeline/video_file_frame_getter.py` → `_create_osd_pipeline` → `modify_osd_meta`).
   This runs whenever `chunk.cv_metadata_json_file` is set — **independent of VLM type**, which is
   why it works with the remote openai-compat VLM.

---

## Step-by-step: what was done

### 1. Turn the CV pipeline on (`.env` only — no compose edits)

`compose.yaml` already declares every CV env var and the tracker-config mount with safe
defaults, so enabling CV is purely an `.env` change (`.env.bak` backup exists).

```bash
export DISABLE_CV_PIPELINE=false        # turn the CV pipeline ON
export INSTALL_PROPRIETARY_CODECS=true  # installs ffmpeg_for_overlay_video (SoM overlay artifact)
export NUM_CV_CHUNKS_PER_GPU=1          # single-GPU: keep CV memory footprint small
# GDINO_MODEL_PATH: leave UNSET -> auto-downloads
#   ngc:nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0
# GDINO_INFERENCE_INTERVAL: unset (default 1). Set 0 for max accuracy (heavier).
# CV_PIPELINE_TRACKER_CONFIG: unset -> container default /opt/nvidia/via/config/default_tracker_config.yml
```

### 2. Fix the GPU memory contention (the real blocker)

Both GPUs were oversubscribed by NIM containers; GPU 1 had only ~8–10 GB free, so the
TensorRT engine builds and the ReID engine load hit OOM:

- `[TRT] Requested amount of GPU memory (...) could not be allocated`
- `[NvMultiObjectTracker] ... Failed to allocate memory to create engine`

**Root cause of the memory hog:** the cosmos NIM reserved a huge KV cache.
`NIM_KVCACHE_PERCENT` is a **no-op** for this NIM — the effective knob is
**`NIM_GPU_MEMORY_UTILIZATION`** (`/opt/nim/inference.py:43`).

Relaunched the cosmos NIM with:
```
NIM_GPU_MEMORY_UTILIZATION=0.50
NIM_MAX_MODEL_LEN=32768
```
Result: **GPU 1 free went from ~10 GB → ~81 GB.**

### 3. Clear the 0-byte engine files (the hidden trap)

The earlier OOM killed `trtexec` mid-build, leaving **0-byte** engine files in the cache.
The build code only rebuilds when a file is **absent**
(`src/cv_pipeline/cv_pipeline.py:286` → `if not os.path.exists(engine_file):`), so the empty
files were being treated as "already built" and never regenerated — the pipeline kept
loading empty engines and failing.

Deleted only the empty files (safe — they regenerate):
```bash
docker exec local_deployment_single_gpu-via-server-1 \
  find /root/.via/ngc_model_cache/cv_pipeline_models -type f -size 0 -print -delete

docker exec local_deployment_single_gpu-via-server-1 \
  find /opt/nvidia/TritonGdino/triton_model_repo/gdino_trt/1 -name model.plan -size 0 -print -delete
```
Removed: `swin.fp16.engine` (GDINO), `resnet50_market1501_aicity156.onnx.engine` (ReID),
4× SAM2 engines, and the empty `model.plan`.

### 4. Recreate via-server → rebuild engines (with memory now free)

```bash
cd deploy/docker/local_deployment_single_gpu
docker compose up -d --force-recreate via-server
docker compose logs -f via-server
```
First CV-enabled start is slow (~20–30 min): downloads GDINO from NGC + SAM2 checkpoints,
`pip install sam2-onnx-tensorrt`, builds 6 TRT engines in parallel, copies the swin engine to
`gdino_trt/1/model.plan`. Engines are cached in the `via-ngc-model-cache` volume, so later
restarts are fast.

---

## Verification (read-only)

```bash
# engines now non-zero:
docker exec local_deployment_single_gpu-via-server-1 \
  ls -la /root/.via/ngc_model_cache/cv_pipeline_models/
# GDINO plan populated:
docker exec local_deployment_single_gpu-via-server-1 \
  ls -la /opt/nvidia/TritonGdino/triton_model_repo/gdino_trt/1/model.plan
# server ready:
curl -s http://localhost:8100/health/ready
```

In the UI (`:9100`) with `DISABLE_CV_PIPELINE=false` an **"enable CV metadata"** checkbox and a
**CV prompt** box appear — check the box, set the prompt to **`person`**, and run a summarize on a
multi-person clip. The **Set-of-Marks Preview** tab shows the numbered overlay video.

API equivalent (on `:8100`): request body must include
`{"enable_cv_metadata": true, "cv_pipeline_prompt": "person"}`.

---

## Known limitation: the chat/Q&A agent does not know the tracking IDs

**Symptom:** the SoM video shows numbered people, but the UI chat can't answer questions about
"person 3", etc.

**Why:** the tracking IDs are only drawn **visually** on the frames. Nothing converts them into
**text**:
1. The default caption/summarization prompt never mentions the numbered markers, so the VLM is
   not told to describe people by ID.
2. The one code path that explicitly extracts SoM IDs into text
   (`src/via_stream_handler.py:1711`) is gated to **`COSMOS_REASON1`** only. This deployment's
   VLM is `cosmos3-nano-reasoner` (openai-compat), so that path is skipped.
3. Chat uses **CA-RAG**, which retrieves over the **stored text captions** — not the overlaid
   video. If the captions don't contain the IDs, chat can't surface them.

**Fix path (prompt-level, not code):** set the summarization/caption prompt to explicitly
instruct the VLM to use the on-frame numbers, e.g.:

> "Each person in the frame is labeled with a number. Always refer to people by their number
> (e.g. 'Person 3'). Describe what each numbered person does and when."

That makes the IDs land in the captions, so CA-RAG (and the chat) then has them. Note:
`cosmos3-nano-reasoner` is small and may not reliably read overlay digits — validate on a clip.

---

## Rollback

Restore `.env` from `.env.bak` (or `export DISABLE_CV_PIPELINE=true`) and
`docker compose up -d --force-recreate via-server`. Cached engines stay in the volume and are
inert when CV is off.
