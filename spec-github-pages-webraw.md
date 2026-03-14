# WebRAW — Client-Only SPA Spec

## Status: Ready for Implementation
## Owner: Mazhar
## Target: Claude Code execution
## Repo: github.com/mazharm/webraw
## Branch: master

---

## 1. Goal

Convert WebRAW from a client/server architecture (React frontend + Rust/Axum backend) to a **fully client-side SPA** deployable to GitHub Pages. No backend server. All processing happens in the browser via WebAssembly and WebGL.

### What Changes

| Concern | Current (backend) | Target (client-only) |
|---------|-------------------|----------------------|
| RAW decode + demosaic | Rust on server (`rawloader`) | Rust→WASM in browser (`rawloader`) |
| Edit pipeline (stages 1-9) | Rust on server (`render.rs`) | Rust→WASM in browser |
| Edit pipeline (stages 10-17) | WebGL shaders exist but unused; Canvas2D fallback to server | WebGL shaders (already written) |
| File storage | Server ephemeral cache (`FileCache`) | IndexedDB via `idb-keyval` (already a dependency) |
| Thumbnails | Server generates JPEG | WASM generates in-browser |
| Export | Server renders full pipeline → file download | WASM renders full pipeline → Blob download |
| AI editing | Server proxies to Gemini/OpenAI | Browser calls APIs directly (keys already per-request) |
| ML features (auto-enhance, optimize) | ONNX Runtime on server | Deferred to Phase 3 (`onnxruntime-web`) |
| Health check / session tokens / jobs | Server infrastructure | Removed entirely |

### What Does Not Change

- React + TypeScript + Fluent UI frontend
- Zustand stores (`editStore`, `libraryStore`, `settingsStore`)
- TypeScript types (`types/index.ts`)
- EditState data model and history system
- WebGL shaders (`adjustments.glsl`, `fragment.glsl`, `vertex.glsl`)
- Film sim look definitions and LUT format
- UI components and layout

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Browser                           │
│                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ React UI │───▶│ WASM Engine  │───▶│  WebGL2    │ │
│  │ (Fluent) │    │ (stages 1-9) │    │ (stages    │ │
│  │          │◀───│              │    │  10-17)    │ │
│  └──────────┘    └──────────────┘    └────────────┘ │
│       │                │                     │       │
│       ▼                ▼                     ▼       │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ IndexedDB│    │  Web Worker  │    │  Canvas    │ │
│  │ (files,  │    │ (off-thread  │    │ (display)  │ │
│  │  thumbs) │    │  decode)     │    │            │ │
│  └──────────┘    └──────────────┘    └────────────┘ │
│       │                                              │
│       ▼                                              │
│  ┌──────────────────────────────────────────────┐   │
│  │ Gemini / OpenAI APIs (direct, user API key)  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 3. Implementation Phases

### Phase 1: WASM Engine + Local File Storage (Core — Must Ship)

This phase replaces the backend with in-browser equivalents. The app becomes fully functional without a server.

### Phase 2: WebGL Preview Pipeline (Performance)

Wire up the existing WebGL shaders for real-time slider response. Phase 1 works without this (WASM re-renders on every change), but Phase 2 makes it fast.

### Phase 3: ML Features via onnxruntime-web (Optional, Deferred)

Port auto-enhance and optimize to browser-side ONNX inference.

---

## 4. Phase 1 — WASM Engine + Local Storage

### 4.1 Create Rust WASM Crate

**New directory: `wasm-engine/`**

```
wasm-engine/
├── Cargo.toml
├── src/
│   └── lib.rs
└── build.sh
```

**`wasm-engine/Cargo.toml`:**

```toml
[package]
name = "webraw-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
rawloader = "0.37"
image = { version = "0.25", default-features = false, features = ["png", "jpeg", "tiff"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console"] }
sha2 = "0.10"
hex = "0.4"
base64 = "0.22"

[profile.release]
opt-level = "s"      # optimize for size
lto = true
```

**WASM compatibility — `rawloader` and `rayon`:**

`rawloader` is pure Rust (no C FFI) and its core API accepts `impl Read` (the backend uses `Cursor::new(&[u8])` — no filesystem access). However, `rawloader 0.37` depends on `rayon` for parallel demosaicing, and `rayon` spawns OS threads which do not exist on `wasm32-unknown-unknown`.

**Validation gate (must pass before proceeding):**

```bash
cd wasm-engine
cargo build --target wasm32-unknown-unknown
```

**If it fails due to rayon**, apply one of these mitigations in order of preference:

1. **Check if rayon is feature-gated** in rawloader. If so, disable the feature in Cargo.toml:
   ```toml
   rawloader = { version = "0.37", default-features = false }
   ```
   Demosaicing runs single-threaded — ~2x slower but functionally identical.

2. **Patch rawloader** via `[patch.crates-io]` to stub out rayon with a single-threaded fallback. The `rayon` crate's `iter::IntoParallelIterator` can be replaced with `iter::IntoIterator` — a one-line change in the demosaic loop.

3. **Use the `rawler` crate** (actively maintained fork of rawloader with updated dependencies). Check `rawler` WASM compatibility first.

4. **Last resort:** Strip RAW decode entirely from Phase 1. Support only raster import (JPG/PNG/TIFF via the `image` crate, which has established WASM support). Add RAW support in Phase 2 once the rayon issue is resolved. This is the fastest path to a deployable SPA but limits functionality.

The `image` crate compiles to WASM without issues.

**`wasm-engine/src/lib.rs` — Exported Functions:**

Port the processing logic from `backend/src/services/render.rs` into these `#[wasm_bindgen]` exports:

```rust
use wasm_bindgen::prelude::*;

/// Decode a RAW or raster file from bytes. Returns JSON metadata:
/// { width, height, exif, hash }
/// Stores the decoded pixel data internally for subsequent operations.
#[wasm_bindgen]
pub fn load_image(file_bytes: &[u8], filename: &str) -> Result<String, JsValue>;

/// Generate a thumbnail JPEG (max_edge px). Returns JPEG bytes.
#[wasm_bindgen]
pub fn generate_thumbnail(file_bytes: &[u8], filename: &str, max_edge: u32) -> Result<Vec<u8>, JsValue>;

/// Render a preview with edit state applied (all stages 1-17).
/// edit_state_json: serialized EditState (without history/snapshots).
/// Returns JSON: { imageBase64, mimeType, width, height, histogram }
#[wasm_bindgen]
pub fn render_preview(
    file_bytes: &[u8],
    filename: &str,
    edit_state_json: &str,
    max_edge: u32,
) -> Result<String, JsValue>;

/// Render a base image (stages 1-9 only, no edits applied).
/// Returns raw RGBA pixel data for WebGL texture upload.
/// Also returns JSON metadata via a separate call.
#[wasm_bindgen]
pub fn render_base(
    file_bytes: &[u8],
    filename: &str,
    max_edge: u32,
) -> Result<Vec<u8>, JsValue>;

/// Export at full quality. Returns encoded image bytes (PNG/JPEG/TIFF).
#[wasm_bindgen]
pub fn render_export(
    file_bytes: &[u8],
    filename: &str,
    edit_state_json: &str,
    format: &str,       // "JPG" | "PNG" | "TIFF"
    quality: u32,       // 1-100 for JPEG
    bit_depth: u32,     // 8 or 16 (for PNG/TIFF)
    color_space: &str,  // "sRGB" | "DisplayP3"
) -> Result<Vec<u8>, JsValue>;

/// Compute histogram from RGBA pixel data. Returns JSON HistogramData.
#[wasm_bindgen]
pub fn compute_histogram(rgba_data: &[u8], width: u32, height: u32) -> String;
```

**Implementation approach:** The render logic in `backend/src/services/render.rs` ports directly. Copy these functions verbatim into the WASM crate:
- `load_raw_image()` — rawloader decode + demosaic + tone mapping
- `apply_edit_state()` — 3-phase edit pipeline (exposure/WB → tone/contrast → HSL/effects)
- `film_sim_presets()` — 6 built-in film looks with color matrices
- Thumbnail resize via `image::imageops::resize`
- Export encoding via `image` crate

**No changes needed** to the rawloader or image crate APIs — they operate on byte slices in memory, same as WASM.

**Critical porting note — copy algorithms exactly:**

These are custom implementations in `render.rs`, not standard library calls. They must be copied verbatim (not reimplemented) for visual parity with the current backend:

1. **Filmic tone mapping** (lines 388-410): Uncharted 2 curve with specific constants `A=0.22, B=0.30, C=0.10, D=0.20, E=0.01, F=0.30, W=2.0`. This is applied during RAW decode to map HDR sensor data to [0,1]. It is NOT standard sRGB gamma.

2. **Tone curve interpolation** (lines 814-836): Hermite spline between control points, pre-computed into a 256-entry LUT. Not linear interpolation.

3. **Film grain hash** (lines 893-910): Deterministic per-pixel noise using multiplicative hash `(x * 2654435761) >> 16`. Same input must produce identical grain pattern.

4. **HSL channel weighting** (lines 625-634): Hue weight uses a linear falloff with 30-degree width per channel, not a Gaussian.

5. **Bilinear demosaic** (lines 326-378): CFA-aware neighbor averaging. R pixel: center + avg 4 ortho neighbors for G + avg 4 diagonal for B. Green pixel: differs by row parity.

6. **sRGB gamma** (lines 413-420): Standard piecewise function with breakpoint at 0.0031308.

**Features defined in EditState but NOT implemented in current backend** (do not implement in WASM — they are no-ops for now):
- `crop`, `transform` — parsed but not applied in render pipeline
- `optics` (distortion, vignette, CA correction) — parsed but not applied
- `sharpening` — parsed but not applied
- `denoise` — only applied via ML buffer ops path (auto-enhance), not standard render
- `grading` (color grading wheels) — parsed but not applied
- `localAdjustments` — mask definitions parsed but never rendered
- `aiLayers` compositing — layer metadata parsed but pixels never composited

These are all specified in `spec.md` for future implementation but have no backend code. The WASM engine should accept these fields in the EditState JSON without error, but skip them during rendering — same as the current backend.

### 4.2 Build Pipeline

**`wasm-engine/build.sh`:**

```bash
#!/bin/bash
set -e
wasm-pack build --target web --out-dir ../frontend/src/wasm-pkg --release
```

This outputs the WASM binary + JS glue code directly into the frontend source tree where Vite can import it.

**Install wasm-pack:** `cargo install wasm-pack` (CI will also need this).

### 4.3 Replace `api/client.ts` with Local Engine

**Replace: `frontend/src/api/client.ts`**

The current file makes HTTP calls to `API_BASE`. Replace every function with a local equivalent.

**Phase 1 approach: call WASM directly on the main thread.** This is simpler and avoids the Worker message-passing complexity. RAW decoding will briefly block the UI (1-2s for a typical file), but the existing `Spinner` overlay in `DevelopView.tsx` already handles this UX. The Web Worker optimization is deferred to Phase 2 (§5) alongside the WebGL split, since both target slider performance.

```typescript
// frontend/src/api/client.ts — rewritten for client-only

import init, {
  load_image,
  generate_thumbnail,
  render_preview,
  render_export,
  compute_histogram,
} from '../wasm-pkg/webraw_wasm';
import { get, set, del, keys } from 'idb-keyval';
import type { EditState, HistogramData } from '../types';

let wasmReady = false;

export async function initEngine(): Promise<void> {
  if (!wasmReady) {
    await init();
    wasmReady = true;
  }
}

// ─── File Storage (IndexedDB) ───

/** Store file bytes in IndexedDB, return a fileId */
export async function uploadFile(file: File): Promise<{
  fileId: string;
  exif: Record<string, unknown> | null;
  sizeBytes: number;
}> {
  await initEngine();
  const bytes = new Uint8Array(await file.arrayBuffer());
  const fileId = crypto.randomUUID();

  // Get metadata from WASM
  const metaJson = load_image(bytes, file.name);
  const meta = JSON.parse(metaJson);

  // Store raw bytes in IndexedDB
  await set(`file:${fileId}`, bytes);
  await set(`meta:${fileId}`, {
    filename: file.name,
    size: file.size,
    hash: meta.hash,
    exif: meta.exif,
  });

  return {
    fileId,
    exif: meta.exif ?? null,
    sizeBytes: file.size,
  };
}

export async function deleteFile(fileId: string): Promise<void> {
  await del(`file:${fileId}`);
  await del(`meta:${fileId}`);
  await del(`thumb:${fileId}`);
}

// ─── Thumbnail ───

export async function generateThumbnailLocal(
  fileId: string,
  maxEdge = 300
): Promise<Blob> {
  await initEngine();
  const bytes = await get(`file:${fileId}`) as Uint8Array;
  if (!bytes) throw new Error(`File ${fileId} not found in storage`);

  const meta = await get(`meta:${fileId}`) as { filename: string };
  const jpegBytes = generate_thumbnail(bytes, meta.filename, maxEdge);
  const blob = new Blob([jpegBytes], { type: 'image/jpeg' });

  // Cache the thumbnail
  await set(`thumb:${fileId}`, jpegBytes);
  return blob;
}

// ─── Preview ───

export interface PreviewResult {
  imageBase64: string;
  mimeType: string;
  bitDepth: number;
  colorSpace: string;
  width: number;
  height: number;
  histogram?: HistogramData;
}

export async function renderPreviewLocal(
  fileId: string,
  editState: EditState,
  options: { maxEdge?: number } = {}
): Promise<PreviewResult> {
  await initEngine();
  const bytes = await get(`file:${fileId}`) as Uint8Array;
  if (!bytes) throw new Error(`File ${fileId} not found in storage`);

  const meta = await get(`meta:${fileId}`) as { filename: string };
  const editJson = JSON.stringify({
    schemaVersion: editState.schemaVersion,
    assetId: editState.assetId,
    global: editState.global,
    localAdjustments: editState.localAdjustments,
    aiLayers: editState.aiLayers,
    filmSim: editState.filmSim,
  });

  const resultJson = render_preview(
    bytes,
    meta.filename,
    editJson,
    options.maxEdge ?? 2560
  );
  return JSON.parse(resultJson);
}

// ─── Export ───

export async function renderExportLocal(
  fileId: string,
  editState: EditState,
  format: 'JPG' | 'PNG' | 'TIFF',
  options: { bitDepth?: number; quality?: number; colorSpace?: string } = {}
): Promise<Blob> {
  await initEngine();
  const bytes = await get(`file:${fileId}`) as Uint8Array;
  if (!bytes) throw new Error(`File ${fileId} not found in storage`);

  const meta = await get(`meta:${fileId}`) as { filename: string };
  const editJson = JSON.stringify({
    schemaVersion: editState.schemaVersion,
    assetId: editState.assetId,
    global: editState.global,
    localAdjustments: editState.localAdjustments,
    aiLayers: editState.aiLayers,
    filmSim: editState.filmSim,
  });

  const exportBytes = render_export(
    bytes,
    meta.filename,
    editJson,
    format,
    options.quality ?? 95,
    options.bitDepth ?? 8,
    options.colorSpace ?? 'sRGB',
  );

  const mimeMap = { JPG: 'image/jpeg', PNG: 'image/png', TIFF: 'image/tiff' };
  return new Blob([exportBytes], { type: mimeMap[format] });
}

// ─── AI Editing (direct browser→API) ───

export async function createAiEdit(
  imageBase64: string,
  prompt: string,
  mode: string,
  apiKey: string,
  options?: {
    negativePrompt?: string;
    provider?: 'gemini' | 'openai' | 'google-imagen';
  }
): Promise<{ imageData: Uint8Array; mimeType: string; model: string }> {
  const provider = options?.provider ?? 'gemini';

  const modePrompts: Record<string, string> = {
    edit: `Edit this image: ${prompt}`,
    remove: `Remove the following from this image: ${prompt}`,
    replace_bg: `Replace the background of this image with: ${prompt}`,
    relight: `Relight this image: ${prompt}`,
    expand: `Expand this image outward: ${prompt}`,
  };
  const systemPrompt = modePrompts[mode] ?? prompt;

  if (provider === 'gemini') {
    return callGemini(imageBase64, systemPrompt, apiKey);
  } else if (provider === 'openai') {
    return callOpenAI(imageBase64, systemPrompt, apiKey);
  } else if (provider === 'google-imagen') {
    return callImagen(imageBase64, systemPrompt, apiKey);
  }
  throw new Error(`Unsupported provider: ${provider}`);
}

async function callGemini(
  imageBase64: string,
  prompt: string,
  apiKey: string
): Promise<{ imageData: Uint8Array; mimeType: string; model: string }> {
  // Match backend default (config.rs GEMINI_MODEL). User can override in settings.
  const model = 'gemini-2.5-pro';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-goog-api-key': apiKey,
    },
    body: JSON.stringify({
      contents: [{
        parts: [
          { text: prompt },
          { inlineData: { mimeType: 'image/jpeg', data: imageBase64 } },
        ],
      }],
      generationConfig: {
        responseModalities: ['IMAGE', 'TEXT'],
        responseMimeType: 'image/png',
      },
    }),
  });

  if (!response.ok) {
    const status = response.status;
    if (status === 429) throw new Error('AI_QUOTA_EXCEEDED');
    if (status === 401 || status === 403) throw new Error('AI_INVALID_KEY');
    throw new Error(`Gemini API error: ${status}`);
  }

  const data = await response.json();
  const part = data.candidates?.[0]?.content?.parts?.find(
    (p: any) => p.inlineData
  );
  if (!part) throw new Error('No image in Gemini response');

  const binary = atob(part.inlineData.data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  return { imageData: bytes, mimeType: part.inlineData.mimeType, model };
}

async function callOpenAI(
  imageBase64: string,
  prompt: string,
  apiKey: string
): Promise<{ imageData: Uint8Array; mimeType: string; model: string }> {
  const model = 'gpt-image-1';
  const binary = atob(imageBase64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const imageBlob = new Blob([bytes], { type: 'image/png' });

  const form = new FormData();
  form.append('image', imageBlob, 'image.png');
  form.append('prompt', prompt);
  form.append('model', model);

  const response = await fetch('https://api.openai.com/v1/images/edits', {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}` },
    body: form,
  });

  if (!response.ok) {
    const status = response.status;
    if (status === 429) throw new Error('AI_QUOTA_EXCEEDED');
    if (status === 401 || status === 403) throw new Error('AI_INVALID_KEY');
    throw new Error(`OpenAI API error: ${status}`);
  }

  const data = await response.json();
  const b64 = data.data?.[0]?.b64_json;
  if (!b64) throw new Error('No image in OpenAI response');

  const resultBin = atob(b64);
  const resultBytes = new Uint8Array(resultBin.length);
  for (let i = 0; i < resultBin.length; i++) resultBytes[i] = resultBin.charCodeAt(i);

  return { imageData: resultBytes, mimeType: 'image/png', model };
}

async function callImagen(
  imageBase64: string,
  prompt: string,
  apiKey: string
): Promise<{ imageData: Uint8Array; mimeType: string; model: string }> {
  const model = 'imagen-3.0-edit-001';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:predict`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-goog-api-key': apiKey,
    },
    body: JSON.stringify({
      instances: [{
        prompt,
        image: { bytesBase64Encoded: imageBase64 },
      }],
      parameters: { sampleCount: 1 },
    }),
  });

  if (!response.ok) {
    const status = response.status;
    if (status === 429) throw new Error('AI_QUOTA_EXCEEDED');
    if (status === 401 || status === 403) throw new Error('AI_INVALID_KEY');
    throw new Error(`Imagen API error: ${status}`);
  }

  const data = await response.json();
  const b64 = data.predictions?.[0]?.bytesBase64Encoded;
  if (!b64) throw new Error('No image in Imagen response');

  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  return { imageData: bytes, mimeType: 'image/png', model };
}

// ─── Removed ───
// healthCheck — no backend
// pollJob / getJob — no async jobs, everything is synchronous in WASM
// baseRenderStream — no SSE, WASM renders directly
// listEnhanceModels / runAutoEnhance — deferred to Phase 3
// listOptimizeModels / runOptimize / computeOptimizeMasks — deferred to Phase 3
```

### 4.4 Update `usePreviewRenderer` Hook

**Modify: `frontend/src/hooks/usePreviewRenderer.ts`**

Replace the `renderPreview` API call with the local WASM call:

```typescript
// Change the import:
// Before: import { renderPreview, type PreviewResult } from '../api/client';
// After:
import { renderPreviewLocal, type PreviewResult } from '../api/client';

// In fetchPreview(), replace:
//   const result = await renderPreview(fileId, apiEditState as EditState, { ... });
// With:
//   const result = await renderPreviewLocal(fileId, apiEditState as EditState, {
//     maxEdge: 2560,
//   });
```

The rest of the hook (debouncing, generation counter, blob URL management) stays the same. Remove the `AbortSignal` plumbing — WASM calls are synchronous and cannot be aborted mid-execution. The generation counter already handles stale responses correctly.

**Note:** In Phase 1, WASM calls run on the main thread and will block the UI briefly (1-2s for a typical RAW file). The existing `<Spinner label="Rendering..." />` overlay in `DevelopView.tsx` (lines 77-89) provides UX feedback. Phase 2 moves WASM to a Web Worker for non-blocking rendering — see §5.6.

### 4.6 Update `AppShell.tsx` — Import Flow

**Modify: `frontend/src/components/common/AppShell.tsx`**

The import flow changes from "upload to server → get fileId" to "read file → store in IndexedDB → get fileId":

```typescript
// Before:
import { uploadFile, generateThumbnail } from '../../api/client';

// After:
import { uploadFile, generateThumbnailLocal } from '../../api/client';
```

The `uploadFile` function signature stays the same (takes a `File`, returns `{ fileId, exif, sizeBytes }`). The implementation changes internally (writes to IndexedDB instead of POSTing to server).

The `backgroundThumbnail` helper changes `generateThumbnail` → `generateThumbnailLocal`.

### 4.7 Remove Backend Health Check

**Modify: `frontend/src/App.tsx`**

Delete:
- `import { healthCheck } from './api/client'` (line 5)
- `import { useSettingsStore } from './stores/settingsStore'` (line 6)
- `const setBackendHealthy = useSettingsStore(...)` (line 20)
- The entire `useEffect` block (lines 22-34) that runs `checkHealth` on a 15-second interval

The resulting `App` component is just the FluentProvider + QueryClientProvider + AppShell.

**Modify: `frontend/src/components/common/AppShell.tsx`**

Delete:
- `const backendHealthy = useSettingsStore(s => s.backendHealthy)` (line 46)
- The "Backend service unavailable" `MessageBar` block (lines 184-190)
- The `useSettingsStore` import (line 18) if no other settings are used in this file

### 4.8 Update Export Dialog

**Modify: `frontend/src/components/export/ExportDialog.tsx`**

**Change imports (line 21):**
```typescript
// Before:
import { renderExport, pollJob, getFileUrl } from '../../api/client';

// After:
import { renderExportLocal } from '../../api/client';
```

**Replace `handleExport` (lines 41-75):**

```typescript
const handleExport = useCallback(async () => {
  if (!editState || !activeAsset?.fileId) return;

  setExporting(true);
  setError(null);

  try {
    const blob = await renderExportLocal(
      activeAsset.fileId,
      editState,
      format,
      { bitDepth, quality, colorSpace },
    );

    // Trigger browser download
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const baseName = activeAsset.filename.replace(/\.[^.]+$/, '');
    a.download = `${baseName}.${format.toLowerCase() === 'jpg' ? 'jpg' : format.toLowerCase()}`;
    a.click();
    URL.revokeObjectURL(url);

    setDownloadUrl('done'); // signal success to UI
  } catch (err: any) {
    setError(err.message ?? 'Export failed');
  } finally {
    setExporting(false);
  }
}, [editState, activeAsset, format, quality, bitDepth, colorSpace]);
```

**Key changes:**
- No `pollJob` — export is synchronous (WASM renders and encodes in one call).
- No `getFileUrl` — download is triggered directly from a Blob.
- `progress` state is no longer incremental (no job polling). While `exporting` is true, show the existing `<ProgressBar>` in indeterminate mode (remove `value={progress}`, use `value={undefined}`).
- The `downloadUrl` state can be simplified to a boolean success flag, or keep the existing success `MessageBar` and set `downloadUrl` to any truthy string.
- `bitDepth` and `colorSpace` are passed through to the WASM engine — the UI controls for these (lines 111-128) remain unchanged.

### 4.9 Update AI Edit Flow

**Modify: `frontend/src/components/ai/AiEditPanel.tsx`**

The AI flow changes from "send to backend proxy → poll job" to "render preview via WASM → call API directly → store result in IndexedDB."

**Change imports (line 16):**
```typescript
// Before:
import { createAiEdit, pollJob, getFileUrl } from '../../api/client';

// After:
import { renderPreviewLocal, createAiEdit } from '../../api/client';
import { set } from 'idb-keyval';
```

**Replace `handleSubmit` (lines 43-93):**

```typescript
const handleSubmit = useCallback(async () => {
  if (!editState || !activeAsset?.fileId || !activeApiKey || !prompt.trim()) return;

  setIsSubmitting(true);
  setError(null);

  try {
    // Step 1: Render current edit state to a preview image via WASM
    const preview = await renderPreviewLocal(activeAsset.fileId, editState, { maxEdge: 2048 });

    // Step 2: Call Gemini/OpenAI directly from browser (no backend proxy)
    const aiResult = await createAiEdit(
      preview.imageBase64,
      prompt,
      mode,
      activeApiKey,
      { provider }
    );

    // Step 3: Store the AI result image in IndexedDB
    const resultFileId = crypto.randomUUID();
    await set(`file:${resultFileId}`, aiResult.imageData);
    await set(`meta:${resultFileId}`, {
      filename: `ai-${resultFileId}.png`,
      size: aiResult.imageData.length,
    });

    // Step 4: Add as AI layer (same structure as before)
    const layer: AiLayer = {
      id: crypto.randomUUID(),
      assetId: resultFileId,
      opacity: 1.0,
      blendMode: 'NORMAL',
      meta: {
        provider,
        model: aiResult.model,
        prompt,
        createdAt: new Date().toISOString(),
      },
      enabled: true,
    };
    addAiLayer(layer);
    pushHistory(`AI: ${mode}`);
    setPrompt('');
  } catch (err: any) {
    setError(err.message ?? 'AI edit failed');
  } finally {
    setIsSubmitting(false);
  }
}, [editState, activeAsset, activeApiKey, prompt, mode, provider, addAiLayer, pushHistory]);
```

**Key changes:**
- No `pollJob` — no async jobs. The WASM render + API call are sequential awaits.
- `progress` state is no longer needed (remove `const [progress, setProgress] = useState(0)` on line 38).
- The `createAiEdit` signature is different: takes `imageBase64` (rendered preview) instead of `fileId + editState`.
- Result image bytes are stored directly in IndexedDB, not on a backend cache.

**Note on CORS:** Gemini API and OpenAI API both support browser-origin requests with API key auth. No CORS issues for direct calls.

### 4.10 Deferred Features

These features require ONNX Runtime and are removed from the UI in Phase 1.

**Modify: `frontend/src/components/develop/DevelopView.tsx`**

Remove the `AutoEnhancePanel` import (line 14) and its render call (line 102):

```typescript
// Line 14 — delete this import:
// import { AutoEnhancePanel } from './panels/AutoEnhancePanel';

// Line 102 — delete this render:
// <AutoEnhancePanel />
```

Also remove the unused `backendHealthy` read (line 24):
```typescript
// Line 24 — delete:
// const backendHealthy = useSettingsStore(s => s.backendHealthy);
```
And the `useSettingsStore` import (line 4) if no other settings are used in this file.

**Enhanced denoise:** The `enhanced?: boolean` field on `Denoise` in `types/index.ts` can remain in the type definition. The WASM engine should ignore it (treat all denoise as standard bilateral). The `DetailPanel` UI toggle for enhanced denoise should be hidden or disabled with a tooltip: "Requires server backend."

### 4.11 Update `settingsStore.ts`

**Modify: `frontend/src/stores/settingsStore.ts`**

Remove these from the `SettingsStore` interface and implementation:
- `backendHealthy: boolean` (line 12)
- `setBackendHealthy: (healthy: boolean) => void` (line 21)
- `backendHealthy: false` initial value (line 34)
- `setBackendHealthy` setter (line 43)

The `colorSpace` field (line 10) stays — it's used by `ExportDialog.tsx` and will be passed to the WASM export function.

### 4.12 Library Persistence Across Refresh

**Current problem:** `libraryStore` is Zustand in-memory. File bytes now persist in IndexedDB, but the asset list (filenames, ratings, flags, thumbnails) is lost on refresh.

**Fix:** Add `zustand/middleware` `persist` to `libraryStore`, backed by `localStorage` (small metadata only — no binary data).

**Modify: `frontend/src/stores/libraryStore.ts`**

```typescript
import { persist } from 'zustand/middleware';

export const useLibraryStore = create<LibraryStore>()(
  persist(
    (set, get) => ({
      // ... existing implementation unchanged
    }),
    {
      name: 'webraw-library',
      partialize: (state) => ({
        assets: state.assets.map(a => ({
          ...a,
          // Exclude blob URLs — they're invalid after refresh.
          // Thumbnails will be re-generated from IndexedDB on next load.
          thumbnailUrl: undefined,
        })),
        activeAssetId: state.activeAssetId,
      }),
    },
  ),
);
```

On app startup, if persisted assets exist but `thumbnailUrl` is undefined, re-generate thumbnails from IndexedDB in the background (same `backgroundThumbnail` pattern as import).

**Why this matters:** Without this, a user imports 20 RAW files, edits some, refreshes the browser, and their library is gone — even though all files are still in IndexedDB. This would be a confusing experience.

### 4.13 Remove Job Polling Infrastructure

The following are no longer needed and should be removed from `types/index.ts`:
- `JobStatus`, `JobKind`, `JobInfo` types — can be removed if no UI references remain
- `ProblemDetail` — can be simplified to a standard Error

From `api/client.ts`:
- `getJob()`, `pollJob()`, `healthCheck()`, `baseRenderStream()` — all removed
- `getSessionToken()`, `apiFetch()`, `apiJson()` — all removed (no server to talk to)

---

## 5. Phase 2 — WebGL Preview Pipeline

### 5.1 Why

Phase 1 works but is slow for slider interaction: every slider change re-runs the full WASM pipeline (stages 1-17), which takes 1-2 seconds for a typical RAW file. The spec targets < 150ms slider response.

Phase 2 fixes this by splitting the pipeline:
- **WASM** renders stages 1-9 (base render) once when the image is loaded or base params change
- **WebGL** applies stages 10-17 (adjustments, film sim, effects) in real-time on the GPU

The WebGL shaders for this already exist in `frontend/src/shaders/` but are not wired up. `ImageCanvas.tsx` currently uses Canvas2D.

### 5.2 Identify Base vs. Adjustment Parameters

Parameters that require a WASM re-render (stages 1-9 — base render):
- `temperature`, `tint` (white balance)
- `denoise.luma`, `denoise.chroma` (pre-color denoise)
- `optics.*` (lens correction)

Parameters handled by WebGL shaders (stages 10-17 — real-time):
- `exposure`, `contrast`
- `highlights`, `shadows`, `whites`, `blacks`
- `vibrance`, `saturation`
- `texture`, `clarity`, `dehaze`
- `toneCurve`
- `filmSim` (3D LUT texture + strength)
- `effects.grainAmount`, `effects.vignetteAmount`
- `hsl`, `grading`
- `sharpening`

### 5.3 Rewrite `ImageCanvas.tsx` to Use WebGL2

**Modify: `frontend/src/components/develop/ImageCanvas.tsx`**

Replace Canvas2D drawing with WebGL2 rendering:

1. Get `webgl2` context instead of `2d`
2. Compile and link the existing vertex + fragment shaders
3. Upload the base render (from WASM) as a `TEXTURE_2D`
4. Set uniform values from `EditState.global.*`
5. Draw a full-screen quad
6. On slider change: update uniforms → redraw (instant, no WASM call)
7. On base param change (WB, denoise, optics): re-run WASM → re-upload texture → redraw

The shaders already handle: exposure, contrast, highlights/shadows/whites/blacks, vibrance, saturation, dehaze, film sim LUT, grain, vignette. They match the pipeline stages 10-17.

### 5.4 Update `usePreviewRenderer` Hook

Split the hook into two concerns:

```typescript
// 1. Base render: triggered when fileId or base params change
//    Calls WASM render_base() → returns RGBA pixel data
//    Uploads to WebGL texture

// 2. Adjustment render: triggered when any non-base param changes
//    Updates WebGL uniforms → redraws (sub-frame, no WASM call)
```

The debounce logic (currently 150ms) can be tightened for WebGL-only changes since they're effectively free.

### 5.5 Web Worker for WASM Rendering

Phase 1 calls WASM on the main thread (blocking UI for 1-2s during RAW decode). Phase 2 moves WASM to a Web Worker so the UI stays responsive during base renders.

**New file: `frontend/src/workers/render-worker.ts`**

```typescript
import init, {
  render_preview,
  generate_thumbnail,
  render_export,
  render_base,
  load_image,
} from '../wasm-pkg/webraw_wasm';

let ready = false;

async function ensureInit() {
  if (!ready) {
    await init();
    ready = true;
  }
}

self.onmessage = async (e: MessageEvent) => {
  await ensureInit();
  const { id, type, payload } = e.data;

  try {
    let result: unknown;
    switch (type) {
      case 'load':
        result = load_image(payload.bytes, payload.filename);
        break;
      case 'thumbnail':
        result = generate_thumbnail(payload.bytes, payload.filename, payload.maxEdge);
        break;
      case 'preview':
        result = render_preview(payload.bytes, payload.filename, payload.editJson, payload.maxEdge);
        break;
      case 'base':
        result = render_base(payload.bytes, payload.filename, payload.maxEdge);
        break;
      case 'export':
        result = render_export(
          payload.bytes, payload.filename, payload.editJson,
          payload.format, payload.quality, payload.bitDepth, payload.colorSpace,
        );
        break;
    }
    self.postMessage({ id, result });
  } catch (err) {
    self.postMessage({ id, error: String(err) });
  }
};
```

**Integration with `api/client.ts`:** Replace the direct WASM function calls with a worker message wrapper:

```typescript
const worker = new Worker(
  new URL('../workers/render-worker.ts', import.meta.url),
  { type: 'module' }
);

let nextId = 0;
const pending = new Map<number, { resolve: Function; reject: Function }>();

worker.onmessage = (e: MessageEvent) => {
  const { id, result, error } = e.data;
  const p = pending.get(id);
  if (!p) return;
  pending.delete(id);
  error ? p.reject(new Error(error)) : p.resolve(result);
};

function callWorker(type: string, payload: Record<string, unknown>): Promise<unknown> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, payload });
  });
}
```

Each `api/client.ts` function then calls `callWorker(type, payload)` instead of the WASM function directly. Use `Transferable` for large `Uint8Array` buffers to avoid copying.

**Vite config for worker:** Vite supports `new Worker(new URL(...), { type: 'module' })` natively. No plugin needed.

### 5.6 Film Sim LUT as 3D Texture

The shader already has `uniform sampler3D u_filmSimLUT`. When a film sim is selected:

1. Load the `.cube` LUT file (33x33x33 = ~108KB per look)
2. Parse into a Float32Array
3. Upload as a `TEXTURE_3D` with `gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB32F, 33, 33, 33, 0, gl.RGB, gl.FLOAT, lutData)`
4. Set `u_filmSimEnabled = true` and `u_filmSimStrength`

LUT files ship as static assets in `frontend/public/luts/`.

---

## 6. Vite Config + GitHub Pages

### 6.1 Update Vite Config

**Modify: `frontend/vite.config.ts`**

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
  base: '/webraw/',
  plugins: [react(), wasm()],
  build: {
    target: 'esnext',  // needed for top-level await (WASM init)
  },
  worker: {
    format: 'es',
  },
});
```

**Add dev dependency:** `npm install -D vite-plugin-wasm`

### 6.2 GitHub Actions Workflow

**New file: `.github/workflows/deploy.yml`**

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [master]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown

      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

      - name: Build WASM engine
        working-directory: wasm-engine
        run: wasm-pack build --target web --out-dir ../frontend/src/wasm-pkg --release

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - run: npm ci
        working-directory: frontend

      - run: npm run build
        working-directory: frontend

      - run: cp frontend/dist/index.html frontend/dist/404.html

      - uses: actions/upload-pages-artifact@v3
        with:
          path: frontend/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

### 6.3 Repository Settings

- Settings → Pages → Source: **GitHub Actions**
- No secrets needed (no backend URL, no API keys in build)

---

## 7. Files Changed / Created / Deleted

```
Phase 1:
  New:   wasm-engine/Cargo.toml
  New:   wasm-engine/src/lib.rs
  New:   wasm-engine/build.sh
  New:   .github/workflows/deploy.yml

  Mod:   frontend/src/api/client.ts              (rewrite: WASM + IndexedDB instead of HTTP)
  Mod:   frontend/src/hooks/usePreviewRenderer.ts (use renderPreviewLocal, remove AbortSignal)
  Mod:   frontend/src/components/common/AppShell.tsx (remove backend health banner, use generateThumbnailLocal)
  Mod:   frontend/src/components/export/ExportDialog.tsx (Blob download, keep bitDepth/colorSpace)
  Mod:   frontend/src/components/ai/AiEditPanel.tsx (direct API calls, idb-keyval storage)
  Mod:   frontend/src/components/develop/DevelopView.tsx (remove AutoEnhancePanel, remove backendHealthy)
  Mod:   frontend/src/stores/settingsStore.ts     (remove backendHealthy + setBackendHealthy)
  Mod:   frontend/src/stores/libraryStore.ts      (add zustand persist for asset list)
  Mod:   frontend/src/App.tsx                     (remove health check useEffect)
  Mod:   frontend/vite.config.ts                  (add base path, WASM plugin)
  Mod:   frontend/package.json                    (add vite-plugin-wasm)

  Del:   backend/                                 (entire directory — after WASM engine is verified)
  Del:   docker-compose.yml                       (no longer needed)

Phase 2:
  New:   frontend/src/workers/render-worker.ts    (Web Worker for non-blocking WASM)
  Mod:   frontend/src/components/develop/ImageCanvas.tsx (Canvas2D → WebGL2)
  Mod:   frontend/src/api/client.ts               (route WASM calls through Worker)
  Mod:   frontend/src/hooks/usePreviewRenderer.ts (split base vs. adjustment renders)
```

---

## 8. Build + Dev Workflow

### Local Development

```bash
# Terminal 1: Build WASM (watch mode not available; rebuild manually after Rust changes)
cd wasm-engine && wasm-pack build --target web --out-dir ../frontend/src/wasm-pkg --dev

# Terminal 2: Run Vite dev server
cd frontend && npm run dev
```

### Prerequisites

```bash
# One-time setup
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cd frontend && npm install
```

---

## 9. Performance Considerations

- **RAW file size in memory:** A 50MB RAW file loaded into WASM uses ~50MB of WASM linear memory. Decoded to RGBA at full resolution (e.g., 6000x4000 = 96MB). Preview resolution (2560px long edge) is ~25MB. Total per-image working set: ~175MB. Modern browsers handle this fine.

- **WASM memory limit:** Default is 256MB, growable to 4GB. Set initial memory to 512MB in wasm-pack config if needed.

- **IndexedDB storage:** RAW files stored as `Uint8Array`. Browser storage quota is typically 50-80% of available disk space. Monitor with `navigator.storage.estimate()`.

- **Web Worker overhead:** Message passing copies data between threads. For large buffers, use `Transferable` objects (`postMessage(data, [data.buffer])`) to avoid copying.

- **First load:** WASM binary size is ~2-4MB (with `opt-level = "s"` and LTO). Cache via service worker or rely on browser HTTP cache. Consider lazy-loading WASM only when first image is imported.

---

## 10. Testing Checklist

**Gate 0 — WASM compilation (do this first, before any other work):**
- [ ] `cargo build --target wasm32-unknown-unknown` succeeds in `wasm-engine/`
- [ ] If rawloader/rayon fails, apply mitigation from §4.1 and re-test

**Phase 1 — Functional:**
- [ ] `wasm-pack build` succeeds for `wasm-engine/`
- [ ] `wasm-pack test --headless --chrome` passes (add Rust tests for RAW decode)
- [ ] Frontend `npm run build` succeeds with WASM pkg in source tree
- [ ] Import a RAW file (.DNG, .CR2, .NEF) → metadata + thumbnail generated in-browser
- [ ] Import a raster file (.JPG, .PNG) → works without rawloader path
- [ ] Adjust sliders → preview updates (full WASM re-render, 1-2s with spinner)
- [ ] Film sim selection → look applied correctly
- [ ] Export to JPEG → browser download triggers, file is valid
- [ ] Export to 16-bit TIFF → browser download triggers, file has correct bit depth
- [ ] Export color space setting (sRGB / Display P3) is respected
- [ ] AI edit with Gemini API key → direct API call succeeds, result composited as AI layer
- [ ] AI edit with OpenAI API key → direct API call succeeds
- [ ] Multiple images imported → IndexedDB stores all, thumbnails display in library
- [ ] Refresh browser → asset list persists (library metadata), thumbnails re-generate
- [ ] Refresh browser → imported file bytes persist in IndexedDB
- [ ] AutoEnhancePanel is not visible in DevelopView
- [ ] No console errors about missing backend / failed health checks
- [ ] No references to `backendHealthy` remain in rendered UI
- [ ] Site loads at `https://mazharm.github.io/webraw/`
- [ ] SPA routing works (404.html fallback)
- [ ] WASM loads and initializes on first use

**Phase 2 — Performance:**
- [ ] Web Worker renders WASM off-thread (UI does not freeze during RAW decode)
- [ ] WebGL slider adjustments respond in < 150ms
- [ ] Base param changes (WB, denoise, optics) trigger WASM re-render; other params only update WebGL uniforms

---

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `rawloader` fails to compile to WASM due to `rayon` threading dependency | Medium — rayon uses OS threads | **First task in implementation.** Run `cargo build --target wasm32-unknown-unknown` in `wasm-engine/`. If rayon blocks it: (1) disable default features on rawloader, (2) patch rayon out via `[patch.crates-io]`, (3) try the `rawler` crate, or (4) ship Phase 1 with raster-only support. See §4.1 for full mitigation ladder. |
| WASM binary too large (>10MB) | Medium | Enable LTO + `opt-level = "s"`. Strip debug info. Use `wasm-opt` post-processing. Lazy-load WASM on first import. |
| IndexedDB quota exceeded | Low for typical use | Monitor via `navigator.storage.estimate()`. Show warning at 80%. Implement LRU eviction for thumbnails/previews. Keep originals until user explicitly clears. |
| Gemini API CORS blocks browser requests | Low — API supports browser calls | Tested: Gemini API allows browser-origin requests with API key auth. If blocked for specific models, fall back to a minimal Cloudflare Worker proxy (simple, no state). |
| Performance regression vs. native Rust | Expected — WASM is ~1.5-2x slower | Acceptable for preview (1-3s vs 0.5-1.5s). Phase 2 WebGL pipeline eliminates this for slider interaction. Export can show progress bar. |
