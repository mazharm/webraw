# Web RAW Photo Editor (Fluent + Lightroom-like) with Film Sims + Nano Banana Pro AI
**File:** `spec.md`  
**Backend:** Rust  
**Frontend:** React (TypeScript) + Fluent UI React v9 (Fluent 2)  
**AI:** Nano Banana Pro via Gemini API image generation/editing (user-supplied key)

---

## 1. Summary

Build a cross-browser, React-based, Lightroom-like RAW photo editor using Fluent Design. The system supports:

1) **Standard RAW processing** (demosaic, denoise, lens corrections, tone/color pipeline, local adjustments).  
2) **Film-style looks** inspired by Fujifilm “film simulations”, implemented via calibrated, non-proprietary look stacks (LUT + tone curve + matrix + grain).  
3) **AI photo editing** using **Nano Banana Pro** (Gemini image editing). The user provides their own API key to enable AI features.

Deliverables:
- Production-grade web app (React/TS) with Fluent UI and Lightroom-like UX.
- Rust backend services for RAW decode/render, exports, and AI proxy.
- A defined edit format (`EditState JSON`) for non-destructive editing and portability.

---

## 2. Goals / Non-goals

### 2.1 Goals
- Lightroom-like workflow: **Library → Develop → Export**, with non-destructive edits, history, snapshots, and presets.
- High-quality RAW rendering with consistent color and predictable results.
- Responsive UI: smooth thumbnail grid, fast slider feedback (progressive refinement acceptable).
- Fluent 2 design language via Fluent UI React v9.
- AI edits integrated non-destructively as “AI Layers” with audit trail and re-run support.

### 2.2 Non-goals (v1)
- Full DAM parity (face recognition, advanced search, cloud sync).
- Printing pipeline / printer ICC soft proofing.
- Full offline PWA guarantees.
- Direct RAW-domain generative edits (AI operates on rendered raster layers in v1).

---

## 3. Core User Stories

1) Import a folder of RAW images → auto-generate thumbnails → cull (flags/ratings) → open one in Develop → edit.  
2) Apply a film look preset (e.g., “Chrome-like”) and tweak strength/grain.  
3) Run an AI edit: “remove the person in background” → apply output as an AI Layer with mask/opacity controls.  
4) Export a single image or batch export selected images with consistent output settings.  
5) Save a project file and re-open later with edits intact, without altering original RAWs.

---

## 4. Functional Requirements

### 4.1 Supported file types
**Import**
- RAW (initial target list): `.DNG`, `.CR2/.CR3`, `.NEF`, `.ARW`, `.RAF` (coverage depends on LibRaw and camera support).
- Raster: `.JPG/.JPEG`, `.PNG`, `.TIFF`.

**Export**
- `.JPG` (8-bit), `.PNG` (8/16-bit), `.TIFF` (8/16-bit).

### 4.2 Non-destructive editing model
- Original files never modified.
- All edits are stored as `EditState` parameters + sidecar asset references.
- Maintain:
  - History (branchable)
  - Snapshots (named)
  - Presets (reusable subsets)
  - Per-image “Versions” (optional: named variants)

### 4.3 Develop tools (v1)
**Global adjustments**
- Exposure, contrast
- Highlights, shadows, whites, blacks
- Temperature, tint
- Vibrance, saturation
- Texture: high-frequency local contrast via unsharp mask (radius 2–4px equivalent at full res)
- Clarity: mid-frequency local contrast via unsharp mask (radius 20–40px equivalent at full res)
- Dehaze: simplified dark-channel-prior estimation; invert haze map and blend per `dehaze` strength
- Tone curve: parametric + point curve
- HSL (hue/sat/lum by color range)
- Color grading (shadows/mids/highlights wheels)
- Sharpening (amount/radius/detail/masking)
- Denoise (luma/chroma) + optional “enhanced” mode:
  - **Standard:** spatial bilateral filter applied in real-time on frontend WebGL
  - **Enhanced:** async backend job using non-local-means (NLM) denoising; result returned via job polling and cached
- Lens corrections (distortion/vignette/CA) auto + manual
- Crop/rotate/straighten
- Transform (perspective): vertical/horizontal/rotate/aspect

**Local adjustments (minimal v1)**
- Brush mask (add/subtract), feather/flow
- Linear gradient
- Radial gradient
- Spot heal/clone (basic, small radius)

### 4.4 Library tools (v1)
- Import: drag/drop, file picker, folder import (where supported)
- Grid view with virtualized thumbnails
- Flag (pick/reject), rating (1–5), color label (optional)
- Basic metadata display (EXIF subset)
- Filter by rating/flag

---

## 5. Film Sims (Fujifilm-style looks)

### 5.1 UX
- A “Film Sims” panel in Develop.
- Looks listed as:
  - “Chrome-like”
  - “Velvia-like”
  - “Provia-like”
  - “Eterna-like”
  - “Astia-like”
  - “Acros-like (B&W)”
- Controls:
  - Strength (0–200%)
  - Grain (amount/size)
  - B&W filter (R/Y/G) for monochrome looks
  - Optional: highlight rolloff, color chrome (if implemented)

### 5.2 Implementation constraints
- Do **not** ship proprietary Fuji assets or LUTs.
- Implement each look as a deterministic stack:
  - Camera profile/correction (matrix)
  - Tone curve
  - 3D LUT
  - Grain/noise overlay
  - Optional split-toning/grading tweaks

#### 5.2.1 LUT specification
- **Size:** 33×33×33 3D cube LUTs (`.cube` format).
- **Interpolation:** Trilinear interpolation.
- **Color space:** LUTs are authored and applied in the working color space (linear Rec.2020). Both LUT input and output are in this space.
- **Color matrix:** The 3×3 color matrix from the look manifest is applied **before** the LUT.

#### 5.2.2 Strength blending algorithm
- Strength is applied as a linear interpolation in the working color space between the unmodified input and the full look output:
  ```
  result = lerp(input, lookOutput, strength)
  ```
- `strength = 0.0`: identity (no look applied).
- `strength = 1.0`: full look.
- `strength > 1.0`: linear extrapolation beyond the look output (values are clamped to [0, ∞) before output transform).
- This interpolation applies to the combined LUT + tone curve + matrix output, not to each component individually.

### 5.3 Look Definition Format (v1)
- Looks are defined in a JSON manifest (`.json`) bundled with a standard `.cube` LUT:
```json
{
  "id": "chrome_like",
  "name": "Chrome Like",
  "baseCurve": "contrast_medium",
  "lutFile": "chrome_v1.cube",
  "matrix": [1.1, -0.05, -0.05, -0.02, 1.05, -0.03, 0.0, -0.02, 1.02],
  "defaults": {
    "strength": 1.0,
    "grainAmount": 25,
    "grainSize": 1.0,
    "saturation": 1.1
  }
}
```
- `baseCurve`: References an internal tone curve resource by name.
- `matrix`: 3×3 color correction matrix as a flat array of 9 floats (row-major). Optional.
- **Requirement:** Ship at least one manual example (e.g., "Chrome-like") to validate the engine. No complex authoring tool needed for v1.

---

## 6. AI Editing via Nano Banana Pro (Gemini)

### 6.1 AI scope
AI features operate on a **rendered raster** (not RAW mosaic) and return a raster result:
- Natural language edit (“make it golden hour”)
- Remove object / cleanup
- Replace background
- Relight
- Generative expand (outpainting) where supported

Nano Banana is Gemini’s native image generation/editing capability; Nano Banana Pro is available in the Gemini model lineup.

#### 6.1.1 Gemini API contract
- **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
- **Model identifier:** configured by backend env var `GEMINI_MODEL` (default: `nano-banana-pro`).
- The exact model identifier sent upstream must be persisted in the AI audit trail and in `aiLayers[].meta.model`.
- **Request format:** Multipart `generateContent` request with:
  - `contents[].parts[]`: text prompt + inline image data (base64-encoded rendered raster, JPEG, max 4MP for performance).
  - `generationConfig.responseModalities: ["IMAGE", "TEXT"]`
  - `generationConfig.responseMimeType: "image/png"`
- **Response format:** `candidates[].content.parts[]` containing an `inlineData` part with `mimeType` and base64 `data`.
- **Supported `mode` values** (mapped to prompt engineering strategies by the backend):
  - `"edit"` — natural language edit instruction
  - `"remove"` — object removal (prompt describes what to remove)
  - `"replace_bg"` — background replacement
  - `"relight"` — relighting instruction
  - `"expand"` — outpainting (with aspect ratio in `options`)
- **Error mapping:** HTTP 429 → `AI_QUOTA_EXCEEDED`, HTTP 401/403 → `AI_INVALID_KEY`, HTTP 400 → `AI_BAD_REQUEST`.

### 6.2 API key requirement
- User enters their own Gemini API key in Settings.
- Key modes:
  - **Session-only**: never persisted (default).
  - **Remember**: encrypted at rest in browser storage using WebCrypto, protected by user passphrase.

### 6.3 AI calling model (v1 decision)
**Mandatory:** AI proxy in Rust backend
- Browser sends request to backend with key in a secure, non-logged header (`X-Gemini-Key`).
- Backend forwards to Gemini API.
- Backend must **not** persist keys.
- Backend must **not** persist prompt text server-side; logs must redact secrets.
- Direct browser-to-Gemini calls are **disabled** in v1 to ensure security and auditability.

### 6.4 Non-destructive integration
AI output is added as an **AI Layer** above RAW base:
- Opacity
- Blend mode (at least Normal; optionally Multiply/Screen/Overlay)
- Optional mask (painted)
- Stored prompt (or prompt hash in redacted mode) + parameters + provider metadata for audit/re-run

### 6.5 AI audit trail
Persist in project state (frontend IndexedDB and `.wrp` export only):
- Provider: `gemini`
- Model: exact model identifier used for the request (e.g., `nano-banana-pro`)
- Prompt + negative prompt
- Options (strength, aspect, seed if any)
- Timestamps
- Input render settings used (size, color space)
- Output asset hash (content-addressed)
- Optional privacy mode: when "Redact prompt text" is enabled, store `promptHash` / `negativePromptHash` instead of raw prompt strings.
- `EditState.aiLayers[].meta` supports both raw prompt fields and hash-only fields.

---

## 7. UX / UI Requirements (Fluent + Lightroom-like)

### 7.1 Layout
- **Top command bar:** Import, Export, Undo/Redo, Compare, Copy/Paste edits
- **Left panel:** Library (folders/albums, filters)
- **Center:** Canvas (zoom/pan, overlays)
- **Right panel:** Develop controls in collapsible sections:
  - Basic
  - Tone Curve
  - Color (HSL + grading)
  - Detail (sharpen/denoise)
  - Optics
  - Geometry
  - Effects
  - Film Sims
  - AI Edit
  - History / Snapshots

### 7.2 Interaction patterns
- Sliders with:
  - drag, shift-fine adjust
  - double-click reset
  - numeric entry field
- Presets:
  - hover preview (optional v1)
  - apply/create/update
- Compare:
  - before/after split
  - reference A/B (optional v1)

### 7.3 Responsive behavior
- Desktop: full layout
- Tablet: collapsible panels/drawers
- Mobile browser: simplified Develop-first mode with drawer panels

### 7.4 Accessibility
- Keyboard navigation for panels/sliders
- ARIA labeling
- Visible focus states, WCAG AA contrast targets

---

## 8. System Architecture

### 8.1 Components
**Frontend**
- React + TypeScript
- Fluent UI React v9 (Fluent 2)
- WebGL2 renderer for interactive preview; Canvas2D fallback
- Web Workers for histogram, thumbnails, mask rasterization

**Backend (Rust) — Stateless for metadata; ephemeral file cache only**
- RAW Processing Service (decode + render + export)
- AI Proxy Service (mandatory in v1)
- Ephemeral File Cache: uploaded files are stored temporarily (configurable TTL, default 24h) and referenced by opaque `fileId`. No persistent database in v1.

#### 8.1.1 Rendering architecture split (Frontend vs. Backend)

The 18-stage image pipeline (§9.2) is split between backend and frontend:

**Primary WebGL path: Backend renders stages 1–9 ("base render"):**
- EXIF orientation → RAW decode → black level → white balance → demosaic → highlight reconstruction → denoise (pre-color) → lens correction → base camera profile.
- Output: a 16-bit linear Rec.2020 image (PNG) at preview resolution (long edge ≤ 2560px) or full resolution (for export).
- Cached by `assetHash + whiteBalance + denoise + optics` parameters.

**Primary WebGL path: Frontend applies stages 10–17 via WebGL2 shaders on the base render texture:**
- Tone mapping → color (HSL, grading, vibrance) → film sim (LUT + curve + matrix) → detail (sharpen) → effects (vignette + grain) → local adjustments → AI layer compositing → output transform + gamma.
- This enables sub-150ms slider response without backend round-trips.
- Film sim LUTs are uploaded as 3D textures to the GPU.

**Export always uses the full backend pipeline (stages 1–18)** for deterministic, full-resolution, 16-bit-internal output.

**Canvas2D fallback:** When WebGL2 is unavailable, the frontend sends the full `EditState` to the backend for each preview render via `POST /api/renders/preview`. This path is slower but functionally identical. In this fallback path, `POST /api/renders/preview` applies stages 1–17 backend-side and returns a display-referred preview image.

**Storage**
- **Frontend (Source of Truth for all persistent data):**
  - IndexedDB for project metadata, edit states, AI layer raster assets, and user settings.
  - File System Access API for reading/writing original RAWs and exports (where supported).
  - Fallback: file upload to backend ephemeral cache for processing.
  - **Quota management:** Monitor storage via `navigator.storage.estimate()`. Warn the user at 80% capacity. Implement LRU eviction of AI layer raster assets (metadata is retained for re-generation). Provide a manual "Purge AI Assets" action in Settings.
- **Backend (Ephemeral only):**
  - Uploaded originals cached on disk with TTL (default 24h, configurable).
  - Rendered previews cached via content-addressable keys (see §8.4).
  - No persistent database, no user accounts in v1.
- Server mode (v2+):
  - Postgres for catalog (projects/assets).
  - Object storage (S3/GCS/Azure Blob) for originals and renditions.

### 8.2 RAW engine choice
- Use **LibRaw** for RAW decoding/demosaic via Rust FFI bindings (bindgen).
- License considerations: LibRaw is dual-licensed LGPL 2.1 or CDDL; choose a compliant approach for distribution.

#### 8.2.1 Camera profile source
- **Default:** Use LibRaw's built-in dcraw-compatible color matrices (`libraw_internal_data.color`) for each camera model. These provide the 3×3 matrix for pipeline stage 9 (base camera profile).
- **Tone curve:** Use a default medium-contrast S-curve for all cameras in v1. Camera-specific tone curves are a v2 enhancement.
- **Fallback:** If LibRaw does not have a color matrix for a given camera, use the identity matrix and log a warning. The image will render but with uncorrected color.
- **Future (v2+):** Support loading Adobe DNG Camera Profiles (`.dcp`) for per-camera, per-illuminant dual-matrix interpolation.

### 8.3 Rendering strategy (two paths)
1) **Interactive Preview (fast)**
- Generate long-edge preview (e.g., 2560px) quickly.
- Cache “base render” (demosaic + base profile) and reapply adjustments deltas.
- Progressive refinement allowed:
  - quick low-res render within 200ms
  - refine in 1–2s to full preview

2) **Export Render (full-quality)**
- Full resolution
- 16-bit internal pipeline
- Deterministic results (same params → same output)

3) **Progressive base-render delivery (SSE, WebGL path only)**
- When a new base render is needed (e.g., denoise or optics param changed), the frontend opens an SSE connection: `GET /api/renders/base-render/stream?fileId=...&baseRenderHash=...`
- The backend streams events:
  - `event: fast` — low-res base render (long edge ≤ 1280px), delivered within 200ms.
  - `event: done` — full preview-resolution base render (long edge ≤ 2560px), delivered within 1–2s.
  - `event: error` — error with ProblemDetail payload.
- Each event's `data` field contains base64-encoded image data.
- This SSE stream returns **stage 1–9 linear base renders only** (for WebGL compositing), not display-ready previews.
- Frontend can also use the synchronous `POST /api/renders/preview` with `qualityHint: "FAST"` for simple cases (e.g., Canvas2D fallback).

### 8.4 Caching
- Content-addressable caching keyed by:
  - `assetHash` + `editStateHash` + `renderSettings`
- Cache tiers:
  - in-memory (LRU) for hot previews
  - disk cache for preview pyramids/tiles
- Invalidation:
  - any edit param change creates a new cache key

#### 8.4.1 editStateHash computation
- `editStateHash` = SHA-256 of a canonical JSON serialization of the `EditState`, **excluding** `history` and `snapshots` fields.
- Canonical form: keys sorted alphabetically at every level, no whitespace, numbers serialized without trailing zeros.
- AI layer references contribute their `assetId` (which is itself a content hash), not binary pixel data.
- For **base render caching** (stages 1–9), compute a separate `baseRenderHash` using only the params that affect those stages: `assetHash + temperature + tint + denoise.luma + denoise.chroma + denoise.enhanced + optics.*`.

### 8.5 Performance targets
- Library scroll: 60 fps with virtualization
- First preview open: < 2s for typical RAW
- Slider latency: first visible response < 150ms (progressive ok)
- Batch export: scalable with concurrency controls

---

## 9. Detailed Image Pipeline (v1)

### 9.1 Internal working space
- **Locked decision:** Linear Rec.2020 (ITU-R BT.2020 primaries, D65 white point, linear gamma / scene-referred).
- All internal computation, LUT authoring, and camera profile matrices target this space.
- Output transforms: sRGB (default), Display P3 (optional). Both use the appropriate 3×3 gamut-mapping matrix + gamma curve (sRGB transfer function or Display P3 TRC).

### 9.2 Pipeline stages
1) EXIF orientation: read EXIF `Orientation` tag and apply rotation/flip to produce a canonically-oriented image. All subsequent stages operate on the oriented image. Coordinates in the data model (§10.0) reference the post-orientation dimensions.
2) RAW decode
3) Black level subtraction
4) White balance (camera multipliers + user temp/tint)
5) Demosaic
6) Highlight reconstruction (blend method: interpolate clipped channels from unclipped channels using channel correlation; clip if all channels are clipped)
7) Denoise (pre-color)
8) Lens correction (distortion/vignette/CA)
9) Base camera profile (matrix + curve)
10) Tone mapping (exposure/contrast + curve)
11) Color (HSL, grading, vibrance)
12) Film sim (LUT + tone curve + color matrix — inserted after color, before detail)
13) Detail (sharpen)
14) Effects (vignette + grain)
15) Local adjustments (masked)
16) AI layer compositing (blend AI raster layers in display-referred space; see §6.4 for blend modes)
17) Output transform + gamma (linear Rec.2020 → sRGB or Display P3)
18) Encode export

**Grain precedence:** When a Film Sim is active, its `grainAmount`/`grainSize` values **replace** the manual Effects grain. The Effects `vignetteAmount` is always applied independently. Film sims never override any Effects field other than grain. This avoids double-grain artifacts.

**AI layer compositing:** AI layers are composited at stage 16, after local adjustments and before the output transform. Since AI operations produce display-referred sRGB rasters (from the Gemini API), AI layer pixels are inverse-gamma-decoded to linear and gamut-mapped to the working space before blending. The output transform (stage 17) then converts the composited result to the final display/export space.

### 9.3 Histograms & clipping
- RGB + luminance histograms computed on preview
- Highlight/shadow clipping indicators

---

## 10. Data Model (TypeScript)

### 10.0 Coordinate system convention
All spatial coordinates throughout the data model (crop rects, mask positions, gradient endpoints, brush strokes, heal/clone points) use **normalized [0, 1] coordinates** relative to the **post-EXIF-orientation** image dimensions (i.e., after stage 1 orientation correction), **before** any lens correction or crop is applied. This ensures masks and crops are resolution-independent and stable across pipeline changes.

- `(0, 0)` = top-left of the oriented image.
- `(1, 1)` = bottom-right of the oriented image.
- Brush `pressure` remains in [0, 1] as an opacity/flow multiplier.
- Crop `rect` uses the same coordinate space: `{ x: 0.1, y: 0.1, w: 0.8, h: 0.8 }` crops 10% from each edge.
- When lens corrections change, masks do **not** need to be recomputed — the pipeline applies lens correction first (stage 7), then maps mask coordinates through the same transform.

### 10.1 Entities

#### Asset
```ts
type Asset = {
  id: string;
  filename: string;
  mime: string;
  sourceType: "RAW" | "RASTER";
  originalUri: string; // local handle id or remote URL
  createdAt: string;
  exif?: Record<string, unknown>; // curated subset
  hash?: string; // SHA-256 of original bytes (when available)
};
```

#### Mask Types
```ts
type HslChannel = "red" | "orange" | "yellow" | "green" | "aqua" | "blue" | "purple" | "magenta";

type BrushMask = {
  type: "BRUSH";
  strokes: Array<{
    points: Array<{ x: number; y: number; pressure: number }>;
    size: number;
    feather: number;
    flow: number;
    erase: boolean;
  }>;
};

type LinearGradientMask = {
  type: "LINEAR";
  start: { x: number; y: number };
  end: { x: number; y: number };
  feather: number;
};

type RadialGradientMask = {
  type: "RADIAL";
  center: { x: number; y: number };
  radiusX: number;
  radiusY: number;
  rotation: number;
  feather: number;
  invert: boolean;
};

type HealCloneMask = {
  type: "HEAL" | "CLONE";
  sourcePoint: { x: number; y: number };
  targetPoint: { x: number; y: number };
  radius: number;
  feather: number;
};

type MaskDefinition = BrushMask | LinearGradientMask | RadialGradientMask | HealCloneMask;

/** Subset of global params that can be applied as local adjustments.
 *  Excludes: crop, transform, toneCurve, hsl, grading, optics, effects (structurally complex or not meaningful locally). */
type LocalAdjustmentParams = {
  exposure?: number;
  contrast?: number;
  highlights?: number;
  shadows?: number;
  whites?: number;
  blacks?: number;
  temperature?: number;
  tint?: number;
  vibrance?: number;
  saturation?: number;
  texture?: number;
  clarity?: number;
  dehaze?: number;
  sharpeningAmount?: number;
  denoiseLuma?: number;
};
```

#### EditState
```ts
type EditState = {
  schemaVersion: number;
  assetId: string;

  global: {
    exposure: number;
    contrast: number;
    highlights: number;
    shadows: number;
    whites: number;
    blacks: number;
    temperature: number;
    tint: number;
    vibrance: number;
    saturation: number;
    texture: number;
    clarity: number;
    dehaze: number;
    toneCurve: {
      mode: "POINT" | "PARAMETRIC";
      points: Array<{ x: number; y: number }>;
      parametric?: { highlights: number; lights: number; darks: number; shadows: number };
    };
    hsl: Record<HslChannel, { h: number; s: number; l: number }>;
    grading?: {
      shadows: { h: number; s: number; l: number };
      mids: { h: number; s: number; l: number };
      highlights: { h: number; s: number; l: number };
      balance: number;
    };
    sharpening: { amount: number; radius: number; detail: number; masking: number };
    denoise: { luma: number; chroma: number; enhanced?: boolean };
    optics: { enable: boolean; distortion: number; vignette: number; ca: number };
    crop?: { angle: number; rect: { x: number; y: number; w: number; h: number } };
    transform?: { vertical: number; horizontal: number; rotate: number; aspect: number };
    effects?: { grainAmount: number; grainSize: number; vignetteAmount: number };
  };

  filmSim?: {
    id: string;
    strength: number; // 0..2.0
    grainAmount: number;
    grainSize: number;
    bwFilter?: "R" | "Y" | "G";
  };

  localAdjustments: Array<{
    id: string;
    type: "BRUSH" | "LINEAR" | "RADIAL" | "HEAL" | "CLONE";
    mask: MaskDefinition;
    params: LocalAdjustmentParams; // ignored for HEAL/CLONE types
    enabled: boolean;
  }>;

  aiLayers: Array<{
    id: string;
    assetId: string; // raster output stored in frontend IndexedDB
    opacity: number; // 0..1
    blendMode: "NORMAL" | "MULTIPLY" | "SCREEN" | "OVERLAY";
    maskId?: string;
    meta: {
      provider: "gemini";
      model: string; // exact upstream model identifier from GEMINI_MODEL
      prompt?: string; // omitted when redacted mode stores hashes only
      negativePrompt?: string;
      promptHash?: string; // SHA-256 of prompt when redacted mode is enabled
      negativePromptHash?: string; // SHA-256 of negative prompt when redacted mode is enabled
      createdAt: string;
      options?: Record<string, unknown>;
    };
    enabled: boolean;
  }>;

  history: {
    headId: string;
    nodes: Record<string, {
      id: string;
      parentId?: string;
      label: string;
      snapshot: Omit<EditState, "history" | "snapshots">; // excludes history and snapshots to avoid recursive bloat
      ts: string;
    }>;
  };

  snapshots: Array<{ id: string; name: string; stateHash: string; createdAt: string }>;
};
```

### 10.2 EditState versioning
- Include `schemaVersion` at root.
- Provide forward-compatible migration strategy (frontend + backend).

### 10.2.1 History pruning
- **Runtime constant** `MAX_HISTORY_NODES = 200`. Not stored in the EditState type — enforced by the history management logic.
- When the node count exceeds this limit, prune the oldest leaf branches first, always preserving the main chain (the path from root to current `headId`).
- Named snapshots (§10.1 `snapshots`) are stored separately and are **never pruned** by history eviction.

### 10.3 Project file format (`.wrp` — Web RAW Project)
A project file is a **ZIP archive** (with `.wrp` extension) containing:
```
project.wrp (ZIP)
├── manifest.json         # { version, createdAt, assets: [{ id, filename, originalPath, hash }] }
├── editstates/
│   └── {assetId}.json    # Full EditState per image (history pruned to MAX_HISTORY_NODES = 200 before export)
├── ai-assets/
│   └── {assetId}.png     # AI layer raster outputs (referenced by aiLayers[].assetId)
└── presets/
    └── {presetId}.json   # User-created presets
```
- **Original RAW/raster files are NOT included** — they are referenced by `originalPath` (relative or absolute) and `hash` (SHA-256). On re-open, the app attempts to locate originals; if missing, it prompts the user to re-link.
- **Import/Export:** "Save Project" serializes from IndexedDB to `.wrp`. "Open Project" deserializes back into IndexedDB.
- Maximum project file size: warn at 500MB (driven by AI assets).

---

## 11. Rust Backend Specification

### 11.1 Rust stack (recommended)
- Runtime: `tokio`
- HTTP: `axum` (or `actix-web`)
- Serialization: `serde`, `serde_json`
- Observability: `tracing`, `tracing-subscriber`, OpenTelemetry optional
- Auth (if server mode): `jsonwebtoken`, `argon2` for password-based key wrapping
- DB (server mode): `sqlx` (Postgres)
- Object storage: vendor SDK or S3-compatible client
- Image encode/decode: `image` crate for PNG/JPEG, plus TIFF support as needed
- FFI: `bindgen` for LibRaw bindings

### 11.1.1 Build & Environment Requirements
- **LibRaw Linking:** Statically linked (`libraw.a`) to avoid runtime dependency hell on deployment.
- **Toolchain:**
  - `clang` / `llvm` (required by bindgen)
  - `cmake` (for building LibRaw from source)
  - `c++` compiler (gcc/msvc)
- **Container:** Dockerfile must use a multi-stage build:
  1. `builder`: Install cmake, clang, build LibRaw static lib.
  2. `runtime`: Minimal distroless or alpine image (if static linking succeeds) or debian-slim.

### 11.2 Services (logical)
1) **rawd** (RAW decode, render, and export — stateless computation)
2) **cache** (ephemeral file cache — accepts uploads, serves cached files, TTL-based cleanup)
3) **ai-proxy** (Gemini forwarder — mandatory in v1)

These run as a single Rust binary in v1; may be split in v2+.

### 11.3 API endpoints (HTTP/JSON)

#### 11.3.0 API versioning
- All endpoints are prefixed with `/api/v1/` (e.g., `/api/v1/files/upload`). Endpoint definitions below omit the `/v1/` prefix for brevity.
- Breaking changes require a new version prefix (`/api/v2/`). Non-breaking additions (new optional fields, new endpoints) do not.
- The backend serves a version-unscoped `GET /api/version` endpoint returning `{ apiVersion: "v1", buildHash: string }`.

#### File upload (ephemeral cache)
- `POST /api/files/upload`
  - Multipart upload (original RAW or raster file).
  - response: `{ fileId: string, exif: Record<string, unknown>, sizeBytes: number }`
  - The file is stored in the ephemeral cache; `fileId` is used by all subsequent endpoints.

- `GET /api/files/{fileId}`
  - Returns the cached file bytes (original upload or generated artifact).
  - response: binary file (`Content-Type` inferred from stored MIME type, includes `Content-Length` and `ETag` headers).

- `DELETE /api/files/{fileId}`
  - Explicitly remove a cached file before TTL expiry.

#### Thumbnails
- `POST /api/renders/thumbnail`
  - body: `{ fileId, maxEdge: number }`
  - response: binary image (`Content-Type: image/jpeg`)

#### Preview render
- `POST /api/renders/preview`
  - body: `{ fileId, editState, maxEdge, colorSpace: "sRGB"|"DisplayP3", qualityHint?: "FAST"|"BALANCED"|"BEST" }`
  - FAST mode: synchronous — returns display-referred JSON: `{ imageBase64, mimeType: "image/png", bitDepth: 8, colorSpace: "sRGB"|"DisplayP3", width, height, histogram }`.
  - BEST mode: returns `{ jobId }` → poll via `GET /api/jobs/{jobId}`; final preview result uses the same payload shape as FAST mode.

#### Export
- `POST /api/renders/export`
  - body: `{ fileId, editState, format: "JPG"|"PNG"|"TIFF", bitDepth?: 8|16, quality?: number, colorSpace }`
  - response: `{ jobId }` → poll via `GET /api/jobs/{jobId}` → final result: `{ downloadUrl, sizeBytes, checksum }`.

#### AI edit (proxy)
- `POST /api/ai/edit`
  - headers: `X-Gemini-Key: <user_key>` (never logged)
  - body: `{ fileId, editState, prompt, negativePrompt?, mode, options }`
  - Backend renders the current editState to a raster, sends to Gemini API, returns result.
  - response: `{ jobId }` → poll via `GET /api/jobs/{jobId}` → final: `{ resultFileId, meta }`
  - Frontend downloads the result via `GET /api/files/{resultFileId}` and stores it in IndexedDB.

#### Progressive base-render stream (SSE, WebGL path only)
- `GET /api/renders/base-render/stream`
  - query params: `fileId`, `baseRenderHash`, `maxEdge`
  - Opens an SSE connection. Backend streams:
    - `event: fast` — `data: { imageBase64, mimeType: "image/png", bitDepth: 16, colorSpace: "linear-rec2020", width, height }` (low-res, ≤ 1280px long edge)
    - `event: done` — `data: { imageBase64, mimeType: "image/png", bitDepth: 16, colorSpace: "linear-rec2020", width, height, histogram }` (full preview res, ≤ 2560px)
    - `event: error` — `data: ProblemDetail`
  - This endpoint returns stage 1–9 linear base renders for the WebGL path; it does not return display-ready previews.
  - Connection auto-closes after `done` or `error`.
  - If the requested `baseRenderHash` is already cached, only `done` is sent (no `fast`).
  - **Timeout:** Connection is server-closed after 30 seconds with no event (safety net for stalled renders).
  - **Reconnection:** If the SSE connection drops, the frontend should **not** auto-reconnect with `EventSource` retry. Instead, re-issue the request only if the user's edit state still requires that base render (i.e., the `baseRenderHash` hasn't changed due to further edits). This avoids wasted renders for stale states.

#### Job polling (long-running operations)
- `GET /api/jobs/{jobId}`
  - response is discriminated by `kind`:
    - Preview job: `{ jobId, kind: "PREVIEW", status: "PENDING"|"PROCESSING"|"COMPLETE"|"FAILED", progress?: number, result?: { imageBase64, mimeType: "image/png", bitDepth: 8, colorSpace: "sRGB"|"DisplayP3", width, height, histogram? }, error?: ProblemDetail }`
    - Export job: `{ jobId, kind: "EXPORT", status: "PENDING"|"PROCESSING"|"COMPLETE"|"FAILED", progress?: number, result?: { downloadUrl, sizeBytes, checksum }, error?: ProblemDetail }`
    - AI edit job: `{ jobId, kind: "AI_EDIT", status: "PENDING"|"PROCESSING"|"COMPLETE"|"FAILED", progress?: number, result?: { resultFileId, meta: { provider: "gemini", model: string, prompt?: string, promptHash?: string, createdAt: string } }, error?: ProblemDetail }`
  - Client should poll with exponential backoff starting at 500ms.

**Notes**
- All render/AI endpoints require a valid `fileId`. Return `404` with code `FILE_NOT_FOUND` if expired or missing.
- Enforce strict size limits (default 100MB per file) and MIME type validation.
- Rate-limit per session (configurable). Defaults:
  - Render endpoints: 30 requests/minute per session token.
  - AI proxy: 10 requests/minute per session token.
  - File upload: 60 requests/minute per session token.
- All responses include an `X-Request-Id` header.

### 11.4 Error model
- Use problem+json style with specific error codes:
```json
{
  "type": "https://example.com/problems/raw_decode_error",
  "title": "RAW Decode Failed",
  "status": 422,
  "detail": "LibRaw returned error -7 (LIBRAW_DATA_ERROR)",
  "code": "RAW_CORRUPT_DATA",
  "requestId": "..."
}
```
- **Required Error Codes:**
  - `FILE_NOT_FOUND`: `fileId` missing, expired, or inaccessible.
  - `RAW_UNSUPPORTED_CAMERA`: LibRaw cannot decode this specific make/model.
  - `RAW_CORRUPT_DATA`: File bytes are truncated or invalid.
  - `RENDER_TIMEOUT`: Processing exceeded time limit.
  - `AI_BAD_REQUEST`: Upstream Gemini request payload or parameters are invalid.
  - `AI_QUOTA_EXCEEDED`: Upstream Gemini API rejected request.
  - `AI_INVALID_KEY`: User API key rejected.

### 11.5 Concurrency & resource limits
- Per-request CPU limits via worker pool / semaphore.
- Render jobs executed in bounded tokio task pool.
- Configurable maximum parallel exports and previews.

### 11.6 Security requirements
- Never log:
  - RAW bytes
  - user API keys
  - AI prompts (unless explicit opt-in debug mode)
- Support “strip location metadata” on export (default ON).
- Validate all inputs; reject path traversal and malformed file headers.

#### 11.6.1 Session token (CSRF mitigation)
- On first load, the frontend generates a cryptographically random 128-bit session token (via `crypto.getRandomValues`) and sends it as `X-Session-Token` header on all requests.
- Backend validates that `X-Session-Token` is present and non-empty on all mutating requests (`POST`, `DELETE`). Requests without the header are rejected with `403`.
- Combined with CORS `Access-Control-Allow-Origin` restrictions (§11.7), this prevents cross-origin drive-by requests.
- The token is not cryptographically verified by the backend — its purpose is solely CSRF prevention, not authentication. No user accounts exist in v1.

### 11.7 CORS
- Development: allow explicit local origins from `ALLOWED_ORIGINS` (comma-separated). Default: `http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173`.
- Production: restrict `Access-Control-Allow-Origin` to the deployed frontend origin(s) via `ALLOWED_ORIGINS`.
- Allowed headers: `Content-Type`, `X-Gemini-Key`, `X-Request-Id`, `X-Session-Token`.
- Allowed methods: `GET`, `POST`, `DELETE`, `OPTIONS`.

---

## 12. Frontend Implementation Requirements

### 12.1 Tech stack
- React + TypeScript
- Fluent UI React v9
- State: Zustand
- Data fetching: TanStack Query (React Query)
- Rendering:
  - WebGL2 shader pipeline for adjustments
  - Canvas2D fallback

### 12.2 Web Workers
- Thumbnail decode (when using raster sources)
- Histogram computation
- Mask rasterization

### 12.3 File access
- Prefer File System Access API when available (Chromium-based browsers)
- Fallback to upload-based import (Firefox, Safari, and other browsers)
- For privacy, default to local-first storage unless user opts into server catalog

#### 12.3.1 Non-Chromium fallback behavior
When the File System Access API is unavailable:
- **Import:** User selects files via standard `<input type="file">`. Files are uploaded to the backend ephemeral cache. Backend TTL is extended to **72 hours** for uploaded originals (vs. 24h default) to reduce re-upload friction.
- **Library grid:** Thumbnail grid works identically — thumbnails are generated from cached files and stored in IndexedDB.
- **Develop:** Base renders are generated from the backend-cached file. The `fileId` is the same opaque reference.
- **Export:** Exported files are downloaded via standard browser download (no direct filesystem write).
- **UX:** Display a non-dismissable info banner: *"For the best experience (direct file access, faster imports), use a Chromium-based browser."* Show which features are reduced.
- **Re-upload detection:** On re-open, if a `fileId` has expired, prompt the user to re-upload the original file. Match by `hash` (SHA-256) to confirm identity.

---

## 13. Testing & Quality

### 13.1 Test frameworks
- **Frontend:** Vitest + React Testing Library. Playwright for E2E.
- **Backend:** `cargo test` (built-in). Golden image comparisons use `image` crate pixel-diff with configurable PSNR threshold (default ≥ 45 dB pass).

### 13.2 Test categories
- Unit:
  - EditState validation and migrations
  - LUT application math correctness
  - History branching and pruning at MAX_HISTORY_NODES
- Golden images:
  - Fixed RAW set → exported output within tolerances
  - Run via `cargo test --features golden` with reference images committed to `tests/golden/`
- Performance:
  - slider response metrics
  - thumbnail generation throughput
- Security:
  - ensure key redaction in logs
  - size-limit enforcement

### 13.3 Acceptance criteria (v1)
- Import 50 RAW images and display thumbnails successfully.
- Develop preview opens < 2s typical; subsequent adjustments show response < 150ms to first update.
- Film sims are visually distinct and stable across images.
- AI edit returns results and can be applied as AI Layer with saved prompt metadata.
- Export produces consistent output with correct color transform and optional metadata stripping.

---

## 14. Milestones

**M0 — Scaffolding**
- React + Fluent shell, routing, panels
- Import UI and thumbnail grid (mock rendering)

**M1 — Rust RAW render service**
- LibRaw decode/demosaic via FFI
- Preview endpoint + caching
- Export endpoint (JPG/PNG)

**M2 — Core adjustments**
- Basic, curve, HSL, crop/rotate, denoise/sharpen baseline
- Local masks (brush/gradients)

**M3 — Film sims**
- LUT engine + preset framework
- 6–10 “-like” looks with grain controls

**M4 — AI**
- Key management (session-only + encrypted store)
- AI proxy endpoint
- AI layer integration and history

**M5 — Polish**
- Compare modes, snapshots
- Batch export + batch preset apply
- Perf tuning, telemetry, documentation

---

## 15. Decisions locked for v1 (based on user request)
- Backend is **Rust**.
- Frontend is **React** and uses **Fluent UI React v9** for Fluent 2 styling.
- AI editing uses **Nano Banana Pro** through Gemini API image generation/editing, enabled only with user-provided key.
- Film sims are “Fuji-style” looks implemented with original LUT/curve stacks (no proprietary assets).

---
