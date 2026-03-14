import init, {
  load_image_meta,
  generate_thumbnail as wasm_generate_thumbnail,
  render_preview as wasm_render_preview,
  render_export as wasm_render_export,
} from '../wasm-pkg/webraw_wasm';
import { get, set, del } from 'idb-keyval';
import type { EditState, HistogramData } from '../types';

let wasmReady = false;

export async function initEngine(): Promise<void> {
  if (!wasmReady) {
    await init();
    wasmReady = true;
  }
}

// ─── File Storage (IndexedDB) ───

export async function uploadFile(file: File): Promise<{
  fileId: string;
  exif: Record<string, unknown> | null;
  sizeBytes: number;
}> {
  await initEngine();
  const bytes = new Uint8Array(await file.arrayBuffer());
  const fileId = crypto.randomUUID();

  const metaJson = load_image_meta(bytes, file.name);
  const meta = JSON.parse(metaJson);

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

export async function generateThumbnail(
  fileId: string,
  maxEdge = 300
): Promise<Blob> {
  await initEngine();
  const bytes = await get(`file:${fileId}`) as Uint8Array;
  if (!bytes) throw new Error(`File ${fileId} not found in storage`);

  const meta = await get(`meta:${fileId}`) as { filename: string };
  const jpegBytes = wasm_generate_thumbnail(bytes, meta.filename, maxEdge);
  const blob = new Blob([new Uint8Array(jpegBytes)], { type: 'image/jpeg' });

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

export async function renderPreview(
  fileId: string,
  editState: EditState,
  options: { maxEdge?: number; signal?: AbortSignal } = {}
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

  const resultJson = wasm_render_preview(
    bytes,
    meta.filename,
    editJson,
    options.maxEdge ?? 2560
  );
  return JSON.parse(resultJson);
}

// ─── Export ───

export async function renderExport(
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

  const exportBytes = wasm_render_export(
    bytes,
    meta.filename,
    editJson,
    format,
    options.quality ?? 95,
    options.bitDepth ?? 8,
    options.colorSpace ?? 'sRGB',
  );

  const mimeMap: Record<string, string> = { JPG: 'image/jpeg', PNG: 'image/png', TIFF: 'image/tiff' };
  return new Blob([new Uint8Array(exportBytes)], { type: mimeMap[format] });
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

  const imgBinary = atob(b64);
  const imgBytes = new Uint8Array(imgBinary.length);
  for (let i = 0; i < imgBinary.length; i++) imgBytes[i] = imgBinary.charCodeAt(i);

  return { imageData: imgBytes, mimeType: 'image/png', model };
}
