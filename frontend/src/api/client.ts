import type { EditState, JobInfo, ProblemDetail, HistogramData, EnhanceModelDescriptor } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8080/api/v1';

let sessionToken: string | null = null;

function getSessionToken(): string {
  if (!sessionToken) {
    const arr = new Uint8Array(16);
    crypto.getRandomValues(arr);
    sessionToken = Array.from(arr, b => b.toString(16).padStart(2, '0')).join('');
  }
  return sessionToken;
}

async function apiFetch(path: string, options: RequestInit = {}): Promise<Response> {
  const headers = new Headers(options.headers);
  headers.set('X-Session-Token', getSessionToken());

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    signal: options.signal,
  });

  if (!response.ok) {
    const error: ProblemDetail = await response.json().catch(() => ({
      type: 'unknown',
      title: 'Request Failed',
      status: response.status,
      detail: response.statusText,
      code: 'UNKNOWN',
      requestId: 'unknown',
    }));
    throw error;
  }

  return response;
}

async function apiJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && typeof options.body === 'string') {
    headers.set('Content-Type', 'application/json');
  }
  const response = await apiFetch(path, { ...options, headers });
  return response.json();
}

// File endpoints
export async function uploadFile(file: File, extendedTtl = false): Promise<{
  fileId: string;
  exif: Record<string, unknown> | null;
  sizeBytes: number;
  expiresAt: string;
}> {
  console.log(`[upload] Starting: ${file.name}, size=${file.size}, type="${file.type}"`);
  const formData = new FormData();
  formData.append('file', file);

  const ttlParam = extendedTtl ? '?ttlHint=extended' : '';
  try {
    const response = await apiFetch(`/files/upload${ttlParam}`, {
      method: 'POST',
      body: formData,
    });
    const result = await response.json();
    console.log(`[upload] Success: ${file.name} -> fileId=${result.fileId}`);
    return result;
  } catch (err) {
    console.error(`[upload] Failed: ${file.name}`, err);
    console.error(`[upload] API_BASE=${API_BASE}`);
    throw err;
  }
}

export async function getFileUrl(fileId: string): Promise<string> {
  return `${API_BASE}/files/${fileId}`;
}

export async function deleteFile(fileId: string): Promise<void> {
  await apiFetch(`/files/${fileId}`, { method: 'DELETE' });
}

// Thumbnail
export async function generateThumbnail(fileId: string, maxEdge = 300): Promise<Blob> {
  const response = await apiFetch('/renders/thumbnail', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileId, maxEdge }),
  });
  return response.blob();
}

// Preview
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
  options: { maxEdge?: number; colorSpace?: string; qualityHint?: string; signal?: AbortSignal } = {}
): Promise<PreviewResult | { jobId: string }> {
  return apiJson('/renders/preview', {
    method: 'POST',
    body: JSON.stringify({
      fileId,
      editState,
      maxEdge: options.maxEdge ?? 2560,
      colorSpace: options.colorSpace ?? 'sRGB',
      qualityHint: options.qualityHint ?? 'FAST',
    }),
    signal: options.signal,
  });
}

// Export
export async function renderExport(
  fileId: string,
  editState: EditState,
  format: 'JPG' | 'PNG' | 'TIFF',
  options: { bitDepth?: number; quality?: number; colorSpace?: string } = {}
): Promise<{ jobId: string }> {
  return apiJson('/renders/export', {
    method: 'POST',
    body: JSON.stringify({
      fileId,
      editState,
      format,
      bitDepth: options.bitDepth ?? 8,
      quality: options.quality ?? 95,
      colorSpace: options.colorSpace ?? 'sRGB',
    }),
  });
}

// AI edit
export async function createAiEdit(
  fileId: string,
  editState: EditState,
  prompt: string,
  mode: string,
  apiKey: string,
  options?: {
    negativePrompt?: string;
    idempotencyKey?: string;
    editOptions?: Record<string, unknown>;
  }
): Promise<{ jobId: string }> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Gemini-Key': apiKey,
  };
  if (options?.idempotencyKey) {
    headers['Idempotency-Key'] = options.idempotencyKey;
  }

  return apiJson('/ai/edit', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      fileId,
      editState,
      prompt,
      negativePrompt: options?.negativePrompt,
      mode,
      options: options?.editOptions,
    }),
  });
}

// Auto enhance
export async function listEnhanceModels(): Promise<{ models: EnhanceModelDescriptor[] }> {
  return apiJson('/auto-enhance/models');
}

export async function runAutoEnhance(
  fileId: string,
  modelId: string,
  options?: { strength?: number; signal?: AbortSignal; apiKey?: string; anthropicApiKey?: string }
): Promise<{ jobId: string }> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (options?.apiKey) {
    headers['X-Gemini-Key'] = options.apiKey;
  }
  if (options?.anthropicApiKey) {
    headers['X-Anthropic-Key'] = options.anthropicApiKey;
  }
  return apiJson('/auto-enhance/run', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      fileId,
      modelId,
      strength: options?.strength,
    }),
    signal: options?.signal,
  });
}

// Job polling
export async function getJob(jobId: string): Promise<JobInfo> {
  return apiJson(`/jobs/${jobId}`);
}

export async function pollJob(
  jobId: string,
  onProgress?: (progress: number) => void,
  signal?: AbortSignal,
  maxAttempts = 120,
): Promise<JobInfo> {
  let delay = 500;
  const maxDelay = 5000;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (signal?.aborted) throw new Error('Aborted');

    const job = await getJob(jobId);

    if (job.progress && onProgress) {
      onProgress(job.progress);
    }

    if (job.status === 'COMPLETE' || job.status === 'FAILED') {
      return job;
    }

    await new Promise(resolve => setTimeout(resolve, delay));
    delay = Math.min(delay * 1.5, maxDelay);
  }

  throw new Error(`Job ${jobId} did not complete after ${maxAttempts} polling attempts`);
}

// Health check
export async function healthCheck(): Promise<{ status: string; uptime: number }> {
  const response = await fetch(`${API_BASE.replace('/v1', '')}/health`);
  return response.json();
}

// SSE Base render stream
export function baseRenderStream(
  fileId: string,
  baseRenderHash: string,
  maxEdge = 2560
): EventSource {
  const params = new URLSearchParams({
    fileId,
    baseRenderHash,
    maxEdge: maxEdge.toString(),
  });
  return new EventSource(`${API_BASE}/renders/base-render/stream?${params}`);
}
