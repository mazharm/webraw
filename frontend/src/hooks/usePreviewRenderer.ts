import { useState, useEffect, useRef, useCallback } from 'react';
import { renderPreview, type PreviewResult } from '../api/client';
import type { EditState, HistogramData } from '../types';

export function usePreviewRenderer(fileId: string | null, editState: EditState | null) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [histogram, setHistogram] = useState<HistogramData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const abortRef = useRef<AbortController>();
  const prevUrlRef = useRef<string | null>(null);

  const fetchPreview = useCallback(async () => {
    if (!fileId || !editState) return;

    // Cancel previous request
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setIsLoading(true);
    setError(null);

    try {
      const result = await renderPreview(fileId, editState, {
        maxEdge: 2560,
        colorSpace: 'sRGB',
        qualityHint: 'FAST',
        signal: abortRef.current?.signal,
      });

      // Check if it's a direct result
      if ('imageBase64' in result) {
        const preview = result as PreviewResult;
        const blob = base64ToBlob(preview.imageBase64, preview.mimeType);
        const url = URL.createObjectURL(blob);

        // Clean up old URL
        if (prevUrlRef.current) {
          URL.revokeObjectURL(prevUrlRef.current);
        }
        prevUrlRef.current = url;

        setPreviewUrl(url);
        setHistogram(preview.histogram ?? null);
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.detail ?? err.message ?? 'Preview render failed');
      }
    } finally {
      setIsLoading(false);
    }
  }, [fileId, editState]);

  // Debounce preview requests on edit state changes
  useEffect(() => {
    if (!fileId || !editState) return;

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(fetchPreview, 150);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [fetchPreview]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (prevUrlRef.current) {
        URL.revokeObjectURL(prevUrlRef.current);
      }
      abortRef.current?.abort();
    };
  }, []);

  const retry = useCallback(() => {
    fetchPreview();
  }, [fetchPreview]);

  return { previewUrl, histogram, isLoading, error, retry };
}

function base64ToBlob(b64: string, mimeType: string): Blob {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
}
