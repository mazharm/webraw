import { useState, useEffect, useRef, useCallback } from 'react';
import { renderPreview, type PreviewResult } from '../api/client';
import type { EditState, HistogramData } from '../types';

export function usePreviewRenderer(fileId: string | null, editState: EditState | null) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [histogram, setHistogram] = useState<HistogramData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const prevUrlRef = useRef<string | null>(null);
  const generationRef = useRef(0);

  const fetchPreview = useCallback(async () => {
    if (!fileId || !editState) return;

    const gen = ++generationRef.current;

    setIsLoading(true);
    setError(null);

    const apiEditState = {
      schemaVersion: editState.schemaVersion,
      assetId: editState.assetId,
      global: editState.global,
      localAdjustments: editState.localAdjustments,
      aiLayers: editState.aiLayers,
      filmSim: editState.filmSim,
    };

    try {
      const result = await renderPreview(fileId, apiEditState as EditState, {
        maxEdge: 2560,
      });

      if (gen !== generationRef.current) return;

      if ('imageBase64' in result) {
        const preview = result as PreviewResult;
        const blob = base64ToBlob(preview.imageBase64, preview.mimeType);
        const url = URL.createObjectURL(blob);

        if (prevUrlRef.current) {
          URL.revokeObjectURL(prevUrlRef.current);
        }
        prevUrlRef.current = url;

        setPreviewUrl(url);
        setHistogram(preview.histogram ?? null);
      }
    } catch (err: any) {
      if (gen !== generationRef.current) return;
      setError(err.detail ?? err.message ?? 'Preview render failed');
    } finally {
      if (gen === generationRef.current) {
        setIsLoading(false);
      }
    }
  }, [fileId, editState]);

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

  useEffect(() => {
    return () => {
      if (prevUrlRef.current) {
        URL.revokeObjectURL(prevUrlRef.current);
      }
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
