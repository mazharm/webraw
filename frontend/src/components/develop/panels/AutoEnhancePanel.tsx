import { useState, useEffect, useRef, useCallback } from 'react';
import {
  tokens,
  Button,
  Dropdown,
  Option,
  Slider,
  Spinner,
  Text,
} from '@fluentui/react-components';
import { PaintBrushSparkle24Regular } from '@fluentui/react-icons';
import { PanelSection } from '../../common/PanelSection';
import { useEditStore } from '../../../stores/editStore';
import { useLibraryStore } from '../../../stores/libraryStore';
import { listEnhanceModels, runAutoEnhance, pollJob } from '../../../api/client';
import type { EnhanceModelDescriptor, GlobalAdjustments } from '../../../types';

export function AutoEnhancePanel() {
  const editState = useEditStore(s => s.editState);
  const updateGlobalPartial = useEditStore(s => s.updateGlobalPartial);
  const pushHistory = useEditStore(s => s.pushHistory);
  const activeAsset = useLibraryStore(s => s.assets.find(a => a.id === s.activeAssetId));

  const [models, setModels] = useState<EnhanceModelDescriptor[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('builtin-auto-all');
  const [strength, setStrength] = useState(100);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [adjustedParams, setAdjustedParams] = useState<Record<string, number> | null>(null);

  // Store original values and raw recommendations for live re-interpolation
  const originalRef = useRef<Partial<GlobalAdjustments> | null>(null);
  const recommendedRef = useRef<Record<string, number> | null>(null);

  // Fetch models on mount
  useEffect(() => {
    listEnhanceModels()
      .then(res => {
        setModels(res.models);
        if (res.models.length > 0 && !res.models.find(m => m.id === selectedModelId)) {
          setSelectedModelId(res.models[0].id);
        }
      })
      .catch(() => {/* models endpoint may not be available yet */});
  }, []);

  const applyInterpolated = useCallback(
    (recommended: Record<string, number>, original: Partial<GlobalAdjustments>, s: number) => {
      const factor = s / 100;
      const interpolated: Partial<GlobalAdjustments> = {};
      const display: Record<string, number> = {};

      for (const [key, recVal] of Object.entries(recommended)) {
        const origVal = (original as Record<string, unknown>)[key];
        if (typeof origVal === 'number') {
          const val = origVal + (recVal - origVal) * factor;
          (interpolated as Record<string, unknown>)[key] = Math.round(val * 100) / 100;
          display[key] = Math.round(val * 100) / 100;
        }
      }

      updateGlobalPartial(interpolated);
      setAdjustedParams(display);
    },
    [updateGlobalPartial],
  );

  const handleRun = async () => {
    if (!activeAsset?.fileId || !editState) return;

    setIsRunning(true);
    setError(null);
    setAdjustedParams(null);

    // Capture originals before applying
    const orig: Partial<GlobalAdjustments> = { ...editState.global };
    originalRef.current = orig;

    try {
      const { jobId } = await runAutoEnhance(activeAsset.fileId, selectedModelId);
      const job = await pollJob(jobId);

      if (job.status === 'FAILED') {
        setError(job.error?.detail ?? 'Auto enhance failed');
        return;
      }

      const result = job.result as { type: string; values?: Record<string, number> } | undefined;
      if (result?.type === 'Parameters' && result.values) {
        recommendedRef.current = result.values;
        applyInterpolated(result.values, orig, strength);
        pushHistory('Auto Enhance');
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Auto enhance failed');
    } finally {
      setIsRunning(false);
    }
  };

  // Live re-interpolation when strength slider changes
  const handleStrengthChange = (_: unknown, data: { value: number }) => {
    const s = data.value;
    setStrength(s);
    if (recommendedRef.current && originalRef.current) {
      applyInterpolated(recommendedRef.current, originalRef.current, s);
    }
  };

  if (!editState) return null;

  const selectedModel = models.find(m => m.id === selectedModelId);

  return (
    <PanelSection title="Auto Enhance" defaultOpen={false}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {models.length > 1 && (
          <Dropdown
            value={selectedModel?.name ?? 'Auto Enhance'}
            selectedOptions={[selectedModelId]}
            onOptionSelect={(_, data) => {
              if (data.optionValue) {
                setSelectedModelId(data.optionValue);
                setAdjustedParams(null);
                recommendedRef.current = null;
              }
            }}
            style={{ minWidth: 0 }}
          >
            {models.map(m => (
              <Option key={m.id} value={m.id} text={m.name}>
                {m.name}
              </Option>
            ))}
          </Dropdown>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Text size={200} style={{ minWidth: 52 }}>
            Strength
          </Text>
          <Slider
            min={0}
            max={100}
            value={strength}
            onChange={handleStrengthChange}
            style={{ flex: 1 }}
          />
          <Text size={200} style={{ minWidth: 28, textAlign: 'right' }}>
            {strength}%
          </Text>
        </div>

        <Button
          appearance="primary"
          icon={<PaintBrushSparkle24Regular />}
          onClick={handleRun}
          disabled={isRunning || !activeAsset?.fileId}
          style={{ width: '100%' }}
        >
          {isRunning ? <Spinner size="tiny" /> : 'Auto Enhance'}
        </Button>

        {error && (
          <Text size={200} style={{ color: tokens.colorPaletteRedForeground1 }}>
            {error}
          </Text>
        )}

        {adjustedParams && Object.keys(adjustedParams).length > 0 && (
          <div style={{ marginTop: 4 }}>
            <Text size={200} weight="semibold" style={{ color: tokens.colorNeutralForeground3 }}>
              Adjusted:
            </Text>
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '2px 8px',
                marginTop: 2,
              }}
            >
              {Object.entries(adjustedParams).map(([key, value]) => (
                <Text key={key} size={200} style={{ color: tokens.colorNeutralForeground3 }}>
                  {key}: {value}
                </Text>
              ))}
            </div>
          </div>
        )}
      </div>
    </PanelSection>
  );
}
