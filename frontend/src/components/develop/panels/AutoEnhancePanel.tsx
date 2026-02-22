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
import { useSettingsStore } from '../../../stores/settingsStore';
import { listEnhanceModels, runAutoEnhance, pollJob } from '../../../api/client';
import type { EnhanceModelDescriptor, GlobalAdjustments } from '../../../types';

export function AutoEnhancePanel() {
  const editState = useEditStore(s => s.editState);
  const updateGlobalPartial = useEditStore(s => s.updateGlobalPartial);
  const pushHistory = useEditStore(s => s.pushHistory);
  const activeAsset = useLibraryStore(s => s.assets.find(a => a.id === s.activeAssetId));
  const geminiApiKey = useSettingsStore(s => s.geminiApiKey);
  const anthropicApiKey = useSettingsStore(s => s.anthropicApiKey);

  const [models, setModels] = useState<EnhanceModelDescriptor[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('builtin');
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
      const { jobId } = await runAutoEnhance(activeAsset.fileId, selectedModelId, {
        apiKey: geminiApiKey ?? undefined,
        anthropicApiKey: anthropicApiKey ?? undefined,
      });
      const job = await pollJob(jobId);

      if (job.status === 'FAILED') {
        setError(job.error?.detail ?? 'Auto fix failed');
        return;
      }

      const result = job.result as { type: string; values?: Record<string, number> } | undefined;
      if (result?.type === 'Parameters' && result.values) {
        recommendedRef.current = result.values;
        applyInterpolated(result.values, orig, strength);
        pushHistory('Auto Fix');
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Auto fix failed');
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
  const needsKey = selectedModel?.requiresApiKey && (
    selectedModel.id === 'claude'
      ? !anthropicApiKey
      : !geminiApiKey
  );

  return (
    <PanelSection title="Auto Fix" defaultOpen>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div>
          <Text size={200} weight="semibold" style={{ display: 'block', marginBottom: 4 }}>
            Provider
          </Text>
          <Dropdown
            value={selectedModel?.name ?? 'Loading...'}
            selectedOptions={[selectedModelId]}
            onOptionSelect={(_, data) => {
              if (data.optionValue) {
                setSelectedModelId(data.optionValue);
                setAdjustedParams(null);
                recommendedRef.current = null;
              }
            }}
            style={{ minWidth: 0, width: '100%' }}
          >
            {models.map(m => (
              <Option key={m.id} value={m.id} text={m.name}>
                <div>
                  <Text size={300} weight="semibold">{m.name}</Text>
                  {m.publisher && (
                    <Text
                      size={200}
                      style={{
                        display: 'block',
                        color: tokens.colorNeutralForeground4,
                      }}
                    >
                      by {m.publisher}
                    </Text>
                  )}
                  <Text
                    size={200}
                    style={{
                      display: 'block',
                      color: tokens.colorNeutralForeground3,
                      marginTop: 2,
                    }}
                  >
                    {m.description}
                  </Text>
                </div>
              </Option>
            ))}
          </Dropdown>
        </div>

        {needsKey && (
          <Text
            size={200}
            style={{ color: tokens.colorPaletteYellowForeground1 }}
          >
            This provider requires {selectedModel?.id === 'claude' ? 'an Anthropic' : 'a Gemini'} API key. Set it in Settings.
          </Text>
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
          disabled={isRunning || !activeAsset?.fileId || needsKey}
          style={{ width: '100%' }}
        >
          {isRunning ? <Spinner size="tiny" /> : 'Auto Fix'}
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
