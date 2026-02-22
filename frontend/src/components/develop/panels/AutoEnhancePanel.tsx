import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  tokens,
  Badge,
  Button,
  Dropdown,
  Option,
  Slider,
  Spinner,
  Switch,
  Text,
} from '@fluentui/react-components';
import { PaintBrushSparkle24Regular } from '@fluentui/react-icons';
import { PanelSection } from '../../common/PanelSection';
import { useEditStore } from '../../../stores/editStore';
import { useLibraryStore } from '../../../stores/libraryStore';
import { useSettingsStore } from '../../../stores/settingsStore';
import { listEnhanceModels, runAutoEnhance, runOptimize, pollJob } from '../../../api/client';
import type { EnhanceModelDescriptor, GlobalAdjustments, OptimizeMasksSummary } from '../../../types';

/** Classify a model into one of the three groups */
function modelGroup(m: EnhanceModelDescriptor): 'algorithmic' | 'local' | 'remote' {
  if (m.builtin && !m.requiresApiKey && !m.publisher) return 'algorithmic';
  if (m.requiresApiKey) return 'remote';
  return 'local';
}

/** Collapsible group header */
function GroupHeader({
  label,
  open,
  onToggle,
  active,
}: {
  label: string;
  open: boolean;
  onToggle: () => void;
  active: boolean;
}) {
  return (
    <div
      onClick={onToggle}
      style={{
        cursor: 'pointer',
        userSelect: 'none',
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 0',
      }}
    >
      <Text size={200} style={{ color: tokens.colorNeutralForeground3, width: 10 }}>
        {open ? '\u25BE' : '\u25B8'}
      </Text>
      <Text
        size={200}
        weight="semibold"
        style={{ color: active ? tokens.colorBrandForeground1 : tokens.colorNeutralForeground2 }}
      >
        {label}
      </Text>
      {active && (
        <div style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: tokens.colorBrandForeground1,
          flexShrink: 0,
        }} />
      )}
    </div>
  );
}

export function AutoEnhancePanel() {
  const editState = useEditStore(s => s.editState);
  const updateGlobalPartial = useEditStore(s => s.updateGlobalPartial);
  const pushHistory = useEditStore(s => s.pushHistory);
  const activeAsset = useLibraryStore(s => s.assets.find(a => a.id === s.activeAssetId));
  const geminiApiKey = useSettingsStore(s => s.geminiApiKey);
  const anthropicApiKey = useSettingsStore(s => s.anthropicApiKey);
  const openaiApiKey = useSettingsStore(s => s.openaiApiKey);

  const [models, setModels] = useState<EnhanceModelDescriptor[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('builtin');
  const [strength, setStrength] = useState(100);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [adjustedParams, setAdjustedParams] = useState<Record<string, number> | null>(null);

  // Optimize-specific state
  const [progress, setProgress] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [denoise, setDenoise] = useState(true);
  const [enhance, setEnhance] = useState(true);
  const [masks, setMasks] = useState(true);
  const [maskResults, setMaskResults] = useState<OptimizeMasksSummary | null>(null);

  // Section collapse state
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    algorithmic: true,
    local: false,
    remote: false,
  });

  // Store original values and raw recommendations for live re-interpolation
  const originalRef = useRef<Partial<GlobalAdjustments> | null>(null);
  const recommendedRef = useRef<Record<string, number> | null>(null);

  const isOptimize = selectedModelId === 'optimize';

  // Split models into groups
  const { algorithmic, local, remote } = useMemo(() => {
    const groups = { algorithmic: [] as EnhanceModelDescriptor[], local: [] as EnhanceModelDescriptor[], remote: [] as EnhanceModelDescriptor[] };
    for (const m of models) {
      groups[modelGroup(m)].push(m);
    }
    return groups;
  }, [models]);

  // Which group is the selected model in?
  const selectedModel = models.find(m => m.id === selectedModelId);
  const activeGroup = selectedModel ? modelGroup(selectedModel) : 'algorithmic';

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

  const toggleSection = (section: string) => {
    setOpenSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const selectModel = (id: string) => {
    setSelectedModelId(id);
    setAdjustedParams(null);
    setMaskResults(null);
    recommendedRef.current = null;
  };

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
    setProgress(0);
    setMaskResults(null);

    // Capture originals before applying
    const orig: Partial<GlobalAdjustments> = { ...editState.global };
    originalRef.current = orig;

    try {
      if (isOptimize) {
        // Route to optimize API
        const { jobId } = await runOptimize(activeAsset.fileId, {
          strength,
          denoise,
          enhance,
          masks,
        });
        const job = await pollJob(jobId, setProgress);

        if (job.status === 'FAILED') {
          setError(job.error?.detail ?? 'Optimize failed');
          return;
        }

        const result = job.result as {
          appliedParams?: Record<string, number>;
          masks?: OptimizeMasksSummary;
        } | undefined;

        if (result?.masks) {
          setMaskResults(result.masks);
        }

        if (result?.appliedParams) {
          recommendedRef.current = result.appliedParams;
          applyInterpolated(result.appliedParams, orig, strength);
          pushHistory('Auto Fix');
        }
      } else {
        // Route to auto-enhance API
        const { jobId } = await runAutoEnhance(activeAsset.fileId, selectedModelId, {
          apiKey: geminiApiKey ?? undefined,
          anthropicApiKey: anthropicApiKey ?? undefined,
          openaiApiKey: openaiApiKey ?? undefined,
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

  const needsKey = selectedModel?.requiresApiKey && (
    selectedModel.id === 'claude'
      ? !anthropicApiKey
      : selectedModel.id === 'openai'
        ? !openaiApiKey
        : !geminiApiKey
  );

  return (
    <PanelSection title="Auto Fix" defaultOpen>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>

        {/* ── Algorithmic ── */}
        <GroupHeader
          label="Algorithmic"
          open={openSections.algorithmic}
          onToggle={() => toggleSection('algorithmic')}
          active={activeGroup === 'algorithmic'}
        />
        {openSections.algorithmic && algorithmic.length > 0 && (
          <div style={{ paddingLeft: 16, display: 'flex', flexDirection: 'column', gap: 2 }}>
            {algorithmic.map(m => (
              <div
                key={m.id}
                onClick={() => selectModel(m.id)}
                style={{
                  padding: '4px 8px',
                  borderRadius: 4,
                  cursor: 'pointer',
                  background: selectedModelId === m.id ? tokens.colorBrandBackground2 : 'transparent',
                }}
              >
                <Text size={200} weight={selectedModelId === m.id ? 'semibold' : 'regular'}>
                  {m.name}
                </Text>
                <Text size={200} style={{ display: 'block', color: tokens.colorNeutralForeground3 }}>
                  {m.description}
                </Text>
              </div>
            ))}
          </div>
        )}

        {/* ── Local Models ── */}
        {local.length > 0 && (
          <>
            <GroupHeader
              label="Local Models"
              open={openSections.local}
              onToggle={() => toggleSection('local')}
              active={activeGroup === 'local'}
            />
            {openSections.local && (
              <div style={{ paddingLeft: 16 }}>
                <Dropdown
                  value={activeGroup === 'local' && selectedModel ? selectedModel.name : 'Select...'}
                  selectedOptions={activeGroup === 'local' ? [selectedModelId] : []}
                  onOptionSelect={(_, data) => {
                    if (data.optionValue) selectModel(data.optionValue);
                  }}
                  style={{ minWidth: 0, width: '100%' }}
                >
                  {local.map(m => (
                    <Option key={m.id} value={m.id} text={m.name}>
                      <div>
                        <Text size={300} weight="semibold">{m.name}</Text>
                        {m.publisher && (
                          <Text size={200} style={{ display: 'block', color: tokens.colorNeutralForeground4 }}>
                            by {m.publisher}
                          </Text>
                        )}
                        <Text size={200} style={{ display: 'block', color: tokens.colorNeutralForeground3, marginTop: 2 }}>
                          {m.description}
                        </Text>
                      </div>
                    </Option>
                  ))}
                </Dropdown>
              </div>
            )}
          </>
        )}

        {/* ── Remote Models ── */}
        {remote.length > 0 && (
          <>
            <GroupHeader
              label="Remote Models"
              open={openSections.remote}
              onToggle={() => toggleSection('remote')}
              active={activeGroup === 'remote'}
            />
            {openSections.remote && (
              <div style={{ paddingLeft: 16 }}>
                <Dropdown
                  value={activeGroup === 'remote' && selectedModel ? selectedModel.name : 'Select...'}
                  selectedOptions={activeGroup === 'remote' ? [selectedModelId] : []}
                  onOptionSelect={(_, data) => {
                    if (data.optionValue) selectModel(data.optionValue);
                  }}
                  style={{ minWidth: 0, width: '100%' }}
                >
                  {remote.map(m => (
                    <Option key={m.id} value={m.id} text={m.name}>
                      <div>
                        <Text size={300} weight="semibold">{m.name}</Text>
                        {m.publisher && (
                          <Text size={200} style={{ display: 'block', color: tokens.colorNeutralForeground4 }}>
                            by {m.publisher}
                          </Text>
                        )}
                        <Text size={200} style={{ display: 'block', color: tokens.colorNeutralForeground3, marginTop: 2 }}>
                          {m.description}
                        </Text>
                      </div>
                    </Option>
                  ))}
                </Dropdown>
              </div>
            )}
          </>
        )}

        {/* ── API key warning ── */}
        {needsKey && (
          <Text
            size={200}
            style={{ color: tokens.colorPaletteYellowForeground1, marginTop: 4 }}
          >
            This provider requires {selectedModel?.id === 'claude' ? 'an Anthropic' : selectedModel?.id === 'openai' ? 'an OpenAI' : 'a Gemini'} API key. Set it in Settings.
          </Text>
        )}

        {/* ── Strength slider ── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
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

        {/* ── Advanced Options — only for optimize model ── */}
        {isOptimize && (
          <>
            <div
              style={{ cursor: 'pointer', userSelect: 'none' }}
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>
                {showAdvanced ? '\u25BE' : '\u25B8'} Advanced Options
              </Text>
            </div>

            {showAdvanced && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, paddingLeft: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text size={200}>AI Denoise</Text>
                  <Switch
                    checked={denoise}
                    onChange={(_, data) => setDenoise(data.checked)}
                  />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text size={200}>HDRNet Enhance</Text>
                  <Switch
                    checked={enhance}
                    onChange={(_, data) => setEnhance(data.checked)}
                  />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text size={200}>Smart Masks</Text>
                  <Switch
                    checked={masks}
                    onChange={(_, data) => setMasks(data.checked)}
                  />
                </div>
              </div>
            )}
          </>
        )}

        {/* ── Auto Fix button ── */}
        <Button
          appearance="primary"
          icon={<PaintBrushSparkle24Regular />}
          onClick={handleRun}
          disabled={isRunning || !activeAsset?.fileId || !!needsKey}
          style={{ width: '100%', marginTop: 4 }}
        >
          {isRunning ? (
            isOptimize ? (
              <>
                <Spinner size="tiny" style={{ marginRight: 6 }} />
                {Math.round(progress * 100)}%
              </>
            ) : (
              <Spinner size="tiny" />
            )
          ) : (
            'Auto Fix'
          )}
        </Button>

        {/* ── Error display ── */}
        {error && (
          <Text size={200} style={{ color: tokens.colorPaletteRedForeground1 }}>
            {error}
          </Text>
        )}

        {/* ── Mask detection badges ── */}
        {maskResults && (
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {maskResults.subject && (
              <Badge appearance="filled" color="brand" size="small">Subject</Badge>
            )}
            {maskResults.sky && (
              <Badge appearance="filled" color="informative" size="small">Sky</Badge>
            )}
            {maskResults.skin && (
              <Badge appearance="filled" color="subtle" size="small">Skin</Badge>
            )}
          </div>
        )}

        {/* ── Adjusted params display ── */}
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
