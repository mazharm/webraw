import { PanelSection } from '../common/PanelSection';
import {
  tokens,
  Button,
  Input,
  Select,
  Spinner,
  Text,
  MessageBar,
  MessageBarBody,
  Slider,
} from '@fluentui/react-components';
import { useEditStore } from '../../stores/editStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { useLibraryStore } from '../../stores/libraryStore';
import { createAiEdit, pollJob, getFileUrl } from '../../api/client';
import { useState, useCallback } from 'react';
import type { AiLayer } from '../../types';

type AiMode = 'edit' | 'remove' | 'replace_bg' | 'relight' | 'expand';
type AiProvider = 'gemini' | 'openai' | 'google-imagen';

export function AiEditPanel() {
  const editState = useEditStore(s => s.editState);
  const addAiLayer = useEditStore(s => s.addAiLayer);
  const removeAiLayer = useEditStore(s => s.removeAiLayer);
  const updateAiLayer = useEditStore(s => s.updateAiLayer);
  const pushHistory = useEditStore(s => s.pushHistory);
  const geminiApiKey = useSettingsStore(s => s.geminiApiKey);
  const openaiApiKey = useSettingsStore(s => s.openaiApiKey);
  const activeAsset = useLibraryStore(s => s.assets.find(a => a.id === s.activeAssetId));

  const [prompt, setPrompt] = useState('');
  const [mode, setMode] = useState<AiMode>('edit');
  const [provider, setProvider] = useState<AiProvider>('gemini');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  // Resolve the API key based on provider
  const activeApiKey = provider === 'openai' ? openaiApiKey : geminiApiKey;

  const handleSubmit = useCallback(async () => {
    if (!editState || !activeAsset?.fileId || !activeApiKey || !prompt.trim()) return;

    setIsSubmitting(true);
    setError(null);
    setProgress(0);

    try {
      const { jobId } = await createAiEdit(
        activeAsset.fileId,
        editState,
        prompt,
        mode,
        activeApiKey,
        { provider },
      );

      const result = await pollJob(jobId, setProgress);

      if (result.status === 'FAILED') {
        setError(result.error?.detail ?? 'AI edit failed');
        return;
      }

      const meta = (result.result as any)?.meta;
      const resultFileId = (result.result as any)?.resultFileId;

      if (resultFileId) {
        const layer: AiLayer = {
          id: crypto.randomUUID(),
          assetId: resultFileId,
          opacity: 1.0,
          blendMode: 'NORMAL',
          meta: {
            provider: meta?.provider ?? provider,
            model: meta?.model ?? 'unknown',
            prompt,
            createdAt: new Date().toISOString(),
          },
          enabled: true,
        };
        addAiLayer(layer);
        pushHistory(`AI: ${mode}`);
        setPrompt('');
      }
    } catch (err: any) {
      setError(err.detail ?? err.message ?? 'Request failed');
    } finally {
      setIsSubmitting(false);
    }
  }, [editState, activeAsset, activeApiKey, prompt, mode, provider, addAiLayer, pushHistory]);

  if (!editState) return null;

  return (
    <PanelSection title="AI Edit">
      {!activeApiKey && (
        <MessageBar intent="warning" style={{ marginBottom: 8 }}>
          <MessageBarBody>
            Set your {provider === 'openai' ? 'OpenAI' : 'Gemini'} API key in Settings to enable AI features.
          </MessageBarBody>
        </MessageBar>
      )}

      <Text size={200} weight="semibold" style={{ display: 'block', marginBottom: 4, color: tokens.colorNeutralForeground3 }}>
        Creative / Generative
      </Text>
      <Select
        size="small"
        value={provider}
        onChange={(_, data) => setProvider(data.value as AiProvider)}
        style={{ marginBottom: 8, width: '100%' }}
      >
        <option value="gemini">Gemini</option>
        <option value="openai">OpenAI gpt-image-1</option>
        <option value="google-imagen">Google Imagen 3.0</option>
      </Select>

      <Select
        size="small"
        value={mode}
        onChange={(_, data) => setMode(data.value as AiMode)}
        style={{ marginBottom: 8, width: '100%' }}
      >
        <option value="edit">Edit (natural language)</option>
        <option value="remove">Remove object</option>
        <option value="replace_bg">Replace background</option>
        <option value="relight">Relight</option>
        <option value="expand">Expand (outpaint)</option>
      </Select>

      <Input
        size="small"
        placeholder={
          mode === 'remove' ? 'Describe what to remove...'
            : mode === 'replace_bg' ? 'Describe new background...'
            : mode === 'relight' ? 'Describe lighting change...'
            : 'Describe your edit...'
        }
        value={prompt}
        onChange={(_, data) => setPrompt(data.value)}
        disabled={isSubmitting || !activeApiKey}
        style={{ width: '100%', marginBottom: 8 }}
      />

      <Button
        appearance="primary"
        size="small"
        onClick={handleSubmit}
        disabled={isSubmitting || !activeApiKey || !prompt.trim()}
        style={{ width: '100%', marginBottom: 8 }}
      >
        {isSubmitting ? <Spinner size="tiny" /> : 'Run AI Edit'}
      </Button>

      {error && (
        <MessageBar intent="error" style={{ marginBottom: 8 }}>
          <MessageBarBody>{error}</MessageBarBody>
        </MessageBar>
      )}

      {/* AI Layers list */}
      {editState.aiLayers.length > 0 && (
        <>
          <Text size={200} weight="semibold" style={{ display: 'block', marginTop: 8, marginBottom: 4 }}>
            AI Layers
          </Text>
          {editState.aiLayers.map(layer => (
            <div
              key={layer.id}
              style={{
                padding: '6px 8px',
                background: tokens.colorNeutralBackground3,
                borderRadius: 4,
                marginBottom: 4,
                fontSize: 11,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text size={200} truncate>
                  {layer.meta.prompt?.substring(0, 30) ?? 'AI Layer'}
                </Text>
                <Button
                  size="small"
                  appearance="subtle"
                  onClick={() => {
                    removeAiLayer(layer.id);
                    pushHistory('Remove AI Layer');
                  }}
                >
                  Remove
                </Button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                <Text size={100}>Opacity</Text>
                <Slider
                  size="small"
                  min={0}
                  max={100}
                  value={layer.opacity * 100}
                  onChange={(_, data) => updateAiLayer(layer.id, { opacity: data.value / 100 })}
                  style={{ flex: 1 }}
                />
                <Select
                  size="small"
                  value={layer.blendMode}
                  onChange={(_, data) => updateAiLayer(layer.id, { blendMode: data.value as any })}
                  style={{ width: 90 }}
                >
                  <option value="NORMAL">Normal</option>
                  <option value="MULTIPLY">Multiply</option>
                  <option value="SCREEN">Screen</option>
                  <option value="OVERLAY">Overlay</option>
                </Select>
              </div>
            </div>
          ))}
        </>
      )}
    </PanelSection>
  );
}
