import { tokens, Text, Spinner, MessageBar, MessageBarBody, Button } from '@fluentui/react-components';
import { useEditStore } from '../../stores/editStore';
import { useLibraryStore } from '../../stores/libraryStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { ImageCanvas } from './ImageCanvas';
import { BasicPanel } from './panels/BasicPanel';
import { ToneCurvePanel } from './panels/ToneCurvePanel';
import { ColorPanel } from './panels/ColorPanel';
import { DetailPanel } from './panels/DetailPanel';
import { OpticsPanel } from './panels/OpticsPanel';
import { GeometryPanel } from './panels/GeometryPanel';
import { EffectsPanel } from './panels/EffectsPanel';
import { FilmSimPanel } from '../filmsim/FilmSimPanel';
import { AiEditPanel } from '../ai/AiEditPanel';
import { HistoryPanel } from './panels/HistoryPanel';
import { Histogram } from './Histogram';
import { usePreviewRenderer } from '../../hooks/usePreviewRenderer';

export function DevelopView() {
  const editState = useEditStore(s => s.editState);
  const activeAssetId = useLibraryStore(s => s.activeAssetId);
  const activeAsset = useLibraryStore(s => s.assets.find(a => a.id === activeAssetId));
  const backendHealthy = useSettingsStore(s => s.backendHealthy);

  const { previewUrl, histogram, isLoading, error, retry } = usePreviewRenderer(
    activeAsset?.fileId ?? null,
    editState
  );

  if (!activeAsset || !editState) {
    return (
      <div style={{
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: tokens.colorNeutralForeground3,
      }}>
        <Text size={400}>Select an image from the Library to edit</Text>
      </div>
    );
  }

  return (
    <div style={{ height: '100%', display: 'flex' }}>
      {/* Left panel - could hold filmstrip or local adjustments list */}
      <div style={{
        width: 60,
        borderRight: `1px solid ${tokens.colorNeutralStroke1}`,
        background: tokens.colorNeutralBackground2,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '8px 0',
        flexShrink: 0,
      }}>
        <Text size={100} style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)' }}>
          {activeAsset.filename}
        </Text>
      </div>

      {/* Center - Canvas */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {error && (
          <MessageBar intent="error" style={{ flexShrink: 0 }}>
            <MessageBarBody>
              {error}
              <Button size="small" appearance="transparent" onClick={retry} style={{ marginLeft: 8 }}>
                Retry
              </Button>
            </MessageBarBody>
          </MessageBar>
        )}
        <div style={{ flex: 1, position: 'relative' }}>
          <ImageCanvas previewUrl={previewUrl} isLoading={isLoading} />
          {isLoading && (
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              background: 'rgba(0,0,0,0.5)',
              borderRadius: 8,
              padding: 16,
            }}>
              <Spinner size="large" label="Rendering..." />
            </div>
          )}
        </div>
      </div>

      {/* Right panel - Adjustments */}
      <div style={{
        width: 'var(--panel-width)',
        borderLeft: `1px solid ${tokens.colorNeutralStroke1}`,
        background: tokens.colorNeutralBackground2,
        overflow: 'auto',
        flexShrink: 0,
      }}>
        <Histogram data={histogram} />
        <BasicPanel />
        <ToneCurvePanel />
        <ColorPanel />
        <DetailPanel />
        <OpticsPanel />
        <GeometryPanel />
        <EffectsPanel />
        <FilmSimPanel />
        <AiEditPanel />
        <HistoryPanel />
      </div>
    </div>
  );
}
