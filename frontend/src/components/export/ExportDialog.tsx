import {
  Dialog,
  DialogSurface,
  DialogBody,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Select,
  Field,
  ProgressBar,
  Text,
  Slider,
  MessageBar,
  MessageBarBody,
} from '@fluentui/react-components';
import { useEditStore } from '../../stores/editStore';
import { useLibraryStore } from '../../stores/libraryStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { renderExport } from '../../api/client';
import { useState, useCallback } from 'react';

interface Props {
  onDismiss: () => void;
}

export function ExportDialog({ onDismiss }: Props) {
  const editState = useEditStore(s => s.editState);
  const activeAsset = useLibraryStore(s => s.assets.find(a => a.id === s.activeAssetId));
  const colorSpace = useSettingsStore(s => s.colorSpace);

  const [format, setFormat] = useState<'JPG' | 'PNG' | 'TIFF'>('JPG');
  const [quality, setQuality] = useState(95);
  const [bitDepth, setBitDepth] = useState<8 | 16>(8);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exportDone, setExportDone] = useState(false);

  const handleExport = useCallback(async () => {
    if (!editState || !activeAsset?.fileId) return;

    setExporting(true);
    setError(null);
    setExportDone(false);

    try {
      const blob = await renderExport(
        activeAsset.fileId,
        editState,
        format,
        { bitDepth, quality, colorSpace },
      );

      // Trigger browser download
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const baseName = activeAsset.filename.replace(/\.[^.]+$/, '');
      a.download = `${baseName}.${format.toLowerCase() === 'jpg' ? 'jpg' : format.toLowerCase()}`;
      a.click();
      URL.revokeObjectURL(url);

      setExportDone(true);
    } catch (err: any) {
      setError(err.message ?? 'Export failed');
    } finally {
      setExporting(false);
    }
  }, [editState, activeAsset, format, quality, bitDepth, colorSpace]);

  return (
    <Dialog open onOpenChange={(_, data) => !data.open && onDismiss()}>
      <DialogSurface>
        <DialogBody>
          <DialogTitle>Export Image</DialogTitle>
          <DialogContent>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {!activeAsset && (
                <MessageBar intent="warning">
                  <MessageBarBody>Select an image to export.</MessageBarBody>
                </MessageBar>
              )}

              <Field label="Format">
                <Select
                  value={format}
                  onChange={(_, data) => setFormat(data.value as any)}
                >
                  <option value="JPG">JPEG</option>
                  <option value="PNG">PNG</option>
                  <option value="TIFF">TIFF</option>
                </Select>
              </Field>

              {format === 'JPG' && (
                <Field label={`Quality: ${quality}`}>
                  <Slider
                    min={1} max={100}
                    value={quality}
                    onChange={(_, data) => setQuality(data.value)}
                  />
                </Field>
              )}

              {(format === 'PNG' || format === 'TIFF') && (
                <Field label="Bit Depth">
                  <Select
                    value={bitDepth.toString()}
                    onChange={(_, data) => setBitDepth(parseInt(data.value, 10) as 8 | 16)}
                  >
                    <option value="8">8-bit</option>
                    <option value="16">16-bit</option>
                  </Select>
                </Field>
              )}

              <Field label="Color Space">
                <Select value={colorSpace} disabled>
                  <option value="sRGB">sRGB</option>
                  <option value="DisplayP3">Display P3</option>
                </Select>
              </Field>

              {exporting && (
                <div>
                  <ProgressBar />
                  <Text size={200}>Exporting...</Text>
                </div>
              )}

              {error && (
                <MessageBar intent="error">
                  <MessageBarBody>{error}</MessageBarBody>
                </MessageBar>
              )}

              {exportDone && (
                <MessageBar intent="success">
                  <MessageBarBody>Export complete! File downloaded.</MessageBarBody>
                </MessageBar>
              )}
            </div>
          </DialogContent>
          <DialogActions>
            <Button appearance="secondary" onClick={onDismiss}>Close</Button>
            <Button
              appearance="primary"
              onClick={handleExport}
              disabled={exporting || !activeAsset}
            >
              Export
            </Button>
          </DialogActions>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  );
}
