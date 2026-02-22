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
  Switch,
  MessageBar,
  MessageBarBody,
} from '@fluentui/react-components';
import { useEditStore } from '../../stores/editStore';
import { useLibraryStore } from '../../stores/libraryStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { renderExport, pollJob, getFileUrl } from '../../api/client';
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
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);

  const handleExport = useCallback(async () => {
    if (!editState || !activeAsset?.fileId) return;

    setExporting(true);
    setError(null);
    setProgress(0);

    try {
      const { jobId } = await renderExport(
        activeAsset.fileId,
        editState,
        format,
        { bitDepth, quality, colorSpace },
      );

      const result = await pollJob(jobId, setProgress);

      if (result.status === 'FAILED') {
        setError(result.error?.detail ?? 'Export failed');
        return;
      }

      const resultData = result.result as any;
      if (resultData?.resultFileId) {
        const url = await getFileUrl(resultData.resultFileId);
        setDownloadUrl(url);
      } else if (resultData?.downloadUrl) {
        setDownloadUrl(resultData.downloadUrl);
      }
    } catch (err: any) {
      setError(err.detail ?? err.message ?? 'Export failed');
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
                    onChange={(_, data) => setBitDepth(parseInt(data.value) as 8 | 16)}
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
                  <ProgressBar value={progress} max={1} />
                  <Text size={200}>Exporting...</Text>
                </div>
              )}

              {error && (
                <MessageBar intent="error">
                  <MessageBarBody>{error}</MessageBarBody>
                </MessageBar>
              )}

              {downloadUrl && (
                <MessageBar intent="success">
                  <MessageBarBody>
                    Export complete!{' '}
                    <a href={downloadUrl} download style={{ color: 'inherit' }}>
                      Download file
                    </a>
                  </MessageBarBody>
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
