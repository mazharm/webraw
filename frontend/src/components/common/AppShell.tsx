import {
  Toolbar,
  ToolbarButton,
  ToolbarDivider,
  MessageBar,
  MessageBarBody,
  tokens,
} from '@fluentui/react-components';
import {
  ArrowImportRegular,
  ArrowExportRegular,
  ArrowUndoRegular,
  ArrowRedoRegular,
  SettingsRegular,
} from '@fluentui/react-icons';
import { useLibraryStore } from '../../stores/libraryStore';
import { useEditStore } from '../../stores/editStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { LibraryView } from '../library/LibraryView';
import { DevelopView } from '../develop/DevelopView';
import { ExportDialog } from '../export/ExportDialog';
import { SettingsDialog } from './SettingsDialog';
import { useCallback, useRef, useState } from 'react';
import { uploadFile, generateThumbnail } from '../../api/client';
import type { Asset } from '../../types';

/** Generate a thumbnail in the background, updating the asset when done */
function backgroundThumbnail(
  fileId: string,
  assetId: string,
  updateAsset: (id: string, updates: Partial<Asset>) => void,
) {
  generateThumbnail(fileId, 300)
    .then(blob => {
      updateAsset(assetId, { thumbnailUrl: URL.createObjectURL(blob) });
    })
    .catch(() => {/* non-critical — grid will show filename fallback */});
}

export function AppShell() {
  const viewMode = useLibraryStore(s => s.viewMode);
  const setViewMode = useLibraryStore(s => s.setViewMode);
  const addAssets = useLibraryStore(s => s.addAssets);
  const updateAsset = useLibraryStore(s => s.updateAsset);
  const setImportProgress = useLibraryStore(s => s.setImportProgress);
  const backendHealthy = useSettingsStore(s => s.backendHealthy);
  const { undo, redo, canUndo, canRedo } = useEditStore();
  const [showExport, setShowExport] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImport = useCallback(async () => {
    fileInputRef.current?.click();
  }, []);

  const handleFiles = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileArray = Array.from(files);
    setImportProgress({ current: 0, total: fileArray.length });
    setImportError(null);

    const errors: string[] = [];
    let completed = 0;
    const CONCURRENCY = 4;

    const importOne = async (file: File) => {
      try {
        const result = await uploadFile(file);

        const isRaw = /\.(dng|cr2|cr3|nef|arw|raf)$/i.test(file.name);
        const assetId = crypto.randomUUID();
        const asset: Asset = {
          id: assetId,
          filename: file.name,
          mime: file.type || 'application/octet-stream',
          sourceType: isRaw ? 'RAW' : 'RASTER',
          originalUri: file.name,
          createdAt: new Date().toISOString(),
          exif: result.exif ?? undefined,
          fileId: result.fileId,
        };
        // Add asset to library immediately — thumbnail fills in later
        addAssets([asset]);
        // Fire-and-forget thumbnail generation
        backgroundThumbnail(result.fileId, assetId, updateAsset);
      } catch (err: unknown) {
        const detail = err && typeof err === 'object' && 'detail' in err
          ? (err as { detail: string }).detail
          : String(err);
        console.error(`Failed to import ${file.name}:`, err);
        errors.push(`${file.name}: ${detail}`);
      }
      completed++;
      setImportProgress({ current: completed, total: fileArray.length });
    };

    // Process files with bounded concurrency
    let cursor = 0;
    const next = async (): Promise<void> => {
      while (cursor < fileArray.length) {
        const idx = cursor++;
        await importOne(fileArray[idx]);
      }
    };
    const workers = Array.from({ length: Math.min(CONCURRENCY, fileArray.length) }, () => next());
    await Promise.all(workers);

    if (errors.length > 0) {
      setImportError(`Import failed: ${errors.join('; ')}`);
    }
    setImportProgress(null);
    e.target.value = '';
  }, [addAssets, setImportProgress]);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: tokens.colorNeutralBackground1 }}>
      {/* Command Bar */}
      <Toolbar
        size="small"
        style={{
          height: 'var(--command-bar-height)',
          borderBottom: `1px solid ${tokens.colorNeutralStroke1}`,
          flexShrink: 0,
          padding: '0 8px',
        }}
      >
        <ToolbarButton
          icon={<ArrowImportRegular />}
          onClick={handleImport}
          aria-label="Import"
        >
          Import
        </ToolbarButton>
        <ToolbarButton
          icon={<ArrowExportRegular />}
          onClick={() => setShowExport(true)}
          aria-label="Export"
        >
          Export
        </ToolbarButton>
        <ToolbarDivider />
        <ToolbarButton
          icon={<ArrowUndoRegular />}
          onClick={undo}
          disabled={!canUndo()}
          aria-label="Undo"
        />
        <ToolbarButton
          icon={<ArrowRedoRegular />}
          onClick={redo}
          disabled={!canRedo()}
          aria-label="Redo"
        />
        <ToolbarDivider />

        <div style={{ display: 'flex', gap: 2, marginLeft: 8 }}>
          <ToolbarButton
            appearance={viewMode === 'library' ? 'primary' : 'subtle'}
            onClick={() => setViewMode('library')}
          >
            Library
          </ToolbarButton>
          <ToolbarButton
            appearance={viewMode === 'develop' ? 'primary' : 'subtle'}
            onClick={() => setViewMode('develop')}
          >
            Develop
          </ToolbarButton>
        </div>

        <div style={{ flex: 1 }} />

        <ToolbarButton
          icon={<SettingsRegular />}
          onClick={() => setShowSettings(true)}
          aria-label="Settings"
        />
      </Toolbar>

      {/* Backend offline banner */}
      {!backendHealthy && (
        <MessageBar intent="warning" style={{ flexShrink: 0 }}>
          <MessageBarBody>
            Backend service unavailable — some editing features are disabled.
          </MessageBarBody>
        </MessageBar>
      )}

      {/* Import error banner */}
      {importError && (
        <MessageBar intent="error" style={{ flexShrink: 0 }}>
          <MessageBarBody>
            {importError}
          </MessageBarBody>
        </MessageBar>
      )}

      {/* Main content area */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {viewMode === 'library' && <LibraryView />}
        {viewMode === 'develop' && <DevelopView />}
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".dng,.cr2,.cr3,.nef,.arw,.raf,.jpg,.jpeg,.png,.tiff,.tif"
        style={{ display: 'none' }}
        onChange={handleFiles}
      />

      {showExport && <ExportDialog onDismiss={() => setShowExport(false)} />}
      {showSettings && <SettingsDialog onDismiss={() => setShowSettings(false)} />}
    </div>
  );
}
