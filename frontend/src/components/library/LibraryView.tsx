import {
  tokens,
  Button,
  Select,
  ProgressBar,
  Text,
  Badge,
  Tooltip,
} from '@fluentui/react-components';
import {
  ArrowImportRegular,
  StarRegular,
  StarFilled,
  CheckmarkCircleRegular,
  DismissCircleRegular,
  FilterRegular,
} from '@fluentui/react-icons';
import { useLibraryStore } from '../../stores/libraryStore';
import { useEditStore } from '../../stores/editStore';
import { ThumbnailGrid } from './ThumbnailGrid';
import { useCallback } from 'react';

export function LibraryView() {
  const assets = useLibraryStore(s => s.assets);
  const filteredAssets = useLibraryStore(s => s.getFilteredAssets());
  const importProgress = useLibraryStore(s => s.importProgress);
  const filterFlag = useLibraryStore(s => s.filterFlag);
  const filterMinRating = useLibraryStore(s => s.filterMinRating);
  const setFilterFlag = useLibraryStore(s => s.setFilterFlag);
  const setFilterMinRating = useLibraryStore(s => s.setFilterMinRating);
  const sortBy = useLibraryStore(s => s.sortBy);
  const setSortBy = useLibraryStore(s => s.setSortBy);
  const setViewMode = useLibraryStore(s => s.setViewMode);
  const setActiveAsset = useLibraryStore(s => s.setActiveAsset);
  const setCurrentAsset = useEditStore(s => s.setCurrentAsset);

  const handleOpenInDevelop = useCallback((assetId: string) => {
    setActiveAsset(assetId);
    setCurrentAsset(assetId);
    setViewMode('develop');
  }, [setActiveAsset, setCurrentAsset, setViewMode]);

  if (assets.length === 0 && !importProgress) {
    return (
      <div style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 16,
        color: tokens.colorNeutralForeground3,
      }}>
        <ArrowImportRegular style={{ fontSize: 64 }} />
        <Text size={500} weight="semibold">No photos imported</Text>
        <Text size={300}>Click Import in the toolbar to add photos</Text>
      </div>
    );
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Filter bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '4px 12px',
        borderBottom: `1px solid ${tokens.colorNeutralStroke1}`,
        flexShrink: 0,
      }}>
        <FilterRegular />
        <Select
          size="small"
          value={filterFlag}
          onChange={(_, data) => setFilterFlag(data.value as any)}
          style={{ minWidth: 120 }}
        >
          <option value="all">All</option>
          <option value="pick">Picks</option>
          <option value="reject">Rejects</option>
          <option value="unflagged">Unflagged</option>
        </Select>

        <Text size={200}>Min Rating:</Text>
        <div style={{ display: 'flex', gap: 2 }}>
          {[0, 1, 2, 3, 4, 5].map(r => (
            <Button
              key={r}
              size="small"
              appearance={filterMinRating === r ? 'primary' : 'subtle'}
              onClick={() => setFilterMinRating(r)}
              style={{ minWidth: 28, padding: '0 4px' }}
            >
              {r === 0 ? 'All' : r.toString()}
            </Button>
          ))}
        </div>

        <div style={{ flex: 1 }} />

        <Select
          size="small"
          value={sortBy}
          onChange={(_, data) => setSortBy(data.value as any)}
        >
          <option value="date">Date</option>
          <option value="name">Name</option>
          <option value="rating">Rating</option>
        </Select>

        <Badge appearance="filled" color="informative">
          {filteredAssets.length} / {assets.length}
        </Badge>
      </div>

      {/* Import progress */}
      {importProgress && (
        <div style={{ padding: '4px 12px', flexShrink: 0 }}>
          <ProgressBar
            value={importProgress.current / importProgress.total}
            max={1}
          />
          <Text size={200}>
            Importing {importProgress.current} / {importProgress.total}
          </Text>
        </div>
      )}

      {/* Thumbnail grid */}
      <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
        <ThumbnailGrid
          assets={filteredAssets}
          onDoubleClick={handleOpenInDevelop}
        />
      </div>
    </div>
  );
}
