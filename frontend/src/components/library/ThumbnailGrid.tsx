import {
  tokens,
  Text,
  Button,
  Tooltip,
} from '@fluentui/react-components';
import {
  StarFilled,
  StarRegular,
  CheckmarkCircleRegular,
  DismissCircleRegular,
  ImageRegular,
} from '@fluentui/react-icons';
import { useLibraryStore } from '../../stores/libraryStore';
import type { Asset } from '../../types';
import { useCallback } from 'react';

interface Props {
  assets: Asset[];
  onDoubleClick: (assetId: string) => void;
}

export function ThumbnailGrid({ assets, onDoubleClick }: Props) {
  const selectedAssetIds = useLibraryStore(s => s.selectedAssetIds);
  const selectAsset = useLibraryStore(s => s.selectAsset);
  const setFlag = useLibraryStore(s => s.setFlag);
  const setRating = useLibraryStore(s => s.setRating);

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(var(--thumbnail-size), 1fr))',
      gap: 8,
    }}>
      {assets.map(asset => (
        <ThumbnailCard
          key={asset.id}
          asset={asset}
          selected={selectedAssetIds.has(asset.id)}
          onSelect={(multi) => selectAsset(asset.id, multi)}
          onDoubleClick={() => onDoubleClick(asset.id)}
          onFlag={(flag) => setFlag(asset.id, flag)}
          onRate={(rating) => setRating(asset.id, rating)}
        />
      ))}
    </div>
  );
}

function ThumbnailCard({
  asset,
  selected,
  onSelect,
  onDoubleClick,
  onFlag,
  onRate,
}: {
  asset: Asset;
  selected: boolean;
  onSelect: (multi: boolean) => void;
  onDoubleClick: () => void;
  onFlag: (flag: 'pick' | 'reject' | null) => void;
  onRate: (rating: number) => void;
}) {
  const handleClick = useCallback((e: React.MouseEvent) => {
    onSelect(e.ctrlKey || e.metaKey);
  }, [onSelect]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') onDoubleClick();
    if (e.key >= '1' && e.key <= '5') onRate(parseInt(e.key));
    if (e.key === 'p') onFlag('pick');
    if (e.key === 'x') onFlag('reject');
    if (e.key === 'u') onFlag(null);
  }, [onDoubleClick, onRate, onFlag]);

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleClick}
      onDoubleClick={onDoubleClick}
      onKeyDown={handleKeyDown}
      aria-label={`${asset.filename}, rating ${asset.rating ?? 0}`}
      style={{
        border: `2px solid ${selected ? tokens.colorBrandStroke1 : 'transparent'}`,
        borderRadius: tokens.borderRadiusMedium,
        overflow: 'hidden',
        cursor: 'pointer',
        background: tokens.colorNeutralBackground3,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Thumbnail image */}
      <div style={{
        width: '100%',
        aspectRatio: '1',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        background: tokens.colorNeutralBackground4,
      }}>
        {asset.thumbnailUrl ? (
          <img
            src={asset.thumbnailUrl}
            alt={asset.filename}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            loading="lazy"
          />
        ) : (
          <ImageRegular style={{ fontSize: 48, color: tokens.colorNeutralForeground4 }} />
        )}
      </div>

      {/* Info bar */}
      <div style={{
        padding: '4px 6px',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}>
        <Text
          size={200}
          truncate
          wrap={false}
          style={{ display: 'block' }}
        >
          {asset.filename}
        </Text>
        <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {/* Rating stars */}
          {[1, 2, 3, 4, 5].map(star => (
            <span
              key={star}
              onClick={(e) => { e.stopPropagation(); onRate(star === asset.rating ? 0 : star); }}
              style={{ cursor: 'pointer', fontSize: 12 }}
            >
              {(asset.rating ?? 0) >= star
                ? <StarFilled style={{ color: tokens.colorPaletteYellowForeground1, fontSize: 12 }} />
                : <StarRegular style={{ fontSize: 12, color: tokens.colorNeutralForeground4 }} />
              }
            </span>
          ))}

          <div style={{ flex: 1 }} />

          {/* Flag indicators */}
          {asset.flag === 'pick' && (
            <CheckmarkCircleRegular style={{ color: tokens.colorPaletteGreenForeground1, fontSize: 14 }} />
          )}
          {asset.flag === 'reject' && (
            <DismissCircleRegular style={{ color: tokens.colorPaletteRedForeground1, fontSize: 14 }} />
          )}
        </div>
      </div>
    </div>
  );
}
