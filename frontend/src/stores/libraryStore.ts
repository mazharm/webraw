import { create } from 'zustand';
import type { Asset, ViewMode } from '../types';

type SortBy = 'name' | 'date' | 'rating';
type FilterFlag = 'all' | 'pick' | 'reject' | 'unflagged';

interface LibraryStore {
  assets: Asset[];
  selectedAssetIds: Set<string>;
  activeAssetId: string | null;
  viewMode: ViewMode;
  sortBy: SortBy;
  filterFlag: FilterFlag;
  filterMinRating: number;
  importProgress: { current: number; total: number } | null;

  addAssets: (assets: Asset[]) => void;
  removeAsset: (id: string) => void;
  selectAsset: (id: string, multi?: boolean) => void;
  setActiveAsset: (id: string | null) => void;
  setViewMode: (mode: ViewMode) => void;
  setSortBy: (sort: SortBy) => void;
  setFilterFlag: (flag: FilterFlag) => void;
  setFilterMinRating: (rating: number) => void;
  setImportProgress: (progress: { current: number; total: number } | null) => void;

  updateAsset: (id: string, updates: Partial<Asset>) => void;
  setFlag: (id: string, flag: 'pick' | 'reject' | null) => void;
  setRating: (id: string, rating: number) => void;
  setColorLabel: (id: string, label: string) => void;

  getFilteredAssets: () => Asset[];
}

export const useLibraryStore = create<LibraryStore>((set, get) => ({
  assets: [],
  selectedAssetIds: new Set(),
  activeAssetId: null,
  viewMode: 'library',
  sortBy: 'date',
  filterFlag: 'all',
  filterMinRating: 0,
  importProgress: null,

  addAssets: (assets) => set(state => ({
    assets: [...state.assets, ...assets],
  })),

  removeAsset: (id) => set(state => ({
    assets: state.assets.filter(a => a.id !== id),
    selectedAssetIds: new Set([...state.selectedAssetIds].filter(sid => sid !== id)),
    activeAssetId: state.activeAssetId === id ? null : state.activeAssetId,
  })),

  selectAsset: (id, multi = false) => set(state => {
    if (multi) {
      const newSet = new Set(state.selectedAssetIds);
      if (newSet.has(id)) newSet.delete(id);
      else newSet.add(id);
      return { selectedAssetIds: newSet };
    }
    return { selectedAssetIds: new Set([id]), activeAssetId: id };
  }),

  setActiveAsset: (id) => set({ activeAssetId: id }),
  setViewMode: (mode) => set({ viewMode: mode }),
  setSortBy: (sort) => set({ sortBy: sort }),
  setFilterFlag: (flag) => set({ filterFlag: flag }),
  setFilterMinRating: (rating) => set({ filterMinRating: rating }),
  setImportProgress: (progress) => set({ importProgress: progress }),

  updateAsset: (id, updates) => set(state => ({
    assets: state.assets.map(a => a.id === id ? { ...a, ...updates } : a),
  })),

  setFlag: (id, flag) => set(state => ({
    assets: state.assets.map(a => a.id === id ? { ...a, flag } : a),
  })),

  setRating: (id, rating) => set(state => ({
    assets: state.assets.map(a => a.id === id ? { ...a, rating } : a),
  })),

  setColorLabel: (id, label) => set(state => ({
    assets: state.assets.map(a => a.id === id ? { ...a, colorLabel: label } : a),
  })),

  getFilteredAssets: () => {
    const { assets, filterFlag, filterMinRating, sortBy } = get();
    let filtered = assets;

    if (filterFlag !== 'all') {
      if (filterFlag === 'unflagged') {
        filtered = filtered.filter(a => !a.flag);
      } else {
        filtered = filtered.filter(a => a.flag === filterFlag);
      }
    }

    if (filterMinRating > 0) {
      filtered = filtered.filter(a => (a.rating ?? 0) >= filterMinRating);
    }

    filtered = [...filtered].sort((a, b) => {
      switch (sortBy) {
        case 'name': return a.filename.localeCompare(b.filename);
        case 'rating': return (b.rating ?? 0) - (a.rating ?? 0);
        case 'date':
        default: return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      }
    });

    return filtered;
  },
}));
