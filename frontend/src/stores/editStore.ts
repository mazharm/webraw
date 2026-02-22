import { create } from 'zustand';
import type {
  EditState,
  GlobalAdjustments,
  FilmSimState,
  LocalAdjustment,
  AiLayer,
  HslChannel,
  HslValues,
  ToneCurve,
  Sharpening,
  Denoise,
  Optics,
  Effects,
  Crop,
  Transform,
  ColorGrading,
  History,
  HistoryNode,
  Snapshot,
} from '../types';

const MAX_HISTORY_NODES = 200;

function defaultHsl(): Record<HslChannel, HslValues> {
  const channels: HslChannel[] = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta'];
  const result: Record<string, HslValues> = {};
  for (const ch of channels) {
    result[ch] = { h: 0, s: 0, l: 0 };
  }
  return result as Record<HslChannel, HslValues>;
}

export function defaultGlobal(): GlobalAdjustments {
  return {
    exposure: 0,
    contrast: 0,
    highlights: 0,
    shadows: 0,
    whites: 0,
    blacks: 0,
    temperature: 5500,
    tint: 0,
    vibrance: 0,
    saturation: 0,
    texture: 0,
    clarity: 0,
    dehaze: 0,
    toneCurve: {
      mode: 'POINT',
      points: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    },
    hsl: defaultHsl(),
    sharpening: { amount: 40, radius: 1.0, detail: 25, masking: 0 },
    denoise: { luma: 0, chroma: 0 },
    optics: { enable: false, distortion: 0, vignette: 0, ca: 0 },
  };
}

export function createDefaultEditState(assetId: string): EditState {
  const historyId = crypto.randomUUID();
  const baseSnapshot = {
    schemaVersion: 1 as const,
    assetId,
    global: defaultGlobal(),
    localAdjustments: [] as LocalAdjustment[],
    aiLayers: [] as AiLayer[],
  };
  return {
    ...baseSnapshot,
    history: {
      headId: historyId,
      nodes: {
        [historyId]: {
          id: historyId,
          label: 'Import',
          snapshot: { ...baseSnapshot },
          ts: new Date().toISOString(),
        },
      },
    },
    snapshots: [] as Snapshot[],
  };
}

interface EditStore {
  currentAssetId: string | null;
  editState: EditState | null;

  setEditState: (state: EditState) => void;
  setCurrentAsset: (assetId: string) => void;
  updateGlobal: (key: keyof GlobalAdjustments, value: unknown) => void;
  updateGlobalPartial: (partial: Partial<GlobalAdjustments>) => void;
  setFilmSim: (filmSim: FilmSimState | undefined) => void;
  addLocalAdjustment: (adj: LocalAdjustment) => void;
  removeLocalAdjustment: (id: string) => void;
  addAiLayer: (layer: AiLayer) => void;
  removeAiLayer: (id: string) => void;
  updateAiLayer: (id: string, updates: Partial<AiLayer>) => void;

  // History
  pushHistory: (label: string) => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  // Snapshots
  createSnapshot: (name: string) => void;

  // Reset
  resetToDefault: () => void;
}

export const useEditStore = create<EditStore>((set, get) => ({
  currentAssetId: null,
  editState: null,

  setEditState: (state) => set({ editState: state }),

  setCurrentAsset: (assetId) => {
    set({
      currentAssetId: assetId,
      editState: createDefaultEditState(assetId),
    });
  },

  updateGlobal: (key, value) => {
    const { editState } = get();
    if (!editState) return;
    set({
      editState: {
        ...editState,
        global: {
          ...editState.global,
          [key]: value,
        },
      },
    });
  },

  updateGlobalPartial: (partial) => {
    const { editState } = get();
    if (!editState) return;
    set({
      editState: {
        ...editState,
        global: { ...editState.global, ...partial },
      },
    });
  },

  setFilmSim: (filmSim) => {
    const { editState } = get();
    if (!editState) return;
    set({
      editState: { ...editState, filmSim },
    });
  },

  addLocalAdjustment: (adj) => {
    const { editState } = get();
    if (!editState) return;
    set({
      editState: {
        ...editState,
        localAdjustments: [...editState.localAdjustments, adj],
      },
    });
  },

  removeLocalAdjustment: (id) => {
    const { editState } = get();
    if (!editState) return;
    set({
      editState: {
        ...editState,
        localAdjustments: editState.localAdjustments.filter(a => a.id !== id),
      },
    });
  },

  addAiLayer: (layer) => {
    const { editState } = get();
    if (!editState) return;
    set({
      editState: {
        ...editState,
        aiLayers: [...editState.aiLayers, layer],
      },
    });
  },

  removeAiLayer: (id) => {
    const { editState } = get();
    if (!editState) return;
    set({
      editState: {
        ...editState,
        aiLayers: editState.aiLayers.filter(l => l.id !== id),
      },
    });
  },

  updateAiLayer: (id, updates) => {
    const { editState } = get();
    if (!editState) return;
    set({
      editState: {
        ...editState,
        aiLayers: editState.aiLayers.map(l =>
          l.id === id ? { ...l, ...updates } : l
        ),
      },
    });
  },

  pushHistory: (label) => {
    const { editState } = get();
    if (!editState) return;

    const newId = crypto.randomUUID();
    const snapshot = {
      schemaVersion: editState.schemaVersion,
      assetId: editState.assetId,
      global: editState.global,
      localAdjustments: editState.localAdjustments,
      aiLayers: editState.aiLayers,
      filmSim: editState.filmSim,
      crop: editState.crop,
      transform: editState.transform,
      effects: editState.effects,
      colorGrading: editState.colorGrading,
    };
    const newNode: HistoryNode = {
      id: newId,
      parentId: editState.history.headId,
      label,
      snapshot,
      ts: new Date().toISOString(),
    };

    const nodes = { ...editState.history.nodes, [newId]: newNode };

    // Prune if needed
    const nodeKeys = Object.keys(nodes);
    if (nodeKeys.length > MAX_HISTORY_NODES) {
      // Build the main chain to protect from pruning
      const mainChain = new Set<string>();
      let current: string | undefined = newId;
      while (current) {
        mainChain.add(current);
        current = nodes[current]?.parentId;
      }

      const sorted = Object.values(nodes).sort(
        (a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime()
      );

      let remaining = nodeKeys.length;
      for (const node of sorted) {
        if (remaining <= MAX_HISTORY_NODES) break;
        if (!mainChain.has(node.id)) {
          delete nodes[node.id];
          remaining--;
        }
      }
    }

    set({
      editState: {
        ...editState,
        history: { headId: newId, nodes },
      },
    });
  },

  undo: () => {
    const { editState } = get();
    if (!editState) return;
    const currentNode = editState.history.nodes[editState.history.headId];
    if (!currentNode?.parentId) return;

    const parentNode = editState.history.nodes[currentNode.parentId];
    if (!parentNode) return;

    set({
      editState: {
        ...parentNode.snapshot,
        history: { ...editState.history, headId: parentNode.id },
        snapshots: editState.snapshots,
      } as EditState,
    });
  },

  redo: () => {
    const { editState } = get();
    if (!editState) return;

    const children = Object.values(editState.history.nodes).filter(
      n => n.parentId === editState.history.headId
    );
    if (children.length === 0) return;

    // Pick the most recent child
    const newest = children.sort(
      (a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime()
    )[0];

    set({
      editState: {
        ...newest.snapshot,
        history: { ...editState.history, headId: newest.id },
        snapshots: editState.snapshots,
      } as EditState,
    });
  },

  canUndo: () => {
    const { editState } = get();
    if (!editState) return false;
    const currentNode = editState.history.nodes[editState.history.headId];
    return !!currentNode?.parentId;
  },

  canRedo: () => {
    const { editState } = get();
    if (!editState) return false;
    return Object.values(editState.history.nodes).some(
      n => n.parentId === editState.history.headId
    );
  },

  createSnapshot: (name) => {
    const { editState } = get();
    if (!editState) return;

    const snapshot: Snapshot = {
      id: crypto.randomUUID(),
      name,
      stateHash: editState.history.headId,
      createdAt: new Date().toISOString(),
    };

    set({
      editState: {
        ...editState,
        snapshots: [...editState.snapshots, snapshot],
      },
    });
  },

  resetToDefault: () => {
    const { currentAssetId } = get();
    if (!currentAssetId) return;
    set({ editState: createDefaultEditState(currentAssetId) });
  },
}));
