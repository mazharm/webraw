import { create } from 'zustand';

interface SettingsStore {
  geminiApiKey: string | null;
  keyMode: 'session' | 'remember';
  redactPrompts: boolean;
  colorSpace: 'sRGB' | 'DisplayP3';
  stripLocationOnExport: boolean;
  backendHealthy: boolean;

  setGeminiApiKey: (key: string | null) => void;
  setKeyMode: (mode: 'session' | 'remember') => void;
  setRedactPrompts: (redact: boolean) => void;
  setColorSpace: (cs: 'sRGB' | 'DisplayP3') => void;
  setStripLocationOnExport: (strip: boolean) => void;
  setBackendHealthy: (healthy: boolean) => void;
}

export const useSettingsStore = create<SettingsStore>((set) => ({
  geminiApiKey: null,
  keyMode: 'session',
  redactPrompts: false,
  colorSpace: 'sRGB',
  stripLocationOnExport: true,
  backendHealthy: false,

  setGeminiApiKey: (key) => set({ geminiApiKey: key }),
  setKeyMode: (mode) => set({ keyMode: mode }),
  setRedactPrompts: (redact) => set({ redactPrompts: redact }),
  setColorSpace: (cs) => set({ colorSpace: cs }),
  setStripLocationOnExport: (strip) => set({ stripLocationOnExport: strip }),
  setBackendHealthy: (healthy) => set({ backendHealthy: healthy }),
}));
