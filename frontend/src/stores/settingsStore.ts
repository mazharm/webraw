import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SettingsStore {
  geminiApiKey: string | null;
  anthropicApiKey: string | null;
  openaiApiKey: string | null;
  keyMode: 'session' | 'remember';
  redactPrompts: boolean;
  colorSpace: 'sRGB' | 'DisplayP3';
  stripLocationOnExport: boolean;
  backendHealthy: boolean;

  setGeminiApiKey: (key: string | null) => void;
  setAnthropicApiKey: (key: string | null) => void;
  setOpenaiApiKey: (key: string | null) => void;
  setKeyMode: (mode: 'session' | 'remember') => void;
  setRedactPrompts: (redact: boolean) => void;
  setColorSpace: (cs: 'sRGB' | 'DisplayP3') => void;
  setStripLocationOnExport: (strip: boolean) => void;
  setBackendHealthy: (healthy: boolean) => void;
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      geminiApiKey: null,
      anthropicApiKey: null,
      openaiApiKey: null,
      keyMode: 'session',
      redactPrompts: false,
      colorSpace: 'sRGB',
      stripLocationOnExport: true,
      backendHealthy: false,

      setGeminiApiKey: (key) => set({ geminiApiKey: key }),
      setAnthropicApiKey: (key) => set({ anthropicApiKey: key }),
      setOpenaiApiKey: (key) => set({ openaiApiKey: key }),
      setKeyMode: (mode) => set({ keyMode: mode }),
      setRedactPrompts: (redact) => set({ redactPrompts: redact }),
      setColorSpace: (cs) => set({ colorSpace: cs }),
      setStripLocationOnExport: (strip) => set({ stripLocationOnExport: strip }),
      setBackendHealthy: (healthy) => set({ backendHealthy: healthy }),
    }),
    {
      name: 'webraw-settings',
      partialize: (state) => ({
        geminiApiKey: state.keyMode === 'remember' ? state.geminiApiKey : null,
        anthropicApiKey: state.keyMode === 'remember' ? state.anthropicApiKey : null,
        openaiApiKey: state.keyMode === 'remember' ? state.openaiApiKey : null,
        keyMode: state.keyMode,
        redactPrompts: state.redactPrompts,
        colorSpace: state.colorSpace,
        stripLocationOnExport: state.stripLocationOnExport,
        // backendHealthy is intentionally excluded — ephemeral runtime state
      }),
    },
  ),
);
