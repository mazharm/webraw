import {
  Dialog,
  DialogTrigger,
  DialogSurface,
  DialogTitle,
  DialogBody,
  DialogContent,
  DialogActions,
  Button,
  Input,
  Label,
  Switch,
  Select,
  Field,
} from '@fluentui/react-components';
import { useSettingsStore } from '../../stores/settingsStore';
import { useState } from 'react';

interface Props {
  onDismiss: () => void;
}

export function SettingsDialog({ onDismiss }: Props) {
  const {
    geminiApiKey,
    keyMode,
    redactPrompts,
    colorSpace,
    stripLocationOnExport,
    setGeminiApiKey,
    setKeyMode,
    setRedactPrompts,
    setColorSpace,
    setStripLocationOnExport,
  } = useSettingsStore();

  const [keyInput, setKeyInput] = useState(geminiApiKey ?? '');

  const handleSaveKey = () => {
    setGeminiApiKey(keyInput || null);
  };

  return (
    <Dialog open onOpenChange={(_, data) => !data.open && onDismiss()}>
      <DialogSurface>
        <DialogBody>
          <DialogTitle>Settings</DialogTitle>
          <DialogContent>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <Field label="Gemini API Key">
                <Input
                  type="password"
                  value={keyInput}
                  onChange={(_, data) => setKeyInput(data.value)}
                  placeholder="Enter your Gemini API key"
                  style={{ width: '100%' }}
                />
              </Field>

              <Field label="Key Storage">
                <Select
                  value={keyMode}
                  onChange={(_, data) => setKeyMode(data.value as 'session' | 'remember')}
                >
                  <option value="session">Session only (not persisted)</option>
                  <option value="remember">Remember (encrypted)</option>
                </Select>
              </Field>

              <Switch
                label="Redact AI prompts in project files"
                checked={redactPrompts}
                onChange={(_, data) => setRedactPrompts(data.checked)}
              />

              <Field label="Output Color Space">
                <Select
                  value={colorSpace}
                  onChange={(_, data) => setColorSpace(data.value as 'sRGB' | 'DisplayP3')}
                >
                  <option value="sRGB">sRGB</option>
                  <option value="DisplayP3">Display P3</option>
                </Select>
              </Field>

              <Switch
                label="Strip location metadata on export"
                checked={stripLocationOnExport}
                onChange={(_, data) => setStripLocationOnExport(data.checked)}
              />
            </div>
          </DialogContent>
          <DialogActions>
            <Button appearance="secondary" onClick={onDismiss}>Cancel</Button>
            <Button appearance="primary" onClick={() => { handleSaveKey(); onDismiss(); }}>
              Save
            </Button>
          </DialogActions>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  );
}
