import { PanelSection } from '../../common/PanelSection';
import { AdjustmentSlider } from '../../common/AdjustmentSlider';
import { useEditStore } from '../../../stores/editStore';
import { Switch, Text } from '@fluentui/react-components';

export function DetailPanel() {
  const editState = useEditStore(s => s.editState);
  const updateGlobal = useEditStore(s => s.updateGlobal);
  const pushHistory = useEditStore(s => s.pushHistory);

  if (!editState) return null;
  const { sharpening, denoise } = editState.global;

  const updateSharpening = (key: string, value: number) => {
    updateGlobal('sharpening', { ...sharpening, [key]: value });
  };

  const updateDenoise = (key: string, value: number | boolean) => {
    updateGlobal('denoise', { ...denoise, [key]: value });
  };

  return (
    <PanelSection title="Detail">
      <Text size={200} weight="semibold" style={{ display: 'block', marginBottom: 4 }}>Sharpening</Text>
      <AdjustmentSlider
        label="Amount"
        value={sharpening.amount}
        min={0} max={150}
        onChange={v => updateSharpening('amount', v)}
        onChangeCommit={() => pushHistory('Sharpening Amount')}
      />
      <AdjustmentSlider
        label="Radius"
        value={sharpening.radius}
        min={0.5} max={3} step={0.1}
        onChange={v => updateSharpening('radius', v)}
        onChangeCommit={() => pushHistory('Sharpening Radius')}
      />
      <AdjustmentSlider
        label="Detail"
        value={sharpening.detail}
        min={0} max={100}
        onChange={v => updateSharpening('detail', v)}
        onChangeCommit={() => pushHistory('Sharpening Detail')}
      />
      <AdjustmentSlider
        label="Masking"
        value={sharpening.masking}
        min={0} max={100}
        onChange={v => updateSharpening('masking', v)}
        onChangeCommit={() => pushHistory('Sharpening Masking')}
      />

      <div style={{ height: 12 }} />

      <Text size={200} weight="semibold" style={{ display: 'block', marginBottom: 4 }}>Noise Reduction</Text>
      <AdjustmentSlider
        label="Luminance"
        value={denoise.luma}
        min={0} max={100}
        onChange={v => updateDenoise('luma', v)}
        onChangeCommit={() => pushHistory('Denoise Luminance')}
      />
      <AdjustmentSlider
        label="Color"
        value={denoise.chroma}
        min={0} max={100}
        onChange={v => updateDenoise('chroma', v)}
        onChangeCommit={() => pushHistory('Denoise Color')}
      />
      <Switch
        label="Enhanced (NLM)"
        checked={denoise.enhanced ?? false}
        onChange={(_, data) => {
          updateDenoise('enhanced', data.checked);
          pushHistory('Enhanced Denoise');
        }}
      />
    </PanelSection>
  );
}
