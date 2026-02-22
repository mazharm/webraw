import { PanelSection } from '../../common/PanelSection';
import { AdjustmentSlider } from '../../common/AdjustmentSlider';
import { Select, Text, tokens } from '@fluentui/react-components';
import { useEditStore } from '../../../stores/editStore';
import { useState } from 'react';
import type { HslChannel } from '../../../types';

const HSL_CHANNELS: HslChannel[] = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta'];

const CHANNEL_COLORS: Record<HslChannel, string> = {
  red: '#ff4444',
  orange: '#ff8800',
  yellow: '#ffcc00',
  green: '#44cc44',
  aqua: '#44cccc',
  blue: '#4488ff',
  purple: '#8844ff',
  magenta: '#ff44cc',
};

export function ColorPanel() {
  const editState = useEditStore(s => s.editState);
  const updateGlobal = useEditStore(s => s.updateGlobal);
  const pushHistory = useEditStore(s => s.pushHistory);
  const [activeChannel, setActiveChannel] = useState<HslChannel>('red');

  if (!editState) return null;
  const hsl = editState.global.hsl;

  const channelValues = hsl[activeChannel] || { h: 0, s: 0, l: 0 };

  const updateHslChannel = (prop: 'h' | 's' | 'l', value: number) => {
    updateGlobal('hsl', {
      ...hsl,
      [activeChannel]: { ...channelValues, [prop]: value },
    });
  };

  return (
    <PanelSection title="Color (HSL)">
      {/* Channel selector */}
      <div style={{ display: 'flex', gap: 2, marginBottom: 8 }}>
        {HSL_CHANNELS.map(ch => (
          <button
            key={ch}
            onClick={() => setActiveChannel(ch)}
            style={{
              flex: 1,
              height: 20,
              border: activeChannel === ch ? '2px solid white' : '1px solid #555',
              borderRadius: 3,
              background: CHANNEL_COLORS[ch],
              cursor: 'pointer',
              opacity: activeChannel === ch ? 1 : 0.5,
            }}
            aria-label={ch}
            title={ch}
          />
        ))}
      </div>

      <Text size={200} style={{ textTransform: 'capitalize', marginBottom: 4, display: 'block' }}>
        {activeChannel}
      </Text>

      <AdjustmentSlider
        label="Hue"
        value={channelValues.h}
        min={-100} max={100}
        onChange={v => updateHslChannel('h', v)}
        onChangeCommit={() => pushHistory(`HSL ${activeChannel} Hue`)}
      />
      <AdjustmentSlider
        label="Saturation"
        value={channelValues.s}
        min={-100} max={100}
        onChange={v => updateHslChannel('s', v)}
        onChangeCommit={() => pushHistory(`HSL ${activeChannel} Saturation`)}
      />
      <AdjustmentSlider
        label="Luminance"
        value={channelValues.l}
        min={-100} max={100}
        onChange={v => updateHslChannel('l', v)}
        onChangeCommit={() => pushHistory(`HSL ${activeChannel} Luminance`)}
      />
    </PanelSection>
  );
}
