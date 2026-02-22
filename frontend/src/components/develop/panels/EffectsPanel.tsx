import { PanelSection } from '../../common/PanelSection';
import { AdjustmentSlider } from '../../common/AdjustmentSlider';
import { useEditStore } from '../../../stores/editStore';
import { Text } from '@fluentui/react-components';

export function EffectsPanel() {
  const editState = useEditStore(s => s.editState);
  const updateGlobal = useEditStore(s => s.updateGlobal);
  const pushHistory = useEditStore(s => s.pushHistory);

  if (!editState) return null;
  const effects = editState.global.effects || { grainAmount: 0, grainSize: 1.0, vignetteAmount: 0 };
  const hasFilmSim = !!editState.filmSim;

  const updateEffects = (key: string, value: number) => {
    updateGlobal('effects', { ...effects, [key]: value });
  };

  return (
    <PanelSection title="Effects">
      {hasFilmSim && (
        <Text size={200} style={{ color: '#888', display: 'block', marginBottom: 4 }}>
          Grain is controlled by Film Sim when active
        </Text>
      )}

      <AdjustmentSlider
        label="Grain"
        value={effects.grainAmount}
        min={0} max={100}
        onChange={v => updateEffects('grainAmount', v)}
        onChangeCommit={() => pushHistory('Grain Amount')}
      />
      <AdjustmentSlider
        label="Grain Size"
        value={effects.grainSize}
        min={0.5} max={3} step={0.1}
        onChange={v => updateEffects('grainSize', v)}
        onChangeCommit={() => pushHistory('Grain Size')}
      />
      <AdjustmentSlider
        label="Vignette"
        value={effects.vignetteAmount}
        min={-100} max={100}
        onChange={v => updateEffects('vignetteAmount', v)}
        onChangeCommit={() => pushHistory('Post-crop Vignette')}
      />
    </PanelSection>
  );
}
