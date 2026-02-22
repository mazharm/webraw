import { PanelSection } from '../../common/PanelSection';
import { AdjustmentSlider } from '../../common/AdjustmentSlider';
import { useEditStore } from '../../../stores/editStore';

export function BasicPanel() {
  const editState = useEditStore(s => s.editState);
  const updateGlobal = useEditStore(s => s.updateGlobal);
  const pushHistory = useEditStore(s => s.pushHistory);

  if (!editState) return null;
  const g = editState.global;

  const commit = (label: string) => pushHistory(label);

  return (
    <PanelSection title="Basic" defaultOpen>
      <AdjustmentSlider
        label="Exposure"
        value={g.exposure}
        min={-5} max={5} step={0.01}
        unit="EV"
        onChange={v => updateGlobal('exposure', v)}
        onChangeCommit={() => commit('Exposure')}
      />
      <AdjustmentSlider
        label="Contrast"
        value={g.contrast}
        min={-100} max={100}
        onChange={v => updateGlobal('contrast', v)}
        onChangeCommit={() => commit('Contrast')}
      />
      <AdjustmentSlider
        label="Highlights"
        value={g.highlights}
        min={-100} max={100}
        onChange={v => updateGlobal('highlights', v)}
        onChangeCommit={() => commit('Highlights')}
      />
      <AdjustmentSlider
        label="Shadows"
        value={g.shadows}
        min={-100} max={100}
        onChange={v => updateGlobal('shadows', v)}
        onChangeCommit={() => commit('Shadows')}
      />
      <AdjustmentSlider
        label="Whites"
        value={g.whites}
        min={-100} max={100}
        onChange={v => updateGlobal('whites', v)}
        onChangeCommit={() => commit('Whites')}
      />
      <AdjustmentSlider
        label="Blacks"
        value={g.blacks}
        min={-100} max={100}
        onChange={v => updateGlobal('blacks', v)}
        onChangeCommit={() => commit('Blacks')}
      />

      <div style={{ height: 8 }} />

      <AdjustmentSlider
        label="Temp"
        value={g.temperature}
        min={2000} max={50000} step={50}
        unit="K"
        onChange={v => updateGlobal('temperature', v)}
        onChangeCommit={() => commit('Temperature')}
      />
      <AdjustmentSlider
        label="Tint"
        value={g.tint}
        min={-150} max={150}
        onChange={v => updateGlobal('tint', v)}
        onChangeCommit={() => commit('Tint')}
      />

      <div style={{ height: 8 }} />

      <AdjustmentSlider
        label="Texture"
        value={g.texture}
        min={-100} max={100}
        onChange={v => updateGlobal('texture', v)}
        onChangeCommit={() => commit('Texture')}
      />
      <AdjustmentSlider
        label="Clarity"
        value={g.clarity}
        min={-100} max={100}
        onChange={v => updateGlobal('clarity', v)}
        onChangeCommit={() => commit('Clarity')}
      />
      <AdjustmentSlider
        label="Dehaze"
        value={g.dehaze}
        min={-100} max={100}
        onChange={v => updateGlobal('dehaze', v)}
        onChangeCommit={() => commit('Dehaze')}
      />

      <div style={{ height: 8 }} />

      <AdjustmentSlider
        label="Vibrance"
        value={g.vibrance}
        min={-100} max={100}
        onChange={v => updateGlobal('vibrance', v)}
        onChangeCommit={() => commit('Vibrance')}
      />
      <AdjustmentSlider
        label="Saturation"
        value={g.saturation}
        min={-100} max={100}
        onChange={v => updateGlobal('saturation', v)}
        onChangeCommit={() => commit('Saturation')}
      />
    </PanelSection>
  );
}
