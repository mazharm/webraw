import { PanelSection } from '../../common/PanelSection';
import { AdjustmentSlider } from '../../common/AdjustmentSlider';
import { useEditStore } from '../../../stores/editStore';
import { Switch } from '@fluentui/react-components';

export function OpticsPanel() {
  const editState = useEditStore(s => s.editState);
  const updateGlobal = useEditStore(s => s.updateGlobal);
  const pushHistory = useEditStore(s => s.pushHistory);

  if (!editState) return null;
  const optics = editState.global.optics;

  const updateOptics = (key: string, value: number | boolean) => {
    updateGlobal('optics', { ...optics, [key]: value });
  };

  return (
    <PanelSection title="Optics">
      <Switch
        label="Enable Lens Corrections"
        checked={optics.enable}
        onChange={(_, data) => {
          updateOptics('enable', data.checked);
          pushHistory('Lens Corrections');
        }}
      />
      <AdjustmentSlider
        label="Distortion"
        value={optics.distortion}
        min={-100} max={100}
        onChange={v => updateOptics('distortion', v)}
        onChangeCommit={() => pushHistory('Distortion')}
      />
      <AdjustmentSlider
        label="Vignette"
        value={optics.vignette}
        min={-100} max={100}
        onChange={v => updateOptics('vignette', v)}
        onChangeCommit={() => pushHistory('Vignette Correction')}
      />
      <AdjustmentSlider
        label="CA"
        value={optics.ca}
        min={0} max={100}
        onChange={v => updateOptics('ca', v)}
        onChangeCommit={() => pushHistory('CA Correction')}
      />
    </PanelSection>
  );
}
