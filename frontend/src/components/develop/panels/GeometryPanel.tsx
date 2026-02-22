import { PanelSection } from '../../common/PanelSection';
import { AdjustmentSlider } from '../../common/AdjustmentSlider';
import { useEditStore } from '../../../stores/editStore';
import { Button, Text } from '@fluentui/react-components';

export function GeometryPanel() {
  const editState = useEditStore(s => s.editState);
  const updateGlobal = useEditStore(s => s.updateGlobal);
  const pushHistory = useEditStore(s => s.pushHistory);

  if (!editState) return null;
  const transform = editState.global.transform || { vertical: 0, horizontal: 0, rotate: 0, aspect: 0 };
  const crop = editState.global.crop || { angle: 0, rect: { x: 0, y: 0, w: 1, h: 1 } };

  const updateTransform = (key: string, value: number) => {
    updateGlobal('transform', { ...transform, [key]: value });
  };

  return (
    <PanelSection title="Geometry">
      <Text size={200} weight="semibold" style={{ display: 'block', marginBottom: 4 }}>Transform</Text>
      <AdjustmentSlider
        label="Vertical"
        value={transform.vertical}
        min={-100} max={100}
        onChange={v => updateTransform('vertical', v)}
        onChangeCommit={() => pushHistory('Transform Vertical')}
      />
      <AdjustmentSlider
        label="Horizontal"
        value={transform.horizontal}
        min={-100} max={100}
        onChange={v => updateTransform('horizontal', v)}
        onChangeCommit={() => pushHistory('Transform Horizontal')}
      />
      <AdjustmentSlider
        label="Rotate"
        value={transform.rotate}
        min={-45} max={45} step={0.1}
        unit="°"
        onChange={v => updateTransform('rotate', v)}
        onChangeCommit={() => pushHistory('Transform Rotate')}
      />
      <AdjustmentSlider
        label="Aspect"
        value={transform.aspect}
        min={-100} max={100}
        onChange={v => updateTransform('aspect', v)}
        onChangeCommit={() => pushHistory('Transform Aspect')}
      />

      <div style={{ height: 12 }} />

      <Text size={200} weight="semibold" style={{ display: 'block', marginBottom: 4 }}>Crop</Text>
      <AdjustmentSlider
        label="Angle"
        value={crop.angle}
        min={-45} max={45} step={0.1}
        unit="°"
        onChange={v => updateGlobal('crop', { ...crop, angle: v })}
        onChangeCommit={() => pushHistory('Crop Angle')}
      />
      <Button
        size="small"
        onClick={() => {
          updateGlobal('crop', undefined);
          updateGlobal('transform', undefined);
          pushHistory('Reset Geometry');
        }}
        style={{ marginTop: 4 }}
      >
        Reset Geometry
      </Button>
    </PanelSection>
  );
}
