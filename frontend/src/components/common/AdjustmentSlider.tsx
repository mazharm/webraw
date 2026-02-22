import { Slider, Label, Input, tokens } from '@fluentui/react-components';
import { useCallback, useState, useRef } from 'react';

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (value: number) => void;
  onChangeCommit?: (value: number) => void;
}

export function AdjustmentSlider({
  label,
  value,
  min,
  max,
  step = 1,
  unit,
  onChange,
  onChangeCommit,
}: Props) {
  const [localValue, setLocalValue] = useState<string>(value.toString());
  const [isEditing, setIsEditing] = useState(false);

  const handleSliderChange = useCallback((_: unknown, data: { value: number }) => {
    onChange(data.value);
    setLocalValue(data.value.toString());
  }, [onChange]);

  const handleDoubleClick = useCallback(() => {
    onChange(0);
    setLocalValue('0');
    onChangeCommit?.(0);
  }, [onChange, onChangeCommit]);

  const handleInputBlur = useCallback(() => {
    setIsEditing(false);
    const parsed = parseFloat(localValue);
    if (!isNaN(parsed)) {
      const clamped = Math.max(min, Math.min(max, parsed));
      onChange(clamped);
      onChangeCommit?.(clamped);
    }
  }, [localValue, min, max, onChange, onChangeCommit]);

  const handleInputKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleInputBlur();
    }
  }, [handleInputBlur]);

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      padding: '2px 0',
    }}>
      <Label
        size="small"
        style={{ width: 80, flexShrink: 0, fontSize: 11 }}
        onDoubleClick={handleDoubleClick}
      >
        {label}
      </Label>
      <Slider
        size="small"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleSliderChange}
        style={{ flex: 1 }}
        aria-label={label}
      />
      {isEditing ? (
        <Input
          size="small"
          value={localValue}
          onChange={(_, data) => setLocalValue(data.value)}
          onBlur={handleInputBlur}
          onKeyDown={handleInputKeyDown}
          style={{ width: 56, textAlign: 'right' }}
          autoFocus
        />
      ) : (
        <span
          onClick={() => { setIsEditing(true); setLocalValue(value.toString()); }}
          style={{
            width: 56,
            textAlign: 'right',
            fontSize: 11,
            cursor: 'text',
            color: tokens.colorNeutralForeground2,
            userSelect: 'none',
          }}
        >
          {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(1)) : value}
          {unit && <span style={{ fontSize: 9, marginLeft: 1 }}>{unit}</span>}
        </span>
      )}
    </div>
  );
}
