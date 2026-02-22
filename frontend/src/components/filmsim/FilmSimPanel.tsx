import { PanelSection } from '../common/PanelSection';
import { AdjustmentSlider } from '../common/AdjustmentSlider';
import { tokens, Button, Text, Select } from '@fluentui/react-components';
import { useEditStore } from '../../stores/editStore';
import type { FilmSimLook } from '../../types';

const FILM_LOOKS: FilmSimLook[] = [
  {
    id: 'chrome_like',
    name: 'Chrome Like',
    baseCurve: 'contrast_medium',
    lutFile: 'chrome_v1.cube',
    matrix: [1.1, -0.05, -0.05, -0.02, 1.05, -0.03, 0.0, -0.02, 1.02],
    defaults: { strength: 1.0, grainAmount: 25, grainSize: 1.0, saturation: 1.1 },
  },
  {
    id: 'velvia_like',
    name: 'Velvia Like',
    baseCurve: 'contrast_high',
    lutFile: 'velvia_v1.cube',
    matrix: [1.15, -0.08, -0.07, -0.03, 1.08, -0.05, 0.01, -0.03, 1.02],
    defaults: { strength: 1.0, grainAmount: 15, grainSize: 0.8, saturation: 1.3 },
  },
  {
    id: 'provia_like',
    name: 'Provia Like',
    baseCurve: 'contrast_medium',
    lutFile: 'provia_v1.cube',
    matrix: [1.05, -0.03, -0.02, -0.01, 1.03, -0.02, 0.0, -0.01, 1.01],
    defaults: { strength: 1.0, grainAmount: 10, grainSize: 1.0, saturation: 1.05 },
  },
  {
    id: 'eterna_like',
    name: 'Eterna Like',
    baseCurve: 'contrast_low',
    lutFile: 'eterna_v1.cube',
    matrix: [1.02, -0.01, -0.01, -0.01, 1.02, -0.01, 0.0, -0.01, 1.01],
    defaults: { strength: 1.0, grainAmount: 20, grainSize: 1.2, saturation: 0.9 },
  },
  {
    id: 'astia_like',
    name: 'Astia Like',
    baseCurve: 'contrast_soft',
    lutFile: 'astia_v1.cube',
    matrix: [1.05, -0.03, -0.02, -0.01, 1.04, -0.03, 0.0, -0.01, 1.01],
    defaults: { strength: 1.0, grainAmount: 12, grainSize: 1.0, saturation: 1.0 },
  },
  {
    id: 'acros_like',
    name: 'Acros Like (B&W)',
    baseCurve: 'contrast_medium_bw',
    lutFile: 'acros_v1.cube',
    matrix: [0.299, 0.587, 0.114, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114],
    defaults: { strength: 1.0, grainAmount: 30, grainSize: 1.2, saturation: 0.0 },
  },
];

export function FilmSimPanel() {
  const editState = useEditStore(s => s.editState);
  const setFilmSim = useEditStore(s => s.setFilmSim);
  const pushHistory = useEditStore(s => s.pushHistory);

  if (!editState) return null;
  const filmSim = editState.filmSim;

  const handleSelectLook = (lookId: string) => {
    if (lookId === 'none') {
      setFilmSim(undefined);
      pushHistory('Remove Film Sim');
      return;
    }
    const look = FILM_LOOKS.find(l => l.id === lookId);
    if (!look) return;
    setFilmSim({
      id: look.id,
      strength: look.defaults.strength,
      grainAmount: look.defaults.grainAmount,
      grainSize: look.defaults.grainSize,
    });
    pushHistory(`Film Sim: ${look.name}`);
  };

  return (
    <PanelSection title="Film Sims">
      {/* Look grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 8 }}>
        <button
          onClick={() => handleSelectLook('none')}
          style={{
            padding: '6px 4px',
            border: !filmSim ? `2px solid ${tokens.colorBrandStroke1}` : '1px solid #444',
            borderRadius: 4,
            background: tokens.colorNeutralBackground3,
            color: tokens.colorNeutralForeground1,
            cursor: 'pointer',
            fontSize: 11,
          }}
        >
          None
        </button>
        {FILM_LOOKS.map(look => (
          <button
            key={look.id}
            onClick={() => handleSelectLook(look.id)}
            style={{
              padding: '6px 4px',
              border: filmSim?.id === look.id ? `2px solid ${tokens.colorBrandStroke1}` : '1px solid #444',
              borderRadius: 4,
              background: tokens.colorNeutralBackground3,
              color: tokens.colorNeutralForeground1,
              cursor: 'pointer',
              fontSize: 11,
            }}
          >
            {look.name}
          </button>
        ))}
      </div>

      {filmSim && (
        <>
          <AdjustmentSlider
            label="Strength"
            value={filmSim.strength * 100}
            min={0} max={200}
            onChange={v => setFilmSim({ ...filmSim, strength: v / 100 })}
            onChangeCommit={() => pushHistory('Film Sim Strength')}
          />
          <AdjustmentSlider
            label="Grain"
            value={filmSim.grainAmount}
            min={0} max={100}
            onChange={v => setFilmSim({ ...filmSim, grainAmount: v })}
            onChangeCommit={() => pushHistory('Film Sim Grain')}
          />
          <AdjustmentSlider
            label="Grain Size"
            value={filmSim.grainSize}
            min={0.5} max={3} step={0.1}
            onChange={v => setFilmSim({ ...filmSim, grainSize: v })}
            onChangeCommit={() => pushHistory('Film Sim Grain Size')}
          />

          {FILM_LOOKS.find(l => l.id === filmSim.id)?.name.includes('B&W') && (
            <div style={{ marginTop: 8 }}>
              <Text size={200} style={{ display: 'block', marginBottom: 4 }}>B&W Filter</Text>
              <div style={{ display: 'flex', gap: 4 }}>
                {(['R', 'Y', 'G'] as const).map(filter => (
                  <Button
                    key={filter}
                    size="small"
                    appearance={filmSim.bwFilter === filter ? 'primary' : 'outline'}
                    onClick={() => {
                      setFilmSim({
                        ...filmSim,
                        bwFilter: filmSim.bwFilter === filter ? undefined : filter,
                      });
                      pushHistory(`B&W Filter: ${filter}`);
                    }}
                  >
                    {filter}
                  </Button>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </PanelSection>
  );
}
