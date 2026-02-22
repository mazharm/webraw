import { tokens, Text } from '@fluentui/react-components';
import { useRef, useEffect } from 'react';
import type { HistogramData } from '../../types';

interface Props {
  data: HistogramData | null;
}

export function Histogram({ data }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!data) {
      ctx.fillStyle = '#222';
      ctx.fillRect(0, 0, w, h);
      return;
    }

    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, w, h);

    const maxVal = Math.max(
      ...data.r.slice(1, 254),
      ...data.g.slice(1, 254),
      ...data.b.slice(1, 254),
      1,
    );

    const drawChannel = (channel: number[], color: string, alpha: number) => {
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let i = 0; i < 256; i++) {
        const x = (i / 255) * w;
        const y = h - (channel[i] / maxVal) * h;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      ctx.fill();
    };

    drawChannel(data.r, '#ff4444', 0.4);
    drawChannel(data.g, '#44ff44', 0.4);
    drawChannel(data.b, '#4444ff', 0.4);

    ctx.globalAlpha = 0.6;
    drawChannel(data.lum, '#cccccc', 0.3);
    ctx.globalAlpha = 1;
  }, [data]);

  return (
    <div style={{ padding: '8px 12px', borderBottom: `1px solid ${tokens.colorNeutralStroke2}` }}>
      <canvas
        ref={canvasRef}
        width={256}
        height={80}
        style={{ width: '100%', height: 80, borderRadius: 4 }}
      />
    </div>
  );
}
