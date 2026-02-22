import { PanelSection } from '../../common/PanelSection';
import { tokens, Select, Text } from '@fluentui/react-components';
import { useEditStore } from '../../../stores/editStore';
import { useRef, useEffect, useCallback, useState } from 'react';
import type { CurvePoint } from '../../../types';

export function ToneCurvePanel() {
  const editState = useEditStore(s => s.editState);
  const updateGlobal = useEditStore(s => s.updateGlobal);
  const pushHistory = useEditStore(s => s.pushHistory);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);

  if (!editState) return null;
  const curve = editState.global.toneCurve;

  const drawCurve = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Background grid
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let i = 0.25; i < 1; i += 0.25) {
      ctx.beginPath();
      ctx.moveTo(i * w, 0);
      ctx.lineTo(i * w, h);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * h);
      ctx.lineTo(w, i * h);
      ctx.stroke();
    }

    // Diagonal reference
    ctx.strokeStyle = '#555';
    ctx.beginPath();
    ctx.moveTo(0, h);
    ctx.lineTo(w, 0);
    ctx.stroke();

    // Curve
    const points = [...curve.points].sort((a, b) => a.x - b.x);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const x = points[i].x * w;
      const y = (1 - points[i].y) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Points
    for (const pt of points) {
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(pt.x * w, (1 - pt.y) * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [curve.points]);

  useEffect(() => { drawCurve(); }, [drawCurve]);

  const handleCanvasMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = 1 - (e.clientY - rect.top) / rect.height;

    // Find nearest point
    let nearest = -1;
    let minDist = Infinity;
    for (let i = 0; i < curve.points.length; i++) {
      const dx = curve.points[i].x - x;
      const dy = curve.points[i].y - y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minDist && dist < 0.05) {
        minDist = dist;
        nearest = i;
      }
    }

    if (nearest >= 0) {
      setDragIndex(nearest);
    } else {
      // Add new point
      const newPoints = [...curve.points, { x, y }].sort((a, b) => a.x - b.x);
      updateGlobal('toneCurve', { ...curve, points: newPoints });
    }
  }, [curve, updateGlobal]);

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (dragIndex === null) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, 1 - (e.clientY - rect.top) / rect.height));

    const newPoints = [...curve.points];
    newPoints[dragIndex] = { x, y };
    updateGlobal('toneCurve', { ...curve, points: newPoints });
  }, [dragIndex, curve, updateGlobal]);

  const handleCanvasMouseUp = useCallback(() => {
    if (dragIndex !== null) {
      pushHistory('Tone Curve');
    }
    setDragIndex(null);
  }, [dragIndex, pushHistory]);

  return (
    <PanelSection title="Tone Curve">
      <Select
        size="small"
        value={curve.mode}
        onChange={(_, data) => updateGlobal('toneCurve', { ...curve, mode: data.value })}
        style={{ marginBottom: 8 }}
      >
        <option value="POINT">Point Curve</option>
        <option value="PARAMETRIC">Parametric</option>
      </Select>
      <canvas
        ref={canvasRef}
        width={200}
        height={200}
        style={{ width: '100%', aspectRatio: '1', cursor: 'crosshair', borderRadius: 4 }}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={handleCanvasMouseUp}
        onMouseLeave={handleCanvasMouseUp}
      />
    </PanelSection>
  );
}
