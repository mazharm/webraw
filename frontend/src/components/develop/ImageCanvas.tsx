import { tokens } from '@fluentui/react-components';
import { useRef, useEffect, useState, useCallback } from 'react';

interface Props {
  previewUrl: string | null;
  isLoading: boolean;
}

export function ImageCanvas({ previewUrl, isLoading }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!previewUrl) return;

    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      drawImage();
    };
    img.src = previewUrl;
  }, [previewUrl]);

  const drawImage = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const img = imageRef.current;
    if (!canvas || !container || !img) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = tokens.colorNeutralBackground4;
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Fit image to canvas
    const scale = Math.min(
      rect.width / img.width,
      rect.height / img.height,
    ) * zoom;

    const drawW = img.width * scale;
    const drawH = img.height * scale;
    const drawX = (rect.width - drawW) / 2 + pan.x;
    const drawY = (rect.height - drawH) / 2 + pan.y;

    ctx.drawImage(img, drawX, drawY, drawW, drawH);
  }, [zoom, pan]);

  useEffect(() => {
    drawImage();
  }, [drawImage]);

  useEffect(() => {
    const observer = new ResizeObserver(() => drawImage());
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [drawImage]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(z => Math.max(0.1, Math.min(10, z * delta)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  }, [pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDoubleClick = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        cursor: isDragging ? 'grabbing' : 'grab',
        overflow: 'hidden',
      }}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onDoubleClick={handleDoubleClick}
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
    </div>
  );
}
