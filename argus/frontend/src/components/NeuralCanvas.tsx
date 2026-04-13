import { useRef, useState, useEffect } from 'react';
import { THREAT_COLORS } from '../constants';
import { randInt } from '../utils/helpers';
import type { AttackEvent } from '../types';

interface NeuralCanvasProps {
  attacks: AttackEvent[];
}

interface Node {
  x: number;
  y: number;
  r: number;
  type: 'CORE' | 'LAYER' | 'SENSOR';
  color: string;
  pulse: number;
}

interface Particle {
  sx: number; sy: number;
  tx: number; ty: number;
  progress: number;
  speed: number;
  arc: number;
  color: string;
  blocked: boolean;
}

interface Ring {
  x: number; y: number;
  progress: number;
  color: string;
}

export function NeuralCanvas({ attacks }: NeuralCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<{
    nodes: Node[];
    particles: Particle[];
    rings: Ring[];
    pulses: Ring[];
    w: number;
    h: number;
  }>({ nodes: [], particles: [], rings: [], pulses: [], w: 0, h: 0 });
  const [size, setSize] = useState({ width: 260, height: 220 });

  // Responsive sizing via ResizeObserver
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const update = () => {
      const { width, height } = container.getBoundingClientRect();
      if (width > 0 && height > 0)
        setSize({ width: Math.round(width), height: Math.round(height) });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // Rebuild node positions when size changes
  useEffect(() => {
    const s = stateRef.current;
    const { width, height } = size;
    const cx = width / 2,
      cy = height / 2;
    const scale = Math.min(width / 260, height / 220);

    const layerColors = ['#00e5ff', '#d500f9', '#00e676', '#ffab00', '#ff6d00', '#ff4081'];

    s.nodes = [];
    s.nodes.push({ x: cx, y: cy, r: 18 * scale, type: 'CORE', color: '#00e5ff', pulse: 0 });
    for (let i = 0; i < 6; i++) {
      const angle = (i / 6) * Math.PI * 2;
      const dist = 70 * scale;
      s.nodes.push({
        x: cx + Math.cos(angle) * dist,
        y: cy + Math.sin(angle) * dist,
        r: 10 * scale,
        type: 'LAYER',
        color: layerColors[i],
        pulse: Math.random() * Math.PI * 2,
      });
    }
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2;
      const dist = 130 * scale;
      s.nodes.push({
        x: cx + Math.cos(angle) * dist,
        y: cy + Math.sin(angle) * dist,
        r: 6 * scale,
        type: 'SENSOR',
        color: '#1a3060',
        pulse: Math.random() * Math.PI * 2,
      });
    }
    s.w = width;
    s.h = height;
  }, [size]);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const s = stateRef.current;
    const { width, height } = size;
    const scale = Math.min(width / 260, height / 220);
    const maxConnDist = 160 * scale;

    let animId: number;
    let t = 0;

    function draw() {
      t += 0.02;
      ctx!.clearRect(0, 0, width, height);

      // Background
      ctx!.fillStyle = '#030508';
      ctx!.fillRect(0, 0, width, height);

      // Connection lines
      s.nodes.forEach((n, i) => {
        s.nodes.forEach((m, j) => {
          if (j <= i) return;
          const dx = n.x - m.x,
            dy = n.y - m.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > maxConnDist) return;
          const alpha = (1 - dist / maxConnDist) * 0.15;
          ctx!.strokeStyle = `rgba(0,229,255,${alpha})`;
          ctx!.lineWidth = 0.5;
          ctx!.beginPath();
          ctx!.moveTo(n.x, n.y);
          ctx!.lineTo(m.x, m.y);
          ctx!.stroke();
        });
      });

      // Draw rings
      s.rings = s.rings.filter((r) => {
        r.progress += 0.025;
        if (r.progress >= 1) return false;
        const radius = r.progress * 90 * scale;
        const alpha = (1 - r.progress) * 0.5;
        ctx!.strokeStyle =
          r.color + Math.round(alpha * 255).toString(16).padStart(2, '0');
        ctx!.lineWidth = 1.5;
        ctx!.beginPath();
        ctx!.arc(r.x, r.y, radius, 0, Math.PI * 2);
        ctx!.stroke();
        return true;
      });

      // Draw particles
      s.particles = s.particles.filter((p) => {
        p.progress += p.speed;
        if (p.progress >= 1) {
          s.rings.push({ x: p.tx, y: p.ty, progress: 0, color: p.color });
          if (p.blocked) {
            s.pulses.push({ x: s.nodes[0].x, y: s.nodes[0].y, progress: 0, color: '#00e676' });
          }
          return false;
        }
        const eased = 1 - Math.pow(1 - p.progress, 3);
        const x = p.sx + (p.tx - p.sx) * eased;
        const y = p.sy + (p.ty - p.sy) * eased + Math.sin(p.progress * Math.PI) * p.arc;
        for (let tr = 0; tr < 5; tr++) {
          const tp = Math.max(0, p.progress - tr * 0.03);
          const ex = 1 - Math.pow(1 - tp, 3);
          const tx = p.sx + (p.tx - p.sx) * ex;
          const ty = p.sy + (p.ty - p.sy) * ex + Math.sin(tp * Math.PI) * p.arc;
          ctx!.beginPath();
          ctx!.arc(tx, ty, (5 - tr) * 0.8, 0, Math.PI * 2);
          ctx!.fillStyle =
            p.color + Math.round((0.7 - tr * 0.12) * 255).toString(16).padStart(2, '0');
          ctx!.fill();
        }
        ctx!.beginPath();
        ctx!.arc(x, y, 4, 0, Math.PI * 2);
        ctx!.fillStyle = p.color;
        ctx!.fill();
        ctx!.beginPath();
        ctx!.arc(x, y, 2, 0, Math.PI * 2);
        ctx!.fillStyle = '#fff';
        ctx!.fill();
        return true;
      });

      // Draw pulses
      s.pulses = s.pulses.filter((p) => {
        p.progress += 0.04;
        if (p.progress >= 1) return false;
        const r = p.progress * 40 * scale;
        ctx!.strokeStyle =
          p.color + Math.round((1 - p.progress) * 180).toString(16).padStart(2, '0');
        ctx!.lineWidth = 2;
        ctx!.beginPath();
        ctx!.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx!.stroke();
        return true;
      });

      // Draw nodes
      s.nodes.forEach((n) => {
        n.pulse += 0.05;
        const glow = 0.5 + 0.5 * Math.sin(n.pulse);
        const r = n.r + glow * (n.type === 'CORE' ? 3 : 1.5);
        ctx!.beginPath();
        ctx!.arc(n.x, n.y, r + 8, 0, Math.PI * 2);
        ctx!.fillStyle = n.color + '18';
        ctx!.fill();
        ctx!.beginPath();
        ctx!.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx!.fillStyle = n.type === 'SENSOR' ? '#0b1830' : n.color + 'cc';
        ctx!.fill();
        ctx!.strokeStyle = n.color;
        ctx!.lineWidth = n.type === 'CORE' ? 2 : 1;
        ctx!.stroke();
        if (n.type === 'CORE') {
          ctx!.font = `bold ${Math.round(9 * scale)}px 'Share Tech Mono'`;
          ctx!.fillStyle = '#00e5ff';
          ctx!.textAlign = 'center';
          ctx!.fillText('ARGUS', n.x, n.y + 3);
        }
      });

      animId = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(animId);
  }, [size]);

  // Trigger particle on new attack
  useEffect(() => {
    if (!attacks.length) return;
    const atk = attacks[0];
    const s = stateRef.current;
    if (!s.nodes.length) return;
    const core = s.nodes[0];
    const sensorIdx = randInt(7, s.nodes.length - 1);
    const sensor = s.nodes[sensorIdx];
    if (!sensor) return;
    const scale = Math.min(s.w / 260, s.h / 220) || 1;
    s.particles.push({
      sx: sensor.x,
      sy: sensor.y,
      tx: core.x,
      ty: core.y,
      progress: 0,
      speed: 0.022,
      arc: randInt(-40, 40) * scale,
      color: THREAT_COLORS[atk.type] || '#ff1744',
      blocked: atk.blocked,
    });
  }, [attacks]);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: 120 }}>
      <canvas
        ref={canvasRef}
        width={size.width}
        height={size.height}
        style={{ display: 'block', borderRadius: 10, width: '100%', height: '100%' }}
      />
    </div>
  );
}
