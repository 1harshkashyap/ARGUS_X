import { useRef, useEffect } from 'react';
import { THREAT_COLORS } from '../constants';
import type { AttackEvent } from '../types';

interface MiniClusterMapProps {
  attacks: AttackEvent[];
}

interface ClusterNode {
  x: number; y: number;
  vx: number; vy: number;
  type: string;
  r: number;
  color: string;
  cx: number; cy: number;
}

export function MiniClusterMap({ attacks }: MiniClusterMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<Record<string, ClusterNode>>({});

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width,
      H = canvas.height;

    const centers: Record<string, { x: number; y: number }> = {
      INSTRUCTION_OVERRIDE: { x: W * 0.2, y: H * 0.3 },
      ROLE_OVERRIDE: { x: W * 0.5, y: H * 0.15 },
      DATA_EXFIL: { x: W * 0.8, y: H * 0.3 },
      INDIRECT_INJECTION: { x: W * 0.3, y: H * 0.75 },
      OBFUSCATED: { x: W * 0.7, y: H * 0.75 },
      MULTI_TURN: { x: W * 0.15, y: H * 0.65 },
      SOCIAL_ENG: { x: W * 0.85, y: H * 0.65 },
    };

    // Seed nodes from attacks
    attacks.slice(-40).forEach((a) => {
      const c = centers[a.type] || { x: W * 0.5, y: H * 0.5 };
      const id = String(a.id);
      if (!nodesRef.current[id]) {
        nodesRef.current[id] = {
          x: c.x + (Math.random() - 0.5) * 50,
          y: c.y + (Math.random() - 0.5) * 50,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          type: a.type,
          r: 3 + a.tier * 0.8,
          color: THREAT_COLORS[a.type] || '#888',
          cx: c.x,
          cy: c.y,
        };
      }
    });

    // Prune stale nodes
    const activeIds = new Set(attacks.slice(-40).map((a) => String(a.id)));
    Object.keys(nodesRef.current).forEach((id) => {
      if (!activeIds.has(id)) delete nodesRef.current[id];
    });

    let animId: number;
    function draw() {
      ctx!.clearRect(0, 0, W, H);
      ctx!.fillStyle = '#030508';
      ctx!.fillRect(0, 0, W, H);

      // Cluster halos
      Object.entries(centers).forEach(([type, c]) => {
        const color = THREAT_COLORS[type] || '#888';
        ctx!.beginPath();
        ctx!.arc(c.x, c.y, 35, 0, Math.PI * 2);
        ctx!.strokeStyle = color + '18';
        ctx!.lineWidth = 1;
        ctx!.stroke();
        ctx!.fillStyle = color + '06';
        ctx!.fill();
      });

      // Update + draw nodes
      Object.values(nodesRef.current).forEach((n) => {
        n.vx += (n.cx - n.x) * 0.003;
        n.vy += (n.cy - n.y) * 0.003;
        n.vx *= 0.96;
        n.vy *= 0.96;
        n.x = Math.max(n.r, Math.min(W - n.r, n.x + n.vx));
        n.y = Math.max(n.r, Math.min(H - n.r, n.y + n.vy));
        ctx!.beginPath();
        ctx!.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx!.fillStyle = n.color + 'cc';
        ctx!.fill();
      });

      animId = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(animId);
  }, [attacks]);

  return (
    <canvas
      ref={canvasRef}
      width={180}
      height={120}
      style={{ display: 'block', borderRadius: 6 }}
    />
  );
}
