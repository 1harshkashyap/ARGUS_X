import { memo } from 'react';

interface SparklineProps {
  data: number[];
  color?: string;
  height?: number;
}

function SparklineInner({ data, color = '#00e5ff', height = 36 }: SparklineProps) {
  if (data.length < 2) return null;
  const max = Math.max(...data, 1);
  const min = Math.min(...data);
  const w = 140;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = height - ((v - min) / (max - min + 1)) * (height - 4) - 2;
    return `${x},${y}`;
  });
  return (
    <svg width={w} height={height} viewBox={`0 0 ${w} ${height}`} style={{ display: 'block' }}>
      <polyline
        points={pts.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <polyline
        points={`0,${height} ${pts.join(' ')} ${w},${height}`}
        fill={`${color}15`}
        stroke="none"
      />
    </svg>
  );
}

export const Sparkline = memo(SparklineInner);
