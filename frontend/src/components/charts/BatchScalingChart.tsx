import { useState } from 'react';
import type { S7Run } from '../../lib/results';

export function BatchScalingChart({ runs }: { runs: S7Run[] }) {
  if (runs.length === 0) return null;

  const [hoveredBatch, setHoveredBatch] = useState<number | null>(null);

  const batches = [1, 2, 4, 8];
  const batchData = batches.map((b) => {
    const hybrid = runs.find((r) => r.batch_size === b && r.execution_mode === 'hybrid');
    const wt = runs.find((r) => r.batch_size === b && r.execution_mode === 'weight_transfer');
    return {
      batch: b,
      hybrid: hybrid ?? { tokens_per_second: 0, hit_rate: 0, transfer_mb: 0, evictions: 0 },
      wt: wt ?? { tokens_per_second: 0, hit_rate: 0, transfer_mb: 0, evictions: 0 },
      speedup:
        hybrid && wt && wt.tokens_per_second > 0
          ? hybrid.tokens_per_second / wt.tokens_per_second
          : 1,
    };
  });

  const W = 520;
  const panelH = 72;
  const gap = 24;
  const M = { top: 18, right: 18, bottom: 26, left: 42 };
  const plotW = W - M.left - M.right;
  const H = M.top + panelH + gap + panelH + M.bottom;

  const slot = plotW / batches.length;
  const xAt = (i: number) => M.left + slot * (i + 0.5);

  const maxTok = 30; // max ~26 tok/s
  const yTokTop = M.top;
  const yTok = (v: number) => yTokTop + panelH - (v / maxTok) * panelH;

  const yHitTop = M.top + panelH + gap;
  const yHit = (v: number) => yHitTop + panelH - v * panelH;

  const active = hoveredBatch !== null ? batchData.find((d) => d.batch === hoveredBatch) : null;

  return (
    <div className="relative">
      {active && (
        <div className="absolute right-2 top-0 flex items-center gap-3 rounded bg-ink-panel/95 px-3 py-1 font-mono text-xs text-cream shadow-md border border-ink-line">
          <span className="text-amber font-semibold">Batch {active.batch}:</span>
          <span>
            Hybrid: {active.hybrid.tokens_per_second.toFixed(1)} tok/s ({(active.hybrid.hit_rate * 100).toFixed(0)}% hit)
          </span>
          <span className="text-khaki/50">|</span>
          <span className="text-rose-400">
            WT: {active.wt.tokens_per_second.toFixed(1)} tok/s ({(active.wt.hit_rate * 100).toFixed(0)}% hit, {active.wt.evictions} evict)
          </span>
        </div>
      )}

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto max-h-[260px] select-none"
        role="img"
        aria-label="Batch scaling comparison between Hybrid and Weight-Transfer modes"
      >
        <defs>
          <linearGradient id="batchHybridGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f5b32b" stopOpacity="0.95" />
            <stop offset="100%" stopColor="#d97706" stopOpacity="0.8" />
          </linearGradient>
          <linearGradient id="batchWtGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#94a3b8" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#64748b" stopOpacity="0.6" />
          </linearGradient>
        </defs>

        {/* ---- Panel 1: Throughput ---- */}
        <text x={M.left} y={yTokTop - 5} fontSize={9} fill="#f4f3ed" opacity={0.75} className="font-mono uppercase tracking-wider font-semibold">
          Aggregate Throughput (tok/s)
        </text>

        {/* Inline Legend */}
        <g transform={`translate(${W - M.right - 150}, ${yTokTop - 12})`}>
          <rect x={0} y={1} width={7} height={7} rx={1.5} fill="#f5b32b" />
          <text x={11} y={8} fontSize={8.5} fill="#f4f3ed" opacity={0.8} className="font-mono">
            Hybrid
          </text>
          <rect x={58} y={1} width={7} height={7} rx={1.5} fill="#64748b" />
          <text x={69} y={8} fontSize={8.5} fill="#f4f3ed" opacity={0.8} className="font-mono">
            Weight-Transfer
          </text>
        </g>

        {[0, 0.5, 1].map((f) => (
          <line
            key={f}
            x1={M.left}
            x2={W - M.right}
            y1={yTokTop + panelH * (1 - f)}
            y2={yTokTop + panelH * (1 - f)}
            stroke="#2d2d2a"
            strokeWidth={1}
            strokeDasharray="2 3"
          />
        ))}

        {batchData.map((d, i) => {
          const cx = xAt(i);
          const barW = slot * 0.28;
          const hBarH = yTokTop + panelH - yTok(d.hybrid.tokens_per_second);
          const wtBarH = yTokTop + panelH - yTok(d.wt.tokens_per_second);
          const isHov = hoveredBatch === d.batch;

          return (
            <g
              key={d.batch}
              className="cursor-pointer"
              onMouseEnter={() => setHoveredBatch(d.batch)}
              onMouseLeave={() => setHoveredBatch(null)}
              opacity={hoveredBatch !== null && !isHov ? 0.45 : 1}
            >
              {/* Hybrid Bar */}
              <rect
                x={cx - barW - 2}
                y={yTok(d.hybrid.tokens_per_second)}
                width={barW}
                height={hBarH}
                rx={2.5}
                fill="url(#batchHybridGrad)"
              />
              {/* Weight Transfer Bar */}
              <rect
                x={cx + 2}
                y={yTok(d.wt.tokens_per_second)}
                width={barW}
                height={wtBarH}
                rx={2.5}
                fill="url(#batchWtGrad)"
              />
              {/* Speedup Badge */}
              <text
                x={cx}
                y={Math.min(yTok(d.hybrid.tokens_per_second), yTok(d.wt.tokens_per_second)) - 5}
                textAnchor="middle"
                fontSize={8}
                fill="#f5b32b"
                fontWeight="bold"
                className="font-mono"
              >
                {d.speedup.toFixed(1)}×
              </text>
            </g>
          );
        })}

        {/* ---- Panel 2: Hit Rate & Thrashing Collapse ---- */}
        <text x={M.left} y={yHitTop - 5} fontSize={9} fill="#f4f3ed" opacity={0.75} className="font-mono uppercase tracking-wider font-semibold">
          Cache Hit Rate & Thrashing (0–50%)
        </text>

        {[0, 0.5, 1].map((f) => (
          <line
            key={f}
            x1={M.left}
            x2={W - M.right}
            y1={yHitTop + panelH * (1 - f * 0.5)}
            y2={yHitTop + panelH * (1 - f * 0.5)}
            stroke="#2d2d2a"
            strokeWidth={1}
            strokeDasharray="2 3"
          />
        ))}

        {/* Hybrid Line (Stable ~39%) */}
        <path
          d={batchData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i)} ${yHit(d.hybrid.hit_rate)}`).join(' ')}
          fill="none"
          stroke="#f5b32b"
          strokeWidth={2}
          strokeLinecap="round"
        />

        {/* Weight Transfer Line (Collapses 35% -> 0.5%) */}
        <path
          d={batchData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i)} ${yHit(d.wt.hit_rate)}`).join(' ')}
          fill="none"
          stroke="#f43f5e"
          strokeWidth={2}
          strokeDasharray="3 3"
          strokeLinecap="round"
        />

        {batchData.map((d, i) => (
          <g key={d.batch}>
            {/* Hybrid dot */}
            <circle
              cx={xAt(i)}
              cy={yHit(d.hybrid.hit_rate)}
              r={hoveredBatch === d.batch ? 4.5 : 3}
              fill="#f5b32b"
              stroke="#121210"
              strokeWidth={1.5}
            />
            {/* WT dot */}
            <circle
              cx={xAt(i)}
              cy={yHit(d.wt.hit_rate)}
              r={hoveredBatch === d.batch ? 4.5 : 3}
              fill="#f43f5e"
              stroke="#121210"
              strokeWidth={1.5}
            />
          </g>
        ))}

        {/* Thrash callout on B=8 */}
        <text
          x={xAt(3)}
          y={yHit(batchData[3].wt.hit_rate) - 6}
          textAnchor="middle"
          fontSize={7.5}
          fill="#f43f5e"
          fontWeight="bold"
          className="font-mono"
        >
          0.5% (1,375 evictions)
        </text>

        {/* Shared X-Axis */}
        {batchData.map((d, i) => (
          <text
            key={d.batch}
            x={xAt(i)}
            y={H - 8}
            textAnchor="middle"
            fontSize={9}
            fill="#f4f3ed"
            opacity={hoveredBatch === d.batch ? 1 : 0.65}
            className="font-mono"
          >
            Batch = {d.batch}
          </text>
        ))}
      </svg>
    </div>
  );
}
