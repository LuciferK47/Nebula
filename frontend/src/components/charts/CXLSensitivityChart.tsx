import { useState } from 'react';
import type { ScenarioRun } from '../../lib/results';

export function CXLSensitivityChart({ runs }: { runs: ScenarioRun[] }) {
  if (runs.length === 0) return null;

  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const bwRuns = runs.filter((r) => r.cxl_emulation_mode === 'full');
  const modeRuns = runs.filter((r) => r.cxl_emulation_mode !== 'full');
  const ordered = [...bwRuns, ...modeRuns];

  const labelFor = (r: ScenarioRun) =>
    r.cxl_emulation_mode === 'full'
      ? `${r.cxl_bandwidth_gbps}G`
      : r.cxl_emulation_mode === 'disabled'
      ? 'No emu'
      : 'Lat only';

  const fullLabelFor = (r: ScenarioRun) =>
    r.cxl_emulation_mode === 'full'
      ? `${r.cxl_bandwidth_gbps} GB/s (Full)`
      : r.cxl_emulation_mode === 'disabled'
      ? 'No Emulation'
      : 'Latency Only';

  const W = 520;
  const panelH = 72;
  const gap = 24;
  const M = { top: 18, right: 18, bottom: 26, left: 42 };
  const plotW = W - M.left - M.right;
  const H = M.top + panelH + gap + panelH + M.bottom;

  const n = ordered.length;
  const slot = plotW / n;
  const xAt = (i: number) => M.left + slot * (i + 0.5);

  const maxTok = Math.max(...ordered.map((r) => r.tokens_per_second)) * 1.15;
  const yTokTop = M.top;
  const yTok = (v: number) => yTokTop + panelH - (v / maxTok) * panelH;

  const yHitTop = M.top + panelH + gap;
  const hitVals = ordered.map((r) => r.hit_rate);
  const hitMid = hitVals.reduce((a, b) => a + b, 0) / hitVals.length;
  const hitSpan = Math.max(0.05, Math.max(...hitVals) - Math.min(...hitVals), 0.02);
  const hitMin = hitMid - hitSpan;
  const hitMax = hitMid + hitSpan;
  const yHit = (v: number) => yHitTop + panelH - ((v - hitMin) / (hitMax - hitMin)) * panelH;

  const isBwPanel = (i: number) => i < bwRuns.length;
  const activeRun = hoveredIdx !== null ? ordered[hoveredIdx] : null;

  return (
    <div className="relative">
      {activeRun && (
        <div className="absolute right-2 top-0 flex items-center gap-3 rounded bg-ink-panel/90 px-2.5 py-1 font-mono text-10 text-cream/90 shadow border border-ink-line">
          <span className="text-amber font-semibold">{fullLabelFor(activeRun)}:</span>
          <span>{activeRun.tokens_per_second.toFixed(2)} tok/s</span>
          <span className="text-khaki/60">|</span>
          <span className="text-purple-300">{(activeRun.hit_rate * 100).toFixed(1)}% hit</span>
        </div>
      )}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto max-h-[260px] select-none"
        role="img"
        aria-label="CXL bandwidth and emulation-mode sensitivity: throughput and hit rate"
      >
        <defs>
          <linearGradient id="cxlBwGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f5b32b" stopOpacity="0.95" />
            <stop offset="100%" stopColor="#d97706" stopOpacity="0.75" />
          </linearGradient>
          <linearGradient id="cxlModeGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#94a3b8" stopOpacity="0.85" />
            <stop offset="100%" stopColor="#64748b" stopOpacity="0.65" />
          </linearGradient>
        </defs>

        {/* Divider between sweep and mode variants */}
        <line
          x1={M.left + slot * bwRuns.length}
          x2={M.left + slot * bwRuns.length}
          y1={M.top - 2}
          y2={yHitTop + panelH}
          stroke="#2d2d2a"
          strokeDasharray="2 3"
        />

        {/* ---- Panel 1: Throughput ---- */}
        <text x={M.left} y={yTokTop - 5} fontSize={8.5} fill="#f4f3ed" opacity={0.65} className="font-mono uppercase tracking-wider">
          Throughput (tok/s)
        </text>
        {[0, 0.5, 1].map((f) => (
          <line
            key={f}
            x1={M.left}
            x2={W - M.right}
            y1={yTokTop + panelH * (1 - f)}
            y2={yTokTop + panelH * (1 - f)}
            stroke="#262624"
            strokeWidth={1}
            strokeDasharray="2 3"
          />
        ))}
        {ordered.map((r, i) => {
          const barW = slot * 0.52;
          const barH = yTokTop + panelH - yTok(r.tokens_per_second);
          const isHov = hoveredIdx === i;
          return (
            <rect
              key={labelFor(r)}
              x={xAt(i) - barW / 2}
              y={yTok(r.tokens_per_second)}
              width={barW}
              height={barH}
              rx={3}
              fill={isBwPanel(i) ? 'url(#cxlBwGrad)' : 'url(#cxlModeGrad)'}
              opacity={hoveredIdx !== null && !isHov ? 0.45 : 0.92}
              className="transition-opacity cursor-pointer"
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
            />
          );
        })}

        {/* ---- Panel 2: Hit Rate (Zoomed) ---- */}
        <text x={M.left} y={yHitTop - 5} fontSize={8.5} fill="#f4f3ed" opacity={0.65} className="font-mono uppercase tracking-wider">
          Hit Rate ({(hitMin * 100).toFixed(0)}–{(hitMax * 100).toFixed(0)}%, Invariant)
        </text>
        {[0, 0.5, 1].map((f) => (
          <line
            key={f}
            x1={M.left}
            x2={W - M.right}
            y1={yHitTop + panelH * (1 - f)}
            y2={yHitTop + panelH * (1 - f)}
            stroke="#262624"
            strokeWidth={1}
            strokeDasharray="2 3"
          />
        ))}
        <path
          d={ordered.map((r, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i)} ${yHit(r.hit_rate)}`).join(' ')}
          fill="none"
          stroke="#a78bfa"
          strokeWidth={2}
          strokeLinecap="round"
        />
        {ordered.map((r, i) => (
          <circle
            key={labelFor(r)}
            cx={xAt(i)}
            cy={yHit(r.hit_rate)}
            r={hoveredIdx === i ? 4.5 : 3}
            fill="#a78bfa"
            stroke="#121210"
            strokeWidth={1.5}
            className="transition-all cursor-pointer"
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
          />
        ))}

        {/* Shared Categorical X-Axis */}
        {ordered.map((r, i) => (
          <text
            key={labelFor(r)}
            x={xAt(i)}
            y={H - 8}
            textAnchor="middle"
            fontSize={8}
            fill="#f4f3ed"
            opacity={hoveredIdx === i ? 0.95 : 0.55}
            className="font-mono"
          >
            {labelFor(r)}
          </text>
        ))}
      </svg>
    </div>
  );
}
