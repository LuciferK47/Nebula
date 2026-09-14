import { useState } from 'react';

interface Step {
  token: number;
  text: string;
  latencyMs: number;
  hbm: number;
  dram: number;
  disk: number;
}

export function QwenWarmupChart({ steps }: { steps: Step[] }) {
  if (steps.length === 0) return null;

  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const W = 520;
  const panelH = 72;
  const gap = 24;
  const M = { top: 18, right: 18, bottom: 26, left: 42 };
  const plotW = W - M.left - M.right;
  const H = M.top + panelH + gap + panelH + M.bottom;

  const n = steps.length;
  const slot = plotW / n;
  const xAt = (i: number) => M.left + slot * (i + 0.5);

  const maxLatency = Math.max(...steps.map((s) => s.latencyMs)) * 1.1;
  const yLatTop = M.top;
  const yLat = (v: number) => yLatTop + panelH - (v / maxLatency) * panelH;

  const maxCount = Math.max(...steps.map((s) => s.hbm + s.dram + s.disk), 1);
  const yBarTop = M.top + panelH + gap;

  const latPath = steps.map((s, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i)} ${yLat(s.latencyMs)}`).join(' ');
  const latAreaPath = `${latPath} L ${xAt(steps.length - 1)} ${yLatTop + panelH} L ${xAt(0)} ${yLatTop + panelH} Z`;

  const GAP_PX = 1.5;
  const activeStep = hoveredIdx !== null ? steps[hoveredIdx] : null;

  return (
    <div className="relative">
      {activeStep && (
        <div className="absolute right-2 top-0 flex items-center gap-2.5 rounded bg-ink-panel/90 px-2.5 py-1 font-mono text-10 text-cream/90 shadow border border-ink-line">
          <span className="text-amber font-semibold">Token {activeStep.token}:</span>
          <span>{activeStep.latencyMs.toFixed(0)} ms</span>
          <span className="text-khaki/60">|</span>
          <span className="text-[#f97316]">HBM: {activeStep.hbm}</span>
          <span className="text-[#eab308]">DRAM: {activeStep.dram}</span>
          <span className="text-[#94a3b8]">Disk: {activeStep.disk}</span>
        </div>
      )}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto max-h-[260px] select-none"
        role="img"
        aria-label="Per-token decode latency and tier-hit composition as the cache warms up"
      >
        <defs>
          <linearGradient id="latGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f5b32b" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#f5b32b" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* ---- Panel 1: Latency ---- */}
        <text x={M.left} y={yLatTop - 5} fontSize={8.5} fill="#f4f3ed" opacity={0.65} className="font-mono uppercase tracking-wider">
          Decode Latency (ms)
        </text>
        {[0, 0.5, 1].map((f) => (
          <line
            key={f}
            x1={M.left}
            x2={W - M.right}
            y1={yLatTop + panelH * (1 - f)}
            y2={yLatTop + panelH * (1 - f)}
            stroke="#262624"
            strokeWidth={1}
            strokeDasharray="2 3"
          />
        ))}
        <path d={latAreaPath} fill="url(#latGrad)" />
        <path d={latPath} fill="none" stroke="#f5b32b" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
        {steps.map((s, i) => (
          <circle
            key={s.token}
            cx={xAt(i)}
            cy={yLat(s.latencyMs)}
            r={hoveredIdx === i ? 4.5 : 2.5}
            fill="#f5b32b"
            stroke="#121210"
            strokeWidth={1.5}
            className="transition-all cursor-pointer"
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
          />
        ))}
        <text x={xAt(0) + 4} y={yLat(steps[0].latencyMs) - 6} fontSize={8} fill="#f4f3ed" opacity={0.65} className="font-mono">
          {steps[0].latencyMs.toFixed(0)} ms
        </text>
        <text x={xAt(n - 1) - 4} y={yLat(steps[n - 1].latencyMs) - 6} textAnchor="end" fontSize={8} fill="#f4f3ed" opacity={0.65} className="font-mono">
          {steps[n - 1].latencyMs.toFixed(0)} ms
        </text>

        {/* ---- Panel 2: Tier-hit composition ---- */}
        <text x={M.left} y={yBarTop - 5} fontSize={8.5} fill="#f4f3ed" opacity={0.65} className="font-mono uppercase tracking-wider">
          Lookups By Tier
        </text>

        {/* Inline Compact Legend */}
        <g transform={`translate(${W - M.right - 145}, ${yBarTop - 12})`}>
          {[
            { label: 'HBM', color: '#ea580c' },
            { label: 'DRAM', color: '#eab308' },
            { label: 'Disk', color: '#94a3b8' },
          ].map((item, i) => (
            <g key={item.label} transform={`translate(${i * 50}, 0)`}>
              <rect x={0} y={1} width={6} height={6} rx={1} fill={item.color} />
              <text x={9} y={7} fontSize={8} fill="#f4f3ed" opacity={0.65} className="font-mono">
                {item.label}
              </text>
            </g>
          ))}
        </g>

        {[0, 0.5, 1].map((f) => (
          <line
            key={f}
            x1={M.left}
            x2={W - M.right}
            y1={yBarTop + panelH * (1 - f)}
            y2={yBarTop + panelH * (1 - f)}
            stroke="#262624"
            strokeWidth={1}
            strokeDasharray="2 3"
          />
        ))}

        {steps.map((s, i) => {
          const barW = slot * 0.56;
          const barX = xAt(i) - barW / 2;
          const total = s.hbm + s.dram + s.disk;
          const scale = (v: number) => (total > 0 ? (v / maxCount) * panelH : 0);
          const hbmH = scale(s.hbm);
          const dramH = scale(s.dram);
          const diskH = scale(s.disk);
          let cursor = yBarTop + panelH;
          const segments: { h: number; color: string }[] = [
            { h: hbmH, color: '#ea580c' },
            { h: dramH, color: '#eab308' },
            { h: diskH, color: '#94a3b8' },
          ];

          const isHov = hoveredIdx === i;

          return (
            <g
              key={s.token}
              className="cursor-pointer"
              opacity={hoveredIdx !== null && !isHov ? 0.45 : 0.95}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              {segments.map((seg, si) => {
                if (seg.h <= 0) return null;
                cursor -= seg.h;
                const y = cursor + (si > 0 ? GAP_PX / 2 : 0);
                const h = Math.max(seg.h - (si > 0 ? GAP_PX : 0), 0);
                return <rect key={si} x={barX} y={y} width={barW} height={h} rx={1.5} fill={seg.color} />;
              })}
            </g>
          );
        })}

        {/* Shared X axis ticks */}
        {steps.map((s, i) => (
          <text
            key={s.token}
            x={xAt(i)}
            y={H - 8}
            textAnchor="middle"
            fontSize={8}
            fill="#f4f3ed"
            opacity={hoveredIdx === i ? 0.95 : 0.5}
            className="font-mono"
          >
            t{s.token}
          </text>
        ))}
      </svg>
    </div>
  );
}
