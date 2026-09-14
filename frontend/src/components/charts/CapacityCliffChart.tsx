import { useState } from 'react';
import type { ScenarioRun } from '../../lib/results';

export function CapacityCliffChart({ runs }: { runs: ScenarioRun[] }) {
  if (runs.length === 0) return null;

  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const W = 520;
  const panelH = 72;
  const gap = 24;
  const M = { top: 18, right: 18, bottom: 26, left: 42 };
  const plotW = W - M.left - M.right;
  const H = M.top + panelH + gap + panelH + M.bottom;

  const budgets = runs.map((r) => r.hbm_budget_mb);
  const minB = Math.min(...budgets);
  const maxB = Math.max(...budgets);
  const maxTok = Math.max(...runs.map((r) => r.tokens_per_second)) * 1.12;

  const x = (b: number) => M.left + ((b - minB) / (maxB - minB)) * plotW;
  const yTokTop = M.top;
  const yTok = (v: number) => yTokTop + panelH - (v / maxTok) * panelH;
  const yHitTop = M.top + panelH + gap;
  const yHit = (v: number) => yHitTop + panelH - v * panelH;

  const tokPath = runs.map((r, i) => `${i === 0 ? 'M' : 'L'} ${x(r.hbm_budget_mb)} ${yTok(r.tokens_per_second)}`).join(' ');
  const tokAreaPath = `${tokPath} L ${x(runs[runs.length - 1].hbm_budget_mb)} ${yTokTop + panelH} L ${x(runs[0].hbm_budget_mb)} ${yTokTop + panelH} Z`;

  const hitPath = runs.map((r, i) => `${i === 0 ? 'M' : 'L'} ${x(r.hbm_budget_mb)} ${yHit(r.hit_rate)}`).join(' ');
  const hitAreaPath = `${hitPath} L ${x(runs[runs.length - 1].hbm_budget_mb)} ${yHitTop + panelH} L ${x(runs[0].hbm_budget_mb)} ${yHitTop + panelH} Z`;

  let dipIndex = -1;
  for (let i = 1; i < runs.length; i++) {
    if (runs[i].hit_rate >= runs[i - 1].hit_rate - 1e-6 && runs[i].tokens_per_second < runs[i - 1].tokens_per_second) {
      dipIndex = i;
    }
  }

  const activeRun = hoveredIdx !== null ? runs[hoveredIdx] : null;

  return (
    <div className="relative">
      {activeRun && (
        <div className="absolute right-2 top-0 flex items-center gap-3 rounded bg-ink-panel/90 px-2.5 py-1 font-mono text-10 text-cream/90 shadow border border-ink-line">
          <span className="text-amber font-semibold">{activeRun.hbm_budget_mb} MB:</span>
          <span>{activeRun.tokens_per_second.toFixed(2)} tok/s</span>
          <span className="text-khaki/60">|</span>
          <span className="text-slate-300">{(activeRun.hit_rate * 100).toFixed(1)}% hit</span>
        </div>
      )}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto max-h-[260px] select-none"
        role="img"
        aria-label="Two compact panels: throughput and hit rate versus expert cache budget"
      >
        <defs>
          <linearGradient id="tokGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f5b32b" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#f5b32b" stopOpacity="0.0" />
          </linearGradient>
          <linearGradient id="hitGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#8fa3b8" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#8fa3b8" stopOpacity="0.0" />
          </linearGradient>
        </defs>

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
        <path d={tokAreaPath} fill="url(#tokGrad)" />
        <path d={tokPath} fill="none" stroke="#f5b32b" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
        {runs.map((r, i) => (
          <circle
            key={r.hbm_budget_mb}
            cx={x(r.hbm_budget_mb)}
            cy={yTok(r.tokens_per_second)}
            r={hoveredIdx === i ? 4.5 : 2.5}
            fill="#f5b32b"
            stroke="#121210"
            strokeWidth={1.5}
            className="transition-all cursor-pointer"
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
          />
        ))}

        {dipIndex >= 0 && (
          <g>
            <line
              x1={x(runs[dipIndex].hbm_budget_mb)}
              x2={x(runs[dipIndex].hbm_budget_mb)}
              y1={yTok(runs[dipIndex].tokens_per_second)}
              y2={yTok(runs[dipIndex].tokens_per_second) - 10}
              stroke="#e5b52f"
              strokeWidth={1}
              strokeDasharray="2 2"
            />
            <text
              x={x(runs[dipIndex].hbm_budget_mb)}
              y={yTok(runs[dipIndex].tokens_per_second) - 12}
              textAnchor="end"
              fontSize={8}
              fill="#e5b52f"
              opacity={0.85}
              className="font-mono"
            >
              thermal drift
            </text>
          </g>
        )}

        {/* ---- Panel 2: Hit rate ---- */}
        <text x={M.left} y={yHitTop - 5} fontSize={8.5} fill="#f4f3ed" opacity={0.65} className="font-mono uppercase tracking-wider">
          Hit Rate (0–100%)
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
        <path d={hitAreaPath} fill="url(#hitGrad)" />
        <path d={hitPath} fill="none" stroke="#8fa3b8" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
        {runs.map((r, i) => (
          <circle
            key={r.hbm_budget_mb}
            cx={x(r.hbm_budget_mb)}
            cy={yHit(r.hit_rate)}
            r={hoveredIdx === i ? 4.5 : 2.5}
            fill="#8fa3b8"
            stroke="#121210"
            strokeWidth={1.5}
            className="transition-all cursor-pointer"
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
          />
        ))}

        {/* Shared X axis ticks */}
        {runs.map((r, i) => (
          (i % 2 === 0 || i === runs.length - 1) && (
            <text
              key={r.hbm_budget_mb}
              x={x(r.hbm_budget_mb)}
              y={H - 8}
              textAnchor="middle"
              fontSize={8}
              fill="#f4f3ed"
              opacity={0.5}
              className="font-mono"
            >
              {r.hbm_budget_mb}M
            </text>
          )
        ))}
        <text x={M.left} y={H - 8} fontSize={8} fill="#f4f3ed" opacity={0.35} className="font-mono">
          Budget
        </text>
      </svg>
    </div>
  );
}
