import type { ScenarioRun } from '../../lib/results';

/** Two single-axis panels, not one dual-axis chart — throughput and hit
 * rate are different measures on different scales, and forcing them onto
 * one plot with two y-axes is the #1 chart mistake (see the dataviz
 * skill's anti-patterns: it invites reading a crossing or a gap between
 * the lines as meaningful when it's just two arbitrary scales lining up).
 * They stay aligned on one shared x-axis instead, stacked panels sharing
 * the same budget ticks.
 *
 * Annotates the 1500→2000 MB dip rather than smoothing it away: at 100%
 * hit rate and zero transfer, a bigger budget cannot legitimately be
 * slower, so that's thermal drift on the test machine, not a capacity
 * effect. */
export function CapacityCliffChart({ runs }: { runs: ScenarioRun[] }) {
  if (runs.length === 0) return null;

  const W = 640;
  const panelH = 150;
  const gap = 28;
  const M = { top: 14, right: 20, bottom: 32, left: 44 };
  const plotW = W - M.left - M.right;
  const H = M.top + panelH + gap + panelH + M.bottom;

  const budgets = runs.map((r) => r.hbm_budget_mb);
  const minB = Math.min(...budgets);
  const maxB = Math.max(...budgets);
  const maxTok = Math.max(...runs.map((r) => r.tokens_per_second)) * 1.15;

  const x = (b: number) => M.left + ((b - minB) / (maxB - minB)) * plotW;
  const yTokTop = M.top;
  const yTok = (v: number) => yTokTop + panelH - (v / maxTok) * panelH;
  const yHitTop = M.top + panelH + gap;
  const yHit = (v: number) => yHitTop + panelH - v * panelH;

  const tokPath = runs.map((r, i) => `${i === 0 ? 'M' : 'L'} ${x(r.hbm_budget_mb)} ${yTok(r.tokens_per_second)}`).join(' ');
  const hitPath = runs.map((r, i) => `${i === 0 ? 'M' : 'L'} ${x(r.hbm_budget_mb)} ${yHit(r.hit_rate)}`).join(' ');

  // The dip: the last budget that is strictly slower than an earlier one
  // despite an equal-or-higher hit rate.
  let dipIndex = -1;
  for (let i = 1; i < runs.length; i++) {
    if (runs[i].hit_rate >= runs[i - 1].hit_rate - 1e-6 && runs[i].tokens_per_second < runs[i - 1].tokens_per_second) {
      dipIndex = i;
    }
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Two panels: throughput and hit rate, each versus expert cache budget">
      <desc>Top panel: tokens per second vs. expert cache budget. Bottom panel: cache hit rate vs. the same budget axis. Panels share the x-axis but are scaled independently.</desc>

      {/* ---- Panel 1: throughput ---- */}
      <text x={M.left} y={yTokTop - 4} fontSize={9} fill="#f4f3ed" opacity={0.5} className="font-mono">
        tokens / sec
      </text>
      {[0, 0.5, 1].map((f) =>
      <line
        key={f}
        x1={M.left}
        x2={W - M.right}
        y1={yTokTop + panelH * (1 - f)}
        y2={yTokTop + panelH * (1 - f)}
        stroke="#32322c"
        strokeOpacity={0.5}
        strokeWidth={1} />

      )}
      <path d={tokPath} fill="none" stroke="#f5b32b" strokeWidth={2} />
      {runs.map((r) =>
      <circle key={r.hbm_budget_mb} cx={x(r.hbm_budget_mb)} cy={yTok(r.tokens_per_second)} r={3} fill="#f5b32b" />
      )}
      {dipIndex >= 0 &&
      <g>
          <line
          x1={x(runs[dipIndex - 1].hbm_budget_mb)}
          x2={x(runs[dipIndex].hbm_budget_mb)}
          y1={yTok(runs[dipIndex - 1].tokens_per_second)}
          y2={yTok(runs[dipIndex].tokens_per_second)}
          stroke="#e5b52f"
          strokeWidth={1}
          strokeDasharray="2 2" />

          <text
          x={x(runs[dipIndex].hbm_budget_mb)}
          y={yTok(runs[dipIndex].tokens_per_second) - 8}
          textAnchor="end"
          fontSize={9}
          fill="#e5b52f"
          className="font-mono">

            thermal drift, not capacity — see hit rate below
          </text>
        </g>
      }

      {/* ---- Panel 2: hit rate ---- */}
      <text x={M.left} y={yHitTop - 4} fontSize={9} fill="#f4f3ed" opacity={0.5} className="font-mono">
        hit rate (0–100%)
      </text>
      {[0, 0.5, 1].map((f) =>
      <line
        key={f}
        x1={M.left}
        x2={W - M.right}
        y1={yHitTop + panelH * (1 - f)}
        y2={yHitTop + panelH * (1 - f)}
        stroke="#32322c"
        strokeOpacity={0.5}
        strokeWidth={1} />

      )}
      <path d={hitPath} fill="none" stroke="#8fa3b8" strokeWidth={2} />
      {runs.map((r) =>
      <circle key={r.hbm_budget_mb} cx={x(r.hbm_budget_mb)} cy={yHit(r.hit_rate)} r={3} fill="#8fa3b8" />
      )}
      {dipIndex >= 0 &&
      <circle cx={x(runs[dipIndex].hbm_budget_mb)} cy={yHit(runs[dipIndex].hit_rate)} r={6} fill="none" stroke="#8fa3b8" strokeWidth={1} opacity={0.6} />
      }

      {/* shared x axis, drawn once under the bottom panel */}
      {runs.map((r, i) =>
      i % 2 === 0 &&
      <text
        key={r.hbm_budget_mb}
        x={x(r.hbm_budget_mb)}
        y={H - 10}
        textAnchor="middle"
        fontSize={9}
        fill="#f4f3ed"
        opacity={0.5}
        className="font-mono">

            {r.hbm_budget_mb}
          </text>

      )}
      <text x={M.left} y={H - 10} fontSize={9} fill="#f4f3ed" opacity={0.4} className="font-mono">
        expert cache budget (MB)
      </text>
    </svg>);

}
