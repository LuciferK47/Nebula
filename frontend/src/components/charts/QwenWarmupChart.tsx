/**
 * The real-MoE cache warm-up story: per-token decode latency on
 * Qwen1.5-MoE-A2.7B collapses as the tier hierarchy fills. Two panels
 * (latency, then tier-hit composition) rather than one dual-axis chart —
 * milliseconds and hit counts are unrelated scales.
 *
 * hbm_hits/dram_hits in the source file are cumulative; disk_fetches is
 * already per-token. src/lib/results.ts's qwenPerTokenDeltas() is the one
 * place that inconsistency gets normalized — this component only ever
 * reads the normalized per-token deltas, never the raw arrays.
 */
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

  const W = 640;
  const panelH = 130;
  const gap = 30;
  const M = { top: 14, right: 20, bottom: 28, left: 50 };
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

  const GAP_PX = 2; // surface gap between stacked segments

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Per-token decode latency and tier-hit composition as the cache warms up">
      <desc>
        Top panel: decode latency per generated token, collapsing from {steps[0].latencyMs.toFixed(0)} ms
        to {steps[n - 1].latencyMs.toFixed(0)} ms as the cache warms. Bottom panel: how many of that
        token's expert lookups hit HBM, DRAM, or fell through to disk.
      </desc>

      {/* ---- Panel 1: latency ---- */}
      <text x={M.left} y={yLatTop - 4} fontSize={9} fill="#f4f3ed" opacity={0.5} className="font-mono">
        decode latency (ms)
      </text>
      {[0, 0.5, 1].map((f) =>
      <line key={f} x1={M.left} x2={W - M.right} y1={yLatTop + panelH * (1 - f)} y2={yLatTop + panelH * (1 - f)} stroke="#32322c" strokeOpacity={0.5} strokeWidth={1} />
      )}
      <path
        d={steps.map((s, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i)} ${yLat(s.latencyMs)}`).join(' ')}
        fill="none"
        stroke="#f5b32b"
        strokeWidth={2} />

      {steps.map((s, i) =>
      <circle key={s.token} cx={xAt(i)} cy={yLat(s.latencyMs)} r={3} fill="#f5b32b" />
      )}
      <text x={xAt(0)} y={yLat(steps[0].latencyMs) - 8} textAnchor="start" fontSize={9} fill="#f4f3ed" opacity={0.7} className="font-mono">
        {steps[0].latencyMs.toFixed(0)} ms
      </text>
      <text x={xAt(n - 1)} y={yLat(steps[n - 1].latencyMs) - 8} textAnchor="end" fontSize={9} fill="#f4f3ed" opacity={0.7} className="font-mono">
        {steps[n - 1].latencyMs.toFixed(0)} ms
      </text>

      {/* ---- Panel 2: tier-hit composition, stacked ---- */}
      <text x={M.left} y={yBarTop - 4} fontSize={9} fill="#f4f3ed" opacity={0.5} className="font-mono">
        expert lookups by tier
      </text>
      {steps.map((s, i) => {
        const barW = slot * 0.62;
        const barX = xAt(i) - barW / 2;
        const total = s.hbm + s.dram + s.disk;
        const scale = (v: number) => total > 0 ? v / maxCount * panelH : 0;
        const hbmH = scale(s.hbm);
        const dramH = scale(s.dram);
        const diskH = scale(s.disk);
        let cursor = yBarTop + panelH;
        const segments: { h: number; color: string }[] = [
        { h: hbmH, color: '#e4512b' },
        { h: dramH, color: '#e5b52f' },
        { h: diskH, color: '#8fa3b8' }];

        return (
          <g key={s.token}>
            {segments.map((seg, si) => {
              if (seg.h <= 0) return null;
              cursor -= seg.h;
              const y = cursor + (si > 0 ? GAP_PX / 2 : 0);
              const h = Math.max(seg.h - (si > 0 ? GAP_PX : 0), 0);
              return <rect key={si} x={barX} y={y} width={barW} height={h} fill={seg.color} />;
            })}
          </g>);

      })}

      {/* shared x-axis: token index */}
      {steps.map((s, i) =>
      <text key={s.token} x={xAt(i)} y={H - 8} textAnchor="middle" fontSize={8.5} fill="#f4f3ed" opacity={0.5} className="font-mono">
          t{s.token}
        </text>
      )}

      {/* legend, right-aligned on the same row as the panel title so it
         never collides with the "expert lookups by tier" label on the left */}
      <g transform={`translate(${W - M.right - 3 * 62}, ${yBarTop - 11})`}>
        {[{ label: 'HBM', color: '#e4512b' }, { label: 'DRAM', color: '#e5b52f' }, { label: 'disk', color: '#8fa3b8' }].map((item, i) =>
        <g key={item.label} transform={`translate(${i * 62}, 0)`}>
            <rect x={0} y={0} width={7} height={7} fill={item.color} />
            <text x={11} y={6.5} fontSize={8.5} fill="#f4f3ed" opacity={0.6} className="font-mono">
              {item.label}
            </text>
          </g>
        )}
      </g>
    </svg>);

}
