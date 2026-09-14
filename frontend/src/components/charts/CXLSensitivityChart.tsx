import type { ScenarioRun } from '../../lib/results';

/**
 * The paper-defense result: hit rate is IDENTICAL across every bandwidth
 * assumption and every emulation mode (disabled / latency-only / full) —
 * proof that the CXL emulation's calibration cannot be silently steering
 * placement decisions. Two single-axis panels, not one dual-axis chart:
 * throughput (which does vary with bandwidth) on top, hit rate (which
 * provably doesn't) on the bottom, sharing one categorical x-axis of run
 * labels rather than a shared numeric scale, since the two panels are
 * different sweeps (5 bandwidth points, plus 2 emulation-mode points at a
 * fixed bandwidth).
 */
export function CXLSensitivityChart({ runs }: { runs: ScenarioRun[] }) {
  if (runs.length === 0) return null;

  const bwRuns = runs.filter((r) => r.cxl_emulation_mode === 'full');
  const modeRuns = runs.filter((r) => r.cxl_emulation_mode !== 'full');
  const ordered = [...bwRuns, ...modeRuns];

  const labelFor = (r: ScenarioRun) =>
  r.cxl_emulation_mode === 'full' ?
  `${r.cxl_bandwidth_gbps} GB/s` :
  r.cxl_emulation_mode === 'disabled' ?
  'no emulation' :
  'latency only';

  const W = 640;
  const panelH = 130;
  const gap = 30;
  const M = { top: 14, right: 20, bottom: 34, left: 44 };
  const plotW = W - M.left - M.right;
  const H = M.top + panelH + gap + panelH + M.bottom;

  const n = ordered.length;
  const slot = plotW / n;
  const xAt = (i: number) => M.left + slot * (i + 0.5);

  const maxTok = Math.max(...ordered.map((r) => r.tokens_per_second)) * 1.15;
  const yTokTop = M.top;
  const yTok = (v: number) => yTokTop + panelH - (v / maxTok) * panelH;

  const yHitTop = M.top + panelH + gap;
  // Hit rate is ~constant near 0.4; zoom the panel around the actual
  // values instead of 0-100% so the (deliberate) flatness is visible as a
  // flat line rather than a flat line squashed to a hairline at the very
  // top of a 0-1 axis.
  const hitVals = ordered.map((r) => r.hit_rate);
  const hitMid = hitVals.reduce((a, b) => a + b, 0) / hitVals.length;
  const hitSpan = Math.max(0.05, Math.max(...hitVals) - Math.min(...hitVals), 0.02);
  const hitMin = hitMid - hitSpan;
  const hitMax = hitMid + hitSpan;
  const yHit = (v: number) => yHitTop + panelH - ((v - hitMin) / (hitMax - hitMin)) * panelH;

  const isBwPanel = (i: number) => i < bwRuns.length;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="CXL bandwidth and emulation-mode sensitivity: throughput and hit rate">
      <desc>
        Top panel: throughput across a 4-64 GB/s CXL bandwidth sweep, plus two emulation-mode
        variants at a fixed bandwidth. Bottom panel: hit rate for the same runs, zoomed to show it
        is constant regardless of the bandwidth assumption or whether emulation is enabled at all.
      </desc>

      {/* divider between the bandwidth sweep and the mode variants */}
      <line
        x1={M.left + slot * bwRuns.length}
        x2={M.left + slot * bwRuns.length}
        y1={M.top}
        y2={yHitTop + panelH}
        stroke="#32322c"
        strokeDasharray="2 3" />

      {/* ---- Panel 1: throughput ---- */}
      <text x={M.left} y={yTokTop - 4} fontSize={9} fill="#f4f3ed" opacity={0.5} className="font-mono">
        tokens / sec
      </text>
      {[0, 0.5, 1].map((f) =>
      <line key={f} x1={M.left} x2={W - M.right} y1={yTokTop + panelH * (1 - f)} y2={yTokTop + panelH * (1 - f)} stroke="#32322c" strokeOpacity={0.5} strokeWidth={1} />
      )}
      {ordered.map((r, i) =>
      <rect
        key={labelFor(r)}
        x={xAt(i) - slot * 0.28}
        y={yTok(r.tokens_per_second)}
        width={slot * 0.56}
        height={yTokTop + panelH - yTok(r.tokens_per_second)}
        fill={isBwPanel(i) ? '#f5b32b' : '#8fa3b8'}
        opacity={0.9} />

      )}

      {/* ---- Panel 2: hit rate, zoomed ---- */}
      <text x={M.left} y={yHitTop - 4} fontSize={9} fill="#f4f3ed" opacity={0.5} className="font-mono">
        hit rate ({(hitMin * 100).toFixed(0)}–{(hitMax * 100).toFixed(0)}%, zoomed)
      </text>
      {[0, 0.5, 1].map((f) =>
      <line key={f} x1={M.left} x2={W - M.right} y1={yHitTop + panelH * (1 - f)} y2={yHitTop + panelH * (1 - f)} stroke="#32322c" strokeOpacity={0.5} strokeWidth={1} />
      )}
      <path
        d={ordered.map((r, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i)} ${yHit(r.hit_rate)}`).join(' ')}
        fill="none"
        stroke="#7c6bd6"
        strokeWidth={2} />

      {ordered.map((r, i) =>
      <circle key={labelFor(r)} cx={xAt(i)} cy={yHit(r.hit_rate)} r={3.5} fill="#7c6bd6" />
      )}

      {/* shared categorical x-axis */}
      {ordered.map((r, i) =>
      <text
        key={labelFor(r)}
        x={xAt(i)}
        y={H - 10}
        textAnchor="middle"
        fontSize={8.5}
        fill="#f4f3ed"
        opacity={0.55}
        className="font-mono">

          {labelFor(r)}
        </text>
      )}
    </svg>);

}
