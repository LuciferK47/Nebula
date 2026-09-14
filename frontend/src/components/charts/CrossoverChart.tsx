import type { ScenarioRun } from '../../lib/results';

/** Paired bars for hybrid vs. weight_transfer at one budget. Throughput
 * shares a linear axis; data moved does NOT — 4.6 MB and 14,308 MB cannot
 * share one linear bar without one of them reading as zero, so that panel
 * is log-scaled and says so on the axis. */
export function CrossoverChart({ hybrid, baseline }: { hybrid: ScenarioRun; baseline: ScenarioRun }) {
  const tokMax = Math.max(hybrid.tokens_per_second, baseline.tokens_per_second) * 1.2;
  const bytesLogMax = Math.log10(Math.max(hybrid.transfer_mb, baseline.transfer_mb, 1) * 1.4);
  const bytesLogMin = 0; // 10^0 = 1 MB floor

  const logScale = (mb: number) => {
    const v = Math.max(mb, 0.1);
    return Math.max(0, (Math.log10(v) - bytesLogMin) / (bytesLogMax - bytesLogMin));
  };

  const rows: { label: string; isSubject: boolean; tok: number; mb: number; evictions: number }[] = [
  { label: 'weight_transfer', isSubject: false, tok: baseline.tokens_per_second, mb: baseline.transfer_mb, evictions: baseline.evictions },
  { label: 'hybrid', isSubject: true, tok: hybrid.tokens_per_second, mb: hybrid.transfer_mb, evictions: hybrid.evictions }];


  return (
    <div className="grid gap-10 sm:grid-cols-2">
      <div>
        <p className="font-mono text-10 uppercase tracking-label text-cream/40">Throughput (tok/s)</p>
        <ul className="mt-4 space-y-5">
          {rows.map((r) =>
          <li key={r.label}>
              <div className="flex items-baseline justify-between gap-4">
                <span className={`font-mono text-11 uppercase tracking-wide ${r.isSubject ? 'text-amber' : 'text-cream/60'}`}>
                  {r.label}
                </span>
                <span className={`font-display text-xl ${r.isSubject ? 'text-amber' : 'text-cream/70'}`}>
                  {r.tok.toFixed(2)}
                </span>
              </div>
              <div className="mt-2 h-[10px] w-full bg-ink-panel">
                <div
                className={`h-full ${r.isSubject ? 'bg-amber' : 'bg-khaki/35'}`}
                style={{ width: `${r.tok / tokMax * 100}%` }} />

              </div>
            </li>
          )}
        </ul>
      </div>

      <div>
        <p className="font-mono text-10 uppercase tracking-label text-cream/40">Data moved (MB, log scale)</p>
        <ul className="mt-4 space-y-5">
          {rows.map((r) =>
          <li key={r.label}>
              <div className="flex items-baseline justify-between gap-4">
                <span className={`font-mono text-11 uppercase tracking-wide ${r.isSubject ? 'text-amber' : 'text-cream/60'}`}>
                  {r.label}
                </span>
                <span className={`font-display text-xl ${r.isSubject ? 'text-amber' : 'text-cream/70'}`}>
                  {r.mb.toLocaleString(undefined, { maximumFractionDigits: 1 })} MB
                </span>
              </div>
              <div className="mt-2 h-[10px] w-full bg-ink-panel">
                <div
                className={`h-full ${r.isSubject ? 'bg-amber' : 'bg-khaki/35'}`}
                style={{ width: `${Math.max(logScale(r.mb) * 100, 1.5)}%` }} />

              </div>
              <p className="mt-1 font-mono text-10 text-cream/35">{r.evictions} evictions</p>
            </li>
          )}
        </ul>
      </div>
    </div>);

}
