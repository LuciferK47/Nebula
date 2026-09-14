import { useState } from 'react';
import type { ScenarioRun } from '../../lib/results';

interface CrossoverData {
  hybrid: ScenarioRun;
  baseline: ScenarioRun;
  speedup: number;
  byteRatio: number;
}

interface CrossoverChartProps {
  hybrid?: ScenarioRun;
  baseline?: ScenarioRun;
  crossover600?: CrossoverData | null;
  crossover900?: CrossoverData | null;
}

export function CrossoverChart({ hybrid, baseline, crossover600, crossover900 }: CrossoverChartProps) {
  const [selectedBudget, setSelectedBudget] = useState<600 | 900>(600);

  const activeData: CrossoverData | null =
    crossover600 && crossover900
      ? selectedBudget === 600
        ? crossover600
        : crossover900
      : hybrid && baseline
      ? {
          hybrid,
          baseline,
          speedup: hybrid.tokens_per_second / Math.max(baseline.tokens_per_second, 0.001),
          byteRatio: baseline.transfer_mb / Math.max(hybrid.transfer_mb, 0.001),
        }
      : null;

  if (!activeData) return null;

  const { hybrid: h, baseline: b, speedup, byteRatio } = activeData;

  const tokMax = Math.max(h.tokens_per_second, b.tokens_per_second) * 1.15;
  const tokHWidth = Math.min(100, Math.max(8, (h.tokens_per_second / tokMax) * 100));
  const tokBWidth = Math.min(100, Math.max(8, (b.tokens_per_second / tokMax) * 100));

  // Log scale for data moved (4.6 MB vs 14,308 MB)
  const maxMb = Math.max(h.transfer_mb, b.transfer_mb, 1);
  const logMax = Math.log10(maxMb * 1.25);
  const logMin = 0; // 1 MB
  const toLogW = (mb: number) => {
    const val = Math.max(mb, 0.5);
    const frac = (Math.log10(val) - logMin) / (logMax - logMin);
    return Math.min(100, Math.max(6, frac * 100));
  };

  const mbHWidth = toLogW(h.transfer_mb);
  const mbBWidth = toLogW(b.transfer_mb);

  return (
    <div className="flex flex-col justify-between h-full space-y-4">
      {/* Budget Selector Tabs if both 600 & 900 available */}
      {crossover600 && crossover900 && (
        <div className="flex items-center justify-between gap-2 border-b border-ink-line/60 pb-2.5">
          <span className="font-mono text-10 uppercase tracking-label text-cream/40">HBM Budget:</span>
          <div className="flex items-center gap-1 rounded bg-ink-panel p-0.5 border border-ink-line/50">
            <button
              type="button"
              onClick={() => setSelectedBudget(600)}
              className={`rounded px-2 py-0.5 font-mono text-10 transition-all ${
                selectedBudget === 600
                  ? 'bg-amber text-ink font-semibold shadow-sm'
                  : 'text-cream/60 hover:text-cream hover:bg-ink-line/40'
              }`}
            >
              600 MB (Hard limit)
            </button>
            <button
              type="button"
              onClick={() => setSelectedBudget(900)}
              className={`rounded px-2 py-0.5 font-mono text-10 transition-all ${
                selectedBudget === 900
                  ? 'bg-amber text-ink font-semibold shadow-sm'
                  : 'text-cream/60 hover:text-cream hover:bg-ink-line/40'
              }`}
            >
              900 MB (Relaxed)
            </button>
          </div>
          <span className="font-mono text-10 font-medium px-2 py-0.5 rounded bg-amber/15 text-amber border border-amber/30">
            {speedup.toFixed(2)}× speedup
          </span>
        </div>
      )}

      {/* Row 1: Throughput Comparison */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-mono text-10 uppercase tracking-label text-cream/50">
            Throughput (Decode Tokens / Sec)
          </span>
          <span className="font-mono text-10 text-amber font-medium">
            +{((speedup - 1) * 100).toFixed(0)}% faster
          </span>
        </div>

        {/* Hybrid bar */}
        <div className="space-y-1">
          <div className="flex items-center justify-between font-mono text-11">
            <span className="text-amber font-medium flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-amber inline-block" />
              Hybrid (Nebula)
            </span>
            <span className="text-amber font-display font-semibold">{h.tokens_per_second.toFixed(2)} tok/s</span>
          </div>
          <div className="h-2 w-full rounded bg-ink-line/40 overflow-hidden">
            <div
              className="h-full rounded bg-gradient-to-r from-amber-400 to-amber-600 transition-all duration-300"
              style={{ width: `${tokHWidth}%` }}
            />
          </div>
        </div>

        {/* Weight transfer bar */}
        <div className="space-y-1">
          <div className="flex items-center justify-between font-mono text-11">
            <span className="text-cream/60 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500 inline-block" />
              Weight-Transfer Baseline
            </span>
            <span className="text-cream/70 font-display">{b.tokens_per_second.toFixed(2)} tok/s</span>
          </div>
          <div className="h-2 w-full rounded bg-ink-line/40 overflow-hidden">
            <div
              className="h-full rounded bg-gradient-to-r from-slate-500 to-slate-600 transition-all duration-300"
              style={{ width: `${tokBWidth}%` }}
            />
          </div>
        </div>
      </div>

      {/* Row 2: Data Moved (Log Scale) */}
      <div className="space-y-2 pt-1 border-t border-ink-line/40">
        <div className="flex items-center justify-between">
          <span className="font-mono text-10 uppercase tracking-label text-cream/50">
            PCIe Data Moved (MB, Log Scale)
          </span>
          <span className="font-mono text-10 text-emerald-400 font-medium">
            {Math.round(byteRatio).toLocaleString()}× less traffic
          </span>
        </div>

        {/* Hybrid bar */}
        <div className="space-y-1">
          <div className="flex items-center justify-between font-mono text-11">
            <span className="text-amber font-medium">Hybrid (4.6 MB total)</span>
            <span className="text-emerald-400 font-mono text-10">0 evictions</span>
          </div>
          <div className="h-2 w-full rounded bg-ink-line/40 overflow-hidden">
            <div
              className="h-full rounded bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300"
              style={{ width: `${mbHWidth}%` }}
            />
          </div>
        </div>

        {/* Weight transfer bar */}
        <div className="space-y-1">
          <div className="flex items-center justify-between font-mono text-11">
            <span className="text-cream/60">
              Weight-Transfer ({b.transfer_mb.toLocaleString(undefined, { maximumFractionDigits: 0 })} MB)
            </span>
            <span className="text-rose-400/90 font-mono text-10">{b.evictions} evictions</span>
          </div>
          <div className="h-2 w-full rounded bg-ink-line/40 overflow-hidden">
            <div
              className="h-full rounded bg-gradient-to-r from-rose-500/80 to-rose-700/80 transition-all duration-300"
              style={{ width: `${mbBWidth}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
