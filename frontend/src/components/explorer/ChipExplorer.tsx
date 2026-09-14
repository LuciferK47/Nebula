import React, { useMemo, useRef, useState } from 'react';
import { Kicker } from '../Kicker';
import { WobbleRule } from '../WobbleRule';
import { ProvenanceBadge } from '../Provenance';
import { ChipDiagram2D } from './ChipDiagram2D';
import { illustrativePlacement, type TierId } from '../../three/layout';
import { crossoverAt } from '../../lib/results';
import { usePrefersReducedMotion, useIsNarrowViewport, useInViewOnce } from '../../lib/useMediaHints';
import type { SceneProps } from '../../three/Scene';

type Mode = 'weight_transfer' | 'hybrid';

// The one scenario config that exercises all three tiers meaningfully —
// see results/scenarios/s3.json (hbm 39.94% / dram 23.74% / cxl 36.33%).
const HBM_FRAC = 0.3994;
const DRAM_FRAC = 0.2374;

const TIER_LABEL: Record<TierId, string> = { hbm: 'HBM', dram: 'DRAM', cxl: 'CXL' };
const TIER_HEX: Record<TierId, string> = { hbm: '#e4512b', dram: '#e5b52f', cxl: '#8fa3b8' };

export function ChipExplorer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const inView = useInViewOnce(containerRef);
  const reducedMotion = usePrefersReducedMotion();
  const narrow = useIsNarrowViewport(768);

  const [mode, setMode] = useState<Mode>('weight_transfer');
  const [explode, setExplode] = useState(0.3);
  const [focusTier, setFocusTier] = useState<TierId | null>(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [playToken, setPlayToken] = useState(0);
  const [canvasFailed, setCanvasFailed] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);

  const handleRun = () => {
    setPlayToken((t) => t + 1);
    setIsAnimating(true);
    setTimeout(() => {
      setIsAnimating(false);
    }, mode === 'weight_transfer' ? 1800 : 1300);
  };

  const placement = useMemo(() => illustrativePlacement(HBM_FRAC, DRAM_FRAC), []);
  const crossover = crossoverAt(600);

  const use2D = reducedMotion || narrow || canvasFailed;

  const [Scene, setScene] = useState<React.ComponentType<SceneProps> | null>(null);
  React.useEffect(() => {
    if (!inView || use2D) return;
    let alive = true;
    import('../../three/Scene').
    then((mod) => {
      if (alive) setScene(() => mod.Scene);
    }).
    catch(() => {
      if (alive) setCanvasFailed(true);
    });
    return () => {
      alive = false;
    };
  }, [inView, use2D]);

  return (
    <section id="explorer" className="on-ink bg-ink py-14 md:py-20 border-t border-ink-line/60" aria-labelledby="explorer-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <Kicker as="p" className="text-amber">
              02 — Interactive Hierarchy
            </Kicker>
            <h2
              id="explorer-title"
              className="mt-5 max-w-[20ch] font-display text-[clamp(2.5rem,5.4vw,4.75rem)] leading-[0.92] text-cream">

              Watch The <span className="italic">Difference.</span>
            </h2>
          </div>
        </div>
        <p className="mt-4 max-w-[75ch] font-mono text-sm leading-relaxed text-cream/80">
          Visualizing expert placement across the three memory tiers (GPU VRAM, Host DRAM, and CXL
          Far Memory) for a 24-layer Mixture-of-Experts architecture. Select an execution mode and
          run an operation to compare full weight promotion (migrating 16.5 MB per miss) against
          asymmetric hybrid offloading (streaming 4 KB activations directly to stationary weights).
        </p>

        <WobbleRule tone="dark" seed={60} className="mt-10" />

        <div className="mt-8 grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
          <div ref={containerRef} className="h-[420px] overflow-hidden rounded-lg bg-ink-soft md:h-[520px]">
            {use2D ? (
              <div className="flex h-full items-center justify-center p-4">
                <ChipDiagram2D
                  placement={placement}
                  focusTier={focusTier}
                  mode={mode}
                  playToken={playToken}
                />
              </div>
            ) : Scene ? (
              <Scene
                placement={placement}
                explode={explode}
                mode={mode}
                playToken={playToken}
                focusTier={focusTier}
                autoRotate={autoRotate}
              />
            ) : (
              <div className="flex h-full items-center justify-center font-mono text-11 uppercase tracking-label text-khaki/40">
                Loading scene…
              </div>
            )}
          </div>

          <div className="flex flex-col gap-6">
            <div>
              <Kicker className="text-cream/40">Mode</Kicker>
              <div className="mt-3 flex gap-2">
                {(['weight_transfer', 'hybrid'] as Mode[]).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setMode(m)}
                    className={`flex-1 rounded-full border px-3 py-2 font-mono text-10 uppercase tracking-label transition-all duration-150 active:scale-95 ${
                      mode === m
                        ? 'border-amber bg-amber/15 text-amber shadow-[0_0_12px_rgba(245,179,43,0.2)]'
                        : 'border-ink-line text-cream/50 hover:border-cream/30 hover:text-cream'
                    }`}
                  >
                    {m === 'weight_transfer' ? 'Weight transfer' : 'Hybrid'}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-[11px] font-mono leading-relaxed text-khaki/75">
                {mode === 'weight_transfer'
                  ? '• Migrates full 16.5 MB expert weights to HBM, evicting a resident block.'
                  : '• Keeps weights stationary in DRAM/CXL; streams ~4 KB activation to compute.'}
              </p>
              <button
                type="button"
                onClick={handleRun}
                disabled={!Scene && !use2D}
                className={`mt-3 flex w-full items-center justify-center gap-2 rounded-full px-4 py-2.5 font-mono text-10 font-medium uppercase tracking-label transition-all duration-150 active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-40 ${
                  isAnimating
                    ? 'bg-amber text-ink shadow-[0_0_16px_rgba(245,179,43,0.6)] ring-2 ring-amber'
                    : 'bg-cream text-ink hover:bg-amber hover:shadow-[0_0_12px_rgba(245,179,43,0.3)]'
                }`}
              >
                {isAnimating && (
                  <span className="inline-block h-2 w-2 animate-ping rounded-full bg-ink" />
                )}
                {isAnimating
                  ? mode === 'weight_transfer'
                    ? 'Promoting expert (16.5 MB)...'
                    : 'Offloading activation (4 KB)...'
                  : mode === 'weight_transfer'
                  ? 'Run a promotion'
                  : 'Run an activation offload'}
              </button>
            </div>

            <div>
              <div className="flex items-center justify-between">
                <Kicker className="text-cream/40">Isolate a tier</Kicker>
                {focusTier && (
                  <button
                    type="button"
                    onClick={() => setFocusTier(null)}
                    className="font-mono text-[10px] uppercase tracking-wider text-amber/80 underline decoration-amber/40 hover:text-amber"
                  >
                    Clear Filter
                  </button>
                )}
              </div>
              <div className="mt-3 flex gap-2">
                {(['hbm', 'dram', 'cxl'] as TierId[]).map((tier) => {
                  const isFocused = focusTier === tier;
                  return (
                    <button
                      key={tier}
                      type="button"
                      onClick={() => setFocusTier((f) => (f === tier ? null : tier))}
                      className="flex-1 rounded-full border px-2 py-2 font-mono text-10 uppercase tracking-label transition-all duration-150 active:scale-95"
                      style={{
                        borderColor: isFocused ? TIER_HEX[tier] : '#32322c',
                        backgroundColor: isFocused ? `${TIER_HEX[tier]}22` : 'transparent',
                        color: isFocused ? TIER_HEX[tier] : 'rgba(244,243,237,0.5)',
                        boxShadow: isFocused ? `0 0 10px ${TIER_HEX[tier]}33` : 'none',
                      }}
                    >
                      {TIER_LABEL[tier]}
                    </button>
                  );
                })}
              </div>
            </div>

            {!use2D &&
            <div>
                <Kicker className="text-cream/40">Explode</Kicker>
                <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={explode}
                onChange={(e) => setExplode(Number(e.target.value))}
                className="mt-3 w-full accent-amber" />

                <label className="mt-2 flex items-center gap-2 font-mono text-10 text-cream/50">
                  <input
                  type="checkbox"
                  checked={autoRotate}
                  onChange={(e) => setAutoRotate(e.target.checked)}
                  className="accent-amber" />

                  Auto-rotate
                </label>
              </div>
            }

            {crossover &&
            <div className="mt-2 border-t border-ink-line pt-5">
                <Kicker className="text-cream/40">Per-operation cost</Kicker>
                <dl className="mt-3 space-y-2 font-mono text-11">
                  <div className="flex justify-between text-cream/70">
                    <dt>Weight promotion</dt>
                    <dd className="text-cold">~16.5 MB</dd>
                  </div>
                  <div className="flex justify-between text-cream/70">
                    <dt>Activation round trip</dt>
                    <dd className="text-ambient">~4–8 KB</dd>
                  </div>
                  <div className="flex justify-between text-cream/70">
                    <dt>Evictions @ 600 MB</dt>
                    <dd>
                      <span className="text-hot">{crossover.baseline.evictions}</span>
                      {' → '}
                      <span className="text-amber">0</span>
                    </dd>
                  </div>
                </dl>
              </div>
            }
          </div>
        </div>
      </div>
    </section>);

}
