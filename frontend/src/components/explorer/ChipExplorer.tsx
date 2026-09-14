import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Kicker } from '../Kicker';
import { WobbleRule } from '../WobbleRule';
import { ChipDiagram2D } from './ChipDiagram2D';
import { illustrativePlacement, type TierId } from '../../three/layout';
import { crossoverAt } from '../../lib/results';
import { usePrefersReducedMotion, useIsNarrowViewport, useInViewOnce } from '../../lib/useMediaHints';
import { RECORDED_S3_TRACE, type RecordedTraceEvent } from '../../data/traceEvents';
import type { SceneProps } from '../../three/Scene';
import { SectionHeaderArrow, SectionBottomJump } from '../SectionNav';

type Mode = 'weight_transfer' | 'hybrid' | 'trace_replay';

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

  const [mode, setMode] = useState<Mode>('trace_replay');
  const [explode, setExplode] = useState(0.3);
  const [focusTier, setFocusTier] = useState<TierId | null>(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [playToken, setPlayToken] = useState(0);
  const [canvasFailed, setCanvasFailed] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);

  // Slowed Recorded Trace Playback State
  const [traceStep, setTraceStep] = useState(0);
  const [tracePlaying, setTracePlaying] = useState(false);
  const [traceSpeed, setTraceSpeed] = useState<number>(850); // 850ms per step

  const handleRun = () => {
    setPlayToken((t) => t + 1);
    setIsAnimating(true);
    setTimeout(() => {
      setIsAnimating(false);
    }, mode === 'weight_transfer' ? 1800 : 1300);
  };

  // Trace timer effect
  useEffect(() => {
    if (mode !== 'trace_replay' || !tracePlaying) return;
    const interval = setInterval(() => {
      setTraceStep((prev) => (prev + 1) % RECORDED_S3_TRACE.length);
    }, traceSpeed);
    return () => clearInterval(interval);
  }, [mode, tracePlaying, traceSpeed]);

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

  const activeEvent: RecordedTraceEvent = RECORDED_S3_TRACE[traceStep] || RECORDED_S3_TRACE[0];

  return (
    <section id="explorer" className="on-ink bg-ink py-14 md:py-20 border-t border-ink-line/60 scroll-mt-20" aria-labelledby="explorer-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <Kicker as="p" className="text-amber">
                02 — Interactive Hierarchy
              </Kicker>
              <SectionHeaderArrow nextId="benchmark" nextNum="03" nextLabel="Empirical Benchmarks" isDark={true} />
            </div>
            <h2
              id="explorer-title"
              className="mt-5 max-w-[20ch] font-display text-[clamp(2.5rem,5.4vw,4.75rem)] leading-[0.92] text-cream">
              Watch The <span className="italic">Difference.</span>
            </h2>
          </div>
        </div>
        <p className="mt-4 max-w-[75ch] font-mono text-sm leading-relaxed text-cream/80">
          Visualizing expert placement across three physical tiers (GPU VRAM, Host DRAM, and CXL
          Far Memory). Replay a slowed-down hardware trace recorded during actual benchmark execution
          to observe how stationary host activations eliminate PCIe weight-transfer bottlenecks.
        </p>

        <WobbleRule tone="dark" seed={60} className="mt-10" />

        <div className="mt-8 grid gap-8 lg:grid-cols-[minmax(0,1fr)_22rem]">
          {/* 3D Canvas / 2D Fallback */}
          <div className="flex flex-col gap-3">
            <div ref={containerRef} className="h-[420px] overflow-hidden rounded-xl border border-white/10 bg-[#0d0e12] shadow-2xl md:h-[520px] relative">
              {use2D ? (
                <div className="flex h-full items-center justify-center p-4">
                  <ChipDiagram2D
                    placement={placement}
                    focusTier={focusTier}
                    mode={mode === 'trace_replay' ? 'hybrid' : mode}
                    playToken={playToken}
                    traceReplay={mode === 'trace_replay'}
                    activeTraceStep={traceStep}
                  />
                </div>
              ) : Scene ? (
                <Scene
                  placement={placement}
                  explode={explode}
                  mode={mode === 'trace_replay' ? 'hybrid' : mode}
                  playToken={playToken}
                  focusTier={focusTier}
                  autoRotate={autoRotate && !tracePlaying}
                  traceReplay={mode === 'trace_replay'}
                  activeTraceStep={traceStep}
                  tracePlaying={tracePlaying}
                />
              ) : (
                <div className="flex h-full items-center justify-center font-mono text-11 uppercase tracking-label text-khaki/40">
                  Loading scene…
                </div>
              )}

              {/* Floating Overlay Badge on 3D Viewport */}
              {mode === 'trace_replay' && (
                <div className="absolute top-4 left-4 rounded-lg bg-ink/90 border border-white/15 px-3 py-2 font-mono text-xs shadow-xl backdrop-blur">
                  <div className="flex items-center gap-2">
                    <span className={`h-2 w-2 rounded-full ${tracePlaying ? 'bg-amber animate-ping' : 'bg-white/40'}`} />
                    <span className="font-semibold text-cream">S3 Hardware Trace Simulation</span>
                    <span className="rounded bg-white/10 px-1.5 py-0.5 text-[10px] text-amber">Slowed 100,000×</span>
                  </div>
                </div>
              )}
            </div>

            {/* Trace Step Timeline Pips (Under Canvas) */}
            {mode === 'trace_replay' && (
              <div className="rounded-lg bg-white/[0.03] border border-white/10 p-3 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 overflow-x-auto py-1">
                  {RECORDED_S3_TRACE.map((ev, idx) => {
                    const isCur = idx === traceStep;
                    return (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => {
                          setTraceStep(idx);
                          setTracePlaying(false);
                        }}
                        className={`h-6 min-w-[24px] px-1.5 rounded font-mono text-[10px] font-bold transition-all ${
                          isCur
                            ? 'ring-2 ring-white scale-110 shadow-lg text-ink'
                            : 'opacity-60 hover:opacity-100 text-white/80'
                        }`}
                        style={{
                          backgroundColor: isCur ? ev.tierColor : `${ev.tierColor}33`,
                          color: isCur ? '#0c0d12' : ev.tierColor,
                        }}
                        title={`${ev.step}. ${ev.title} (${ev.tierName})`}
                      >
                        {ev.step}
                      </button>
                    );
                  })}
                </div>
                <div className="font-mono text-[11px] text-cream/70 shrink-0">
                  Event <strong className="text-amber">{traceStep + 1}</strong> of {RECORDED_S3_TRACE.length}
                </div>
              </div>
            )}
          </div>

          {/* Right Controls & Telemetry Column */}
          <div className="flex flex-col gap-5">
            {/* Mode Selector */}
            <div>
              <Kicker className="text-cream/40">Visualizer Mode</Kicker>
              <div className="mt-2.5 flex flex-col gap-1.5">
                <button
                  type="button"
                  onClick={() => {
                    setMode('trace_replay');
                    setTracePlaying(true);
                  }}
                  className={`w-full rounded-lg border px-3 py-2 font-mono text-xs font-semibold uppercase tracking-wider text-left transition-all ${
                    mode === 'trace_replay'
                      ? 'border-amber bg-amber/15 text-amber shadow-sm'
                      : 'border-white/10 bg-white/5 text-cream/60 hover:text-cream'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span>⏱ Recorded Trace Replay</span>
                    <span className="text-[10px] rounded bg-amber/20 text-amber px-1.5 py-0.2">Slowed Demo</span>
                  </div>
                </button>

                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setMode('hybrid');
                      setTracePlaying(false);
                    }}
                    className={`flex-1 rounded-lg border px-2.5 py-2 font-mono text-[11px] font-semibold uppercase tracking-wider transition-all ${
                      mode === 'hybrid'
                        ? 'border-emerald-500 bg-emerald-500/15 text-emerald-400'
                        : 'border-white/10 bg-white/5 text-cream/50 hover:text-cream'
                    }`}
                  >
                    ⚡ Hybrid Offload
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setMode('weight_transfer');
                      setTracePlaying(false);
                    }}
                    className={`flex-1 rounded-lg border px-2.5 py-2 font-mono text-[11px] font-semibold uppercase tracking-wider transition-all ${
                      mode === 'weight_transfer'
                        ? 'border-rose-500 bg-rose-500/15 text-rose-400'
                        : 'border-white/10 bg-white/5 text-cream/50 hover:text-cream'
                    }`}
                  >
                    ⇄ Weight Transfer
                  </button>
                </div>
              </div>
            </div>

            {/* Trace Replay Controls & Live Event HUD */}
            {mode === 'trace_replay' ? (
              <div className="rounded-xl border border-white/15 bg-white/[0.03] p-4 space-y-4">
                {/* Playback Button Strip */}
                <div className="flex items-center justify-between gap-2 border-b border-white/10 pb-3">
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => setTracePlaying((p) => !p)}
                      className="rounded-lg bg-amber px-3 py-1.5 font-mono text-xs font-bold uppercase tracking-wider text-ink hover:bg-amber-400 transition-all flex items-center gap-1.5"
                    >
                      {tracePlaying ? '⏸ Pause' : '▶ Play Trace'}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setTracePlaying(false);
                        setTraceStep((s) => (s + 1) % RECORDED_S3_TRACE.length);
                      }}
                      className="rounded-lg bg-white/10 px-2.5 py-1.5 font-mono text-xs text-cream hover:bg-white/20 transition-all"
                      title="Step to next recorded event"
                    >
                      ⏭
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setTracePlaying(false);
                        setTraceStep(0);
                      }}
                      className="rounded-lg bg-white/10 px-2.5 py-1.5 font-mono text-xs text-cream hover:bg-white/20 transition-all"
                      title="Reset trace to step 1"
                    >
                      ↺
                    </button>
                  </div>

                  {/* Speed Selector */}
                  <div className="flex items-center gap-1">
                    <span className="font-mono text-[10px] text-cream/40 mr-1">Speed:</span>
                    {[
                      { label: '0.5×', ms: 1400 },
                      { label: '1×', ms: 850 },
                      { label: '2×', ms: 450 },
                    ].map((s) => (
                      <button
                        key={s.label}
                        type="button"
                        onClick={() => setTraceSpeed(s.ms)}
                        className={`rounded px-1.5 py-0.5 font-mono text-[10px] border transition-all ${
                          traceSpeed === s.ms
                            ? 'bg-amber text-ink font-bold border-amber'
                            : 'border-white/10 text-cream/50 hover:text-cream'
                        }`}
                      >
                        {s.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Active Event Telemetry Card */}
                <div className="space-y-2.5 font-mono text-xs">
                  <div className="flex items-center justify-between">
                    <span
                      className="rounded px-2 py-0.5 font-bold uppercase tracking-wider text-[10px]"
                      style={{
                        backgroundColor: `${activeEvent.tierColor}22`,
                        color: activeEvent.tierColor,
                        border: `1px solid ${activeEvent.tierColor}66`,
                      }}
                    >
                      {activeEvent.tierName}
                    </span>
                    <span className="text-emerald-400 font-semibold">{activeEvent.latency}</span>
                  </div>

                  <div>
                    <h4 className="font-display text-base text-cream font-medium">
                      {activeEvent.title}
                    </h4>
                    <p className="mt-1 text-[11px] leading-relaxed text-cream/70">
                      {activeEvent.description}
                    </p>
                  </div>

                  <dl className="grid grid-cols-2 gap-2 rounded-lg bg-white/[0.02] p-2.5 border border-white/5 text-[11px]">
                    <div>
                      <dt className="text-cream/40 uppercase text-[9px]">Layer / Expert</dt>
                      <dd className="font-bold text-cream">L{activeEvent.layerIdx} · E{activeEvent.expertIdx}</dd>
                    </div>
                    <div>
                      <dt className="text-cream/40 uppercase text-[9px]">Payload Size</dt>
                      <dd className="font-bold text-amber">{activeEvent.sizeFormatted}</dd>
                    </div>
                    <div>
                      <dt className="text-cream/40 uppercase text-[9px]">Memory Bus Op</dt>
                      <dd className="font-semibold text-cream/90">{activeEvent.accessType} ({activeEvent.tag})</dd>
                    </div>
                    <div>
                      <dt className="text-cream/40 uppercase text-[9px]">Hardware Benefit</dt>
                      <dd className="font-semibold text-emerald-400">{activeEvent.savingsRatio}</dd>
                    </div>
                  </dl>
                </div>

                {/* Cumulative Hardware Metrics Tally */}
                <div className="border-t border-white/10 pt-3 space-y-1.5 font-mono text-[11px]">
                  <div className="flex justify-between text-cream/60 text-[10px] uppercase">
                    <span>12-Step Cumulative Trace Metrics</span>
                    <span className="text-emerald-400">83.3% Saved</span>
                  </div>
                  <div className="flex justify-between text-cream/80">
                    <span>Nebula Hybrid Bandwidth:</span>
                    <strong className="text-amber">33.0 MB</strong>
                  </div>
                  <div className="flex justify-between text-cream/60">
                    <span>Naive Weight Promotion:</span>
                    <span className="line-through text-rose-400/80">198.0 MB</span>
                  </div>
                </div>
              </div>
            ) : (
              /* Interactive Single Operation Controls */
              <div className="space-y-4">
                <p className="text-xs font-mono leading-relaxed text-cream/70">
                  {mode === 'weight_transfer'
                    ? '• Naive promotion migrates entire 16.5 MB expert weights to HBM over PCIe, forcing a resident block eviction.'
                    : '• Nebula keeps weights stationary in DRAM/CXL and streams only a 4 KB activation vector to compute.'}
                </p>

                <button
                  type="button"
                  onClick={handleRun}
                  disabled={!Scene && !use2D}
                  className={`flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 font-mono text-xs font-bold uppercase tracking-wider transition-all active:scale-[0.98] disabled:opacity-40 ${
                    isAnimating
                      ? 'bg-amber text-ink shadow-[0_0_16px_rgba(245,179,43,0.6)] ring-2 ring-amber'
                      : 'bg-cream text-ink hover:bg-amber hover:shadow-lg'
                  }`}
                >
                  {isAnimating && <span className="h-2 w-2 animate-ping rounded-full bg-ink" />}
                  {isAnimating
                    ? mode === 'weight_transfer'
                      ? 'Promoting 16.5 MB Weight...'
                      : 'Streaming 4 KB Activation...'
                    : mode === 'weight_transfer'
                    ? 'Execute Promotion (16.5 MB)'
                    : 'Execute Activation Stream (4 KB)'}
                </button>

                {crossover && (
                  <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-2 font-mono text-xs">
                    <span className="text-cream/40 uppercase tracking-wider text-[10px] block border-b border-white/5 pb-1.5">
                      Per-Operation Hardware Cost
                    </span>
                    <div className="flex justify-between text-cream/80">
                      <span>Weight promotion cost:</span>
                      <strong className="text-rose-400">~16.5 MB</strong>
                    </div>
                    <div className="flex justify-between text-cream/80">
                      <span>Activation round trip:</span>
                      <strong className="text-emerald-400">~4–8 KB</strong>
                    </div>
                    <div className="flex justify-between text-cream/80">
                      <span>Evictions @ 600 MB:</span>
                      <span>
                        <span className="text-rose-400">{crossover.baseline.evictions}</span>
                        {' → '}
                        <span className="text-emerald-400 font-bold">0 (Zero Thrash)</span>
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Isolate a Tier Filter */}
            <div>
              <div className="flex items-center justify-between">
                <Kicker className="text-cream/40">Isolate Tier</Kicker>
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
              <div className="mt-2.5 flex gap-2">
                {(['hbm', 'dram', 'cxl'] as TierId[]).map((tier) => {
                  const isFocused = focusTier === tier;
                  return (
                    <button
                      key={tier}
                      type="button"
                      onClick={() => setFocusTier((f) => (f === tier ? null : tier))}
                      className="flex-1 rounded-lg border px-2 py-1.5 font-mono text-xs font-semibold uppercase tracking-wider transition-all"
                      style={{
                        borderColor: isFocused ? TIER_HEX[tier] : '#32322c',
                        backgroundColor: isFocused ? `${TIER_HEX[tier]}22` : 'transparent',
                        color: isFocused ? TIER_HEX[tier] : 'rgba(244,243,237,0.6)',
                        boxShadow: isFocused ? `0 0 10px ${TIER_HEX[tier]}33` : 'none',
                      }}
                    >
                      {TIER_LABEL[tier]}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Explode & Auto-rotate Controls */}
            {!use2D && (
              <div className="border-t border-white/10 pt-4 space-y-2">
                <div className="flex items-center justify-between">
                  <Kicker className="text-cream/40">Layer Separation</Kicker>
                  <span className="font-mono text-[10px] text-cream/50">{(explode * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={explode}
                  onChange={(e) => setExplode(Number(e.target.value))}
                  className="w-full accent-amber"
                />

                <label className="flex items-center gap-2 font-mono text-xs text-cream/60 cursor-pointer pt-1">
                  <input
                    type="checkbox"
                    checked={autoRotate}
                    onChange={(e) => setAutoRotate(e.target.checked)}
                    className="accent-amber rounded"
                  />
                  Auto-rotate 3D Hierarchy
                </label>
              </div>
            )}
          </div>
        </div>

        <SectionBottomJump
          nextId="benchmark"
          nextNum="03"
          nextLabel="Empirical Benchmarks"
          isDark={true}
        />
      </div>
    </section>
  );
}
