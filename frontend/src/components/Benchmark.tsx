import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { ProvenanceBadge } from './Provenance';
import { CrossoverChart } from './charts/CrossoverChart';
import { CapacityCliffChart } from './charts/CapacityCliffChart';
import { CXLSensitivityChart } from './charts/CXLSensitivityChart';
import { QwenWarmupChart } from './charts/QwenWarmupChart';
import { crossoverAt, getS1, getS3, getQwenLive, qwenPerTokenDeltas } from '../lib/results';
import { fadeUp, stagger } from '../utils/motion';

const CORE_MECHANISMS = [
  {
    tag: 'Routing',
    title: 'Asymmetric Activation Offload',
    body: 'Cold expert activations (a few KB) route to host CPU rather than swapping 16.5 MB weights into VRAM, eliminating bus saturation and cache thrashing.',
  },
  {
    tag: 'Cache Policy',
    title: 'Decayed-LFU with Active Pinning',
    body: 'Frequency decays over time to match dynamic token routing, while active top-k experts are pinned to prevent intra-layer eviction races.',
  },
  {
    tag: 'Interconnect',
    title: 'Decoupled Async CUDA DMA',
    body: 'Transfers run on a dedicated CUDA stream isolated from compute kernels, enabling speculative prefetching without compute stalls.',
  },
  {
    tag: 'Empirical Rigor',
    title: 'Bandwidth Sensitivity & Zero Drift',
    body: '16× CXL bandwidth sweeps (4–64 GB/s) show 100% placement invariance, verifying that decisions are not steered by simulation calibration.',
  },
];

export function Benchmark() {
  const crossover600 = crossoverAt(600);
  const crossover900 = crossoverAt(900);
  const s1runs = getS1();
  const s3 = getS3();
  const qwenLive = getQwenLive();
  const qwenSteps = qwenPerTokenDeltas();

  return (
    <section id="benchmark" className="on-ink bg-ink-soft pb-14 pt-12 md:pb-20" aria-labelledby="bench-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-ink-line pb-4">
          <div>
            <Kicker as="p" className="text-amber">
              03 — Benchmark & Evaluation
            </Kicker>
            <h2
              id="bench-title"
              className="mt-2 font-display text-[clamp(2.2rem,4.2vw,3.5rem)] leading-[0.95] text-cream"
            >
              Moving Less <span className="italic">Beats Moving Faster.</span>
            </h2>
          </div>
          <p className="max-w-[48ch] font-mono text-[0.78rem] leading-relaxed text-khaki/75">
            Empirical evidence from live physical hardware runs across S1–S3 sweeps and real
            Qwen1.5-MoE autoregressive generation.
          </p>
        </div>

        {/* Interactive 2x2 Modern Benchmark Dashboard Grid */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Card 1: Crossover Advantage */}
          {crossover600 && (
            <div className="rounded-xl border border-ink-line/80 bg-ink-panel/70 p-5 shadow-lg backdrop-blur-sm flex flex-col justify-between hover:border-amber/40 transition-colors">
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-10 font-semibold tracking-label text-amber uppercase">
                    Scenario 2 — Crossover
                  </span>
                  <span className="font-mono text-10 text-khaki/60">600 MB vs 900 MB</span>
                </div>
                <h4 className="mt-1.5 font-display text-lg text-cream">
                  Hybrid Execution vs. Weight-Transfer
                </h4>
                <p className="mt-1 font-mono text-[0.72rem] leading-relaxed text-khaki/70">
                  Moving activations instead of weights eliminates {crossover600.baseline.evictions} cache thrash evictions.
                </p>
              </div>
              <div className="mt-4 pt-2">
                <CrossoverChart crossover600={crossover600} crossover900={crossover900} />
              </div>
            </div>
          )}

          {/* Card 2: Capacity Cliff */}
          {s1runs.length > 0 && (
            <div className="rounded-xl border border-ink-line/80 bg-ink-panel/70 p-5 shadow-lg backdrop-blur-sm flex flex-col justify-between hover:border-amber/40 transition-colors">
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-10 font-semibold tracking-label text-amber uppercase">
                    Scenario 1 — Capacity Sweep
                  </span>
                  <span className="font-mono text-10 text-khaki/60">300 MB → 1200 MB</span>
                </div>
                <h4 className="mt-1.5 font-display text-lg text-cream">
                  Where Tiering Stops Being Free
                </h4>
                <p className="mt-1 font-mono text-[0.72rem] leading-relaxed text-khaki/70">
                  Hit-rate saturates smoothly above 600 MB. Hover data points to inspect per-budget throughput.
                </p>
              </div>
              <div className="mt-4 pt-1">
                <CapacityCliffChart runs={s1runs} />
              </div>
            </div>
          )}

          {/* Card 3: CXL Sensitivity */}
          {s3 && s3.runs.length > 0 && (
            <div className="rounded-xl border border-ink-line/80 bg-ink-panel/70 p-5 shadow-lg backdrop-blur-sm flex flex-col justify-between hover:border-purple-400/40 transition-colors">
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-10 font-semibold tracking-label text-purple-400 uppercase">
                    Scenario 3 — CXL Independence
                  </span>
                  <span className="font-mono text-10 text-emerald-400">
                    {s3.placement_stable_across_bandwidth_sweep ? '100% Invariant' : 'Bandwidth sweep'}
                  </span>
                </div>
                <h4 className="mt-1.5 font-display text-lg text-cream">
                  CXL Link Bandwidth Sensitivity
                </h4>
                <p className="mt-1 font-mono text-[0.72rem] leading-relaxed text-khaki/70">
                  Sweeping emulated CXL bandwidth (4–64 GB/s) or disabling emulation produces zero placement drift.
                </p>
              </div>
              <div className="mt-4 pt-1">
                <CXLSensitivityChart runs={s3.runs} />
              </div>
            </div>
          )}

          {/* Card 4: Qwen Warm-up */}
          {qwenLive && qwenSteps.length > 0 && (
            <div className="rounded-xl border border-ink-line/80 bg-ink-panel/70 p-5 shadow-lg backdrop-blur-sm flex flex-col justify-between hover:border-amber/40 transition-colors">
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-10 font-semibold tracking-label text-amber uppercase">
                    Live Decode — Qwen1.5-MoE-A2.7B
                  </span>
                  <span className="font-mono text-10 text-khaki/60">
                    {qwenSteps[0].latencyMs.toFixed(0)} ms → {qwenSteps[qwenSteps.length - 1].latencyMs.toFixed(0)} ms
                  </span>
                </div>
                <h4 className="mt-1.5 font-display text-lg text-cream">
                  Real Cache Warm-Up & Expert Residency
                </h4>
                <p className="mt-1 font-mono text-[0.72rem] leading-relaxed text-khaki/70">
                  Cold decode costs {qwenSteps[0].latencyMs.toFixed(0)} ms; 10 tokens later, HBM & DRAM hits drop decode to {qwenSteps[qwenSteps.length - 1].latencyMs.toFixed(0)} ms.
                </p>
              </div>
              <div className="mt-4 pt-1">
                <QwenWarmupChart steps={qwenSteps} />
              </div>
            </div>
          )}
        </div>

        {/* 4 Core Architectural Mechanisms */}
        <div className="mt-12 border-t border-ink-line/70 pt-8">
          <div className="flex items-center justify-between">
            <span className="font-mono text-10 font-bold uppercase tracking-wider text-amber">
              Core Mechanisms
            </span>
            <ProvenanceBadge value="measured" dark />
          </div>

          <motion.div
            variants={stagger()}
            initial="hidden"
            whileInView="visible"
            className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
          >
            {CORE_MECHANISMS.map((m) => (
              <motion.div
                key={m.title}
                variants={fadeUp}
                className="rounded-lg border border-ink-line/60 bg-ink-panel/50 p-4"
              >
                <span className="font-mono text-[10px] uppercase tracking-label text-amber/80 font-medium">
                  {m.tag}
                </span>
                <h4 className="mt-1 font-display text-sm text-cream font-medium">
                  {m.title}
                </h4>
                <p className="mt-2 font-mono text-[0.72rem] leading-relaxed text-khaki/70">
                  {m.body}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
