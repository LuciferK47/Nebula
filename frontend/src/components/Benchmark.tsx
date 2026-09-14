import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { CrossoverChart } from './charts/CrossoverChart';
import { CapacityCliffChart } from './charts/CapacityCliffChart';
import { BatchScalingChart } from './charts/BatchScalingChart';
import { QwenWarmupChart } from './charts/QwenWarmupChart';
import { crossoverAt, getS1, getS7, getQwenLive, qwenPerTokenDeltas } from '../lib/results';
import { fadeUp, stagger } from '../utils/motion';
import { SectionHeaderArrow, SectionBottomJump } from './SectionNav';

const CORE_MECHANISMS = [
  {
    tag: 'Routing',
    title: 'Asymmetric Activation Offload',
    body: 'Cold expert activations (a few KB) route over PCIe to host memory rather than migrating 16.5 MB weights into VRAM, eliminating bus saturation and cache thrashing.',
  },
  {
    tag: 'Cache Policy',
    title: 'Decayed-LFU with Active Pinning',
    body: 'Frequency decays over time to match dynamic token routing, while active top-k experts are pinned to prevent intra-layer eviction races.',
  },
  {
    tag: 'Interconnect',
    title: 'Decoupled Async CUDA DMA',
    body: 'Transfers run on a dedicated CUDA stream isolated from compute kernels, enabling speculative prefetching without stalling GPU execution.',
  },
  {
    tag: 'Memory Tiering',
    title: 'CXL Disaggregated Far Memory',
    body: 'Pools secondary experts across host memory and CXL links without OS paging overhead, maintaining 100% placement stability across bandwidth sweeps.',
  },
];

export function Benchmark() {
  const crossover600 = crossoverAt(600);
  const crossover900 = crossoverAt(900);
  const s1runs = getS1();
  const s7 = getS7();
  const qwenLive = getQwenLive();
  const qwenSteps = qwenPerTokenDeltas();

  return (
    <section id="benchmark" className="on-ink bg-ink-soft pb-16 pt-14 md:pb-24 scroll-mt-20" aria-labelledby="bench-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-ink-line pb-5">
          <div>
            <div className="flex items-center gap-3">
              <Kicker as="p" className="text-amber">
                03 — Empirical Performance & Scaling Benchmarks
              </Kicker>
              <SectionHeaderArrow nextId="limits" nextNum="04" nextLabel="Robustness & Limits" isDark={true} />
            </div>
            <h2
              id="bench-title"
              className="mt-2 font-display text-[clamp(2.4rem,4.5vw,3.8rem)] leading-[0.95] text-cream"
            >
              Empirical Evidence: <span className="italic">Moving Activations Beats Thrashing Weights.</span>
            </h2>
          </div>
          <p className="max-w-[54ch] font-mono text-sm leading-relaxed text-cream/80">
            Empirical measurements across capacity sweeps, concurrent batch scaling, and live
            autoregressive generation on Qwen1.5-MoE.
          </p>
        </div>

        {/* Interactive 2x2 Modern Benchmark Dashboard Grid */}
        <div className="mt-10 grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Card 1: Crossover Advantage */}
          {crossover600 && (
            <div className="rounded-xl border border-ink-line/90 bg-ink-panel/80 p-6 shadow-xl backdrop-blur-sm flex flex-col justify-between hover:border-amber/40 transition-colors">
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-semibold tracking-label text-amber uppercase">
                    Scenario 2 — Crossover Advantage
                  </span>
                  <span className="font-mono text-xs text-cream/60">600 MB vs 900 MB</span>
                </div>
                <h4 className="mt-2 font-display text-xl text-cream font-medium">
                  Hybrid Execution vs. Weight-Transfer
                </h4>
                <p className="mt-1.5 font-mono text-xs leading-relaxed text-cream/75">
                  Comparing throughput and PCIe traffic between naive weight swapping and hybrid activation offload.
                </p>
              </div>

              <div className="mt-5 pt-2">
                <CrossoverChart crossover600={crossover600} crossover900={crossover900} />
              </div>

              {/* High-Level Summary */}
              <div className="mt-5 rounded-lg border border-amber/20 bg-amber/5 p-3.5">
                <p className="font-mono text-xs text-cream/90 leading-relaxed">
                  <strong className="text-amber">Key Takeaway:</strong> Weight-transfer continually evicts and refetches 16.5 MB weights over PCIe ({crossover600.baseline.evictions} evictions, 14.3 GB traffic). Hybrid mode keeps weights stationary across tiers and streams lightweight 4 KB activations, eliminating evictions and boosting throughput by {crossover600.speedup.toFixed(1)}×.
                </p>
              </div>
            </div>
          )}

          {/* Card 2: Capacity Cliff */}
          {s1runs.length > 0 && (
            <div className="rounded-xl border border-ink-line/90 bg-ink-panel/80 p-6 shadow-xl backdrop-blur-sm flex flex-col justify-between hover:border-amber/40 transition-colors">
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-semibold tracking-label text-amber uppercase">
                    Scenario 1 — Capacity Sweep
                  </span>
                  <span className="font-mono text-xs text-cream/60">300 MB → 1200 MB</span>
                </div>
                <h4 className="mt-2 font-display text-xl text-cream font-medium">
                  Where Tiering Stops Being Free
                </h4>
                <p className="mt-1.5 font-mono text-xs leading-relaxed text-cream/75">
                  Throughput and cache hit rate vs. allocated VRAM expert budget. Hover data points to inspect values.
                </p>
              </div>

              <div className="mt-5 pt-1">
                <CapacityCliffChart runs={s1runs} />
              </div>

              {/* High-Level Summary */}
              <div className="mt-5 rounded-lg border border-amber/20 bg-amber/5 p-3.5">
                <p className="font-mono text-xs text-cream/90 leading-relaxed">
                  <strong className="text-amber">Key Takeaway:</strong> Cache hit rate saturates smoothly above 600 MB (~50% working set), reaching a plateau where additional VRAM yields diminishing returns. The slight throughput dip at 900 MB reflects real hardware thermal drift during continuous GPU benchmarking.
                </p>
              </div>
            </div>
          )}

          {/* Card 3: Batch Scaling & Thrashing Collapse */}
          {s7 && s7.runs.length > 0 && (
            <div className="rounded-xl border border-ink-line/90 bg-ink-panel/80 p-6 shadow-xl backdrop-blur-sm flex flex-col justify-between hover:border-amber/40 transition-colors">
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-semibold tracking-label text-amber uppercase">
                    Scenario 7 — Concurrent Batch Scaling
                  </span>
                  <span className="font-mono text-xs text-emerald-400 font-semibold">
                    Immune to Cache Thrashing
                  </span>
                </div>
                <h4 className="mt-2 font-display text-xl text-cream font-medium">
                  Multi-Token Concurrency & Thrashing Collapse
                </h4>
                <p className="mt-1.5 font-mono text-xs leading-relaxed text-cream/75">
                  Evaluating throughput and cache stability under concurrent batching (Batch 1 to 8).
                </p>
              </div>

              <div className="mt-5 pt-1">
                <BatchScalingChart runs={s7.runs} />
              </div>

              {/* High-Level Summary */}
              <div className="mt-5 rounded-lg border border-amber/20 bg-amber/5 p-3.5">
                <p className="font-mono text-xs text-cream/90 leading-relaxed">
                  <strong className="text-amber">Key Takeaway:</strong> Under concurrent batching, weight-transfer suffers catastrophic thrashing as requests compete for different experts (hit rate collapses from 35% to 0.5%, causing 1,375 evictions and 23.8 GB traffic). Hybrid mode keeps hit rate rock-solid at ~39%, scaling throughput up to 25.8 tok/s.
                </p>
              </div>
            </div>
          )}

          {/* Card 4: Qwen Warm-up */}
          {qwenLive && qwenSteps.length > 0 && (
            <div className="rounded-xl border border-ink-line/90 bg-ink-panel/80 p-6 shadow-xl backdrop-blur-sm flex flex-col justify-between hover:border-amber/40 transition-colors">
              <div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-semibold tracking-label text-amber uppercase">
                    Live Decode — Qwen1.5-MoE-A2.7B
                  </span>
                  <span className="font-mono text-xs text-cream/60">
                    {qwenSteps[0].latencyMs.toFixed(0)} ms → {qwenSteps[qwenSteps.length - 1].latencyMs.toFixed(0)} ms
                  </span>
                </div>
                <h4 className="mt-2 font-display text-xl text-cream font-medium">
                  Real Cache Warm-Up & Expert Residency
                </h4>
                <p className="mt-1.5 font-mono text-xs leading-relaxed text-cream/75">
                  Token-by-token decode latency and tier lookup composition on a full 14.3B parameter model.
                </p>
              </div>

              <div className="mt-5 pt-1">
                <QwenWarmupChart steps={qwenSteps} />
              </div>

              {/* High-Level Summary */}
              <div className="mt-5 rounded-lg border border-amber/20 bg-amber/5 p-3.5">
                <p className="font-mono text-xs text-cream/90 leading-relaxed">
                  <strong className="text-amber">Key Takeaway:</strong> Cold-cache generation pays {qwenSteps[0].latencyMs.toFixed(0)} ms on the initial token. As hot experts establish residency across GPU VRAM and host DRAM, per-token decode latency drops to {qwenSteps[qwenSteps.length - 1].latencyMs.toFixed(0)} ms (3.2× faster decode) without disk lookups.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* 4 Core Architectural Mechanisms */}
        <div className="mt-14 border-t border-ink-line/80 pt-10">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-amber">
              Architectural Pillars
            </span>
          </div>

          <motion.div
            variants={stagger()}
            initial="hidden"
            whileInView="visible"
            className="mt-6 grid gap-5 sm:grid-cols-2 lg:grid-cols-4"
          >
            {CORE_MECHANISMS.map((m) => (
              <motion.div
                key={m.title}
                variants={fadeUp}
                className="rounded-xl border border-ink-line/80 bg-ink-panel/60 p-5 shadow-sm"
              >
                <span className="font-mono text-[11px] uppercase tracking-label text-amber font-semibold">
                  {m.tag}
                </span>
                <h4 className="mt-1.5 font-display text-base text-cream font-medium">
                  {m.title}
                </h4>
                <p className="mt-2.5 font-mono text-xs leading-relaxed text-cream/75">
                  {m.body}
                </p>
              </motion.div>
            ))}
          </motion.div>

          <SectionBottomJump
            nextId="limits"
            nextNum="04"
            nextLabel="Robustness & Limits"
            isDark={true}
          />
        </div>
      </div>
    </section>
  );
}
