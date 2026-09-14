import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { PillButton } from './PillButton';
import { crossoverAt, qwenLocalityRatio } from '../lib/results';
import { fadeUp, stagger } from '../utils/motion';
import { SectionBottomJump } from './SectionNav';

export function Hero() {
  const crossover600 = crossoverAt(600);
  const locality = qwenLocalityRatio();

  const metrics = [
    {
      value: crossover600 ? `${crossover600.speedup.toFixed(2)}×` : '3.92×',
      label: 'Throughput Speedup',
      detail: 'Hybrid vs weight-transfer @ 600 MB',
    },
    {
      value: crossover600 ? `${Math.round(crossover600.byteRatio).toLocaleString()}×` : '3,110×',
      label: 'Less PCIe Traffic',
      detail: '4.6 MB vs 14,308 MB moved',
    },
    {
      value: crossover600 ? `${crossover600.baseline.evictions} → 0` : '114 → 0',
      label: 'Evictions Eliminated',
      detail: 'Zero cache thrashing in hybrid mode',
    },
    {
      value: locality ? `${locality.toFixed(2)}×` : '5.31×',
      label: 'Real Cache Locality',
      detail: '66.9% hit rate at 12.6% residency',
    },
  ];

  return (
    <section className="relative overflow-hidden bg-cream pb-12 pt-12 md:pb-16 md:pt-16 border-b border-ink/10">
      <motion.div
        variants={stagger()}
        initial="hidden"
        animate="visible"
        className="mx-auto max-w-site px-5 text-center md:px-8"
      >
        <motion.div variants={fadeUp} className="flex items-center justify-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-amber" />
          <Kicker as="p" className="text-ink/60 font-mono text-11 tracking-wider uppercase">
            MemTier-MoE · Hardware Evaluation
          </Kicker>
        </motion.div>

        <motion.h1
          variants={fadeUp}
          className="mx-auto mt-4 max-w-[22ch] font-display text-[clamp(2.5rem,5.8vw,5.25rem)] font-normal leading-[0.94] tracking-[-0.01em] text-ink"
        >
          Asymmetric Memory Hierarchy for{' '}
          <span className="italic">Mixture-of-Experts.</span>
        </motion.h1>

        <motion.p
          variants={fadeUp}
          className="mx-auto mt-6 max-w-[62ch] font-mono text-[0.82rem] leading-relaxed text-ink/75 md:text-[0.88rem]"
        >
          A three-tier residency engine (VRAM, Host DRAM, and CXL) that decodes large MoE models on
          constrained hardware by keeping weights resident and routing token activations over PCIe
          rather than thrashing weights.
        </motion.p>

        <motion.div
          variants={fadeUp}
          className="mt-8 flex flex-wrap items-center justify-center gap-3"
        >
          <PillButton href="#benchmark">View Measured Benchmarks</PillButton>
          <PillButton href="#architecture" variant="outline">
            Architecture & Tiers
          </PillButton>
          <PillButton href="#live-lab" variant="outline">
            Live GPU Inference
          </PillButton>
        </motion.div>

        {/* High-Signal Headline KPI Metric Strip */}
        <motion.div
          variants={fadeUp}
          className="mt-12 grid grid-cols-2 gap-3 sm:grid-cols-4 rounded-xl border border-ink/12 bg-white/80 p-4 shadow-sm backdrop-blur md:p-5 text-left"
        >
          {metrics.map((m) => (
            <div key={m.label} className="px-2 py-1">
              <span className="font-mono text-10 uppercase tracking-label text-ink/45 block">
                {m.label}
              </span>
              <p className="mt-1 font-display text-[clamp(1.75rem,2.8vw,2.5rem)] leading-none text-ink font-medium">
                {m.value}
              </p>
              <p className="mt-1.5 font-mono text-10 text-ink/60 leading-tight">
                {m.detail}
              </p>
            </div>
          ))}
        </motion.div>

        <motion.div variants={fadeUp}>
          <SectionBottomJump
            nextId="architecture"
            nextNum="01"
            nextLabel="Heterogeneous Memory Architecture"
            isDark={false}
          />
        </motion.div>
      </motion.div>
    </section>
  );
}
