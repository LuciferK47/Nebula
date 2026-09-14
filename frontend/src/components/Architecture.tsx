import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { tiers } from '../data/tiers';
import { harness } from '../data/benchmarks';
import { fadeUp, inView, stagger } from '../utils/motion';

const dotFor: Record<string, string> = {
  hbm: '#e4512b',
  dram: '#e5b52f',
  cxl: '#8fa3b8',
};

export function Architecture() {
  return (
    <section id="architecture" className="on-ink bg-ink py-16 md:py-24" aria-labelledby="arch-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-ink-line pb-5">
          <div>
            <Kicker as="p" className="text-amber">
              01 — Heterogeneous Memory Architecture
            </Kicker>
            <h2
              id="arch-title"
              className="mt-2 font-display text-[clamp(2.4rem,4.5vw,3.8rem)] leading-[0.95] text-cream"
            >
              Three Heterogeneous Tiers. <span className="italic">One Unified Residency Hierarchy.</span>
            </h2>
          </div>
          <p className="max-w-[54ch] font-mono text-sm leading-relaxed text-cream/80">
            Real GPU memory and host DRAM with asynchronous CUDA DMA transfers, coupled with a
            disaggregated CXL far memory pool for scalable parameter residency.
          </p>
        </div>

        {/* 3-Tier Concise Grid */}
        <motion.div
          variants={stagger()}
          initial="hidden"
          whileInView="visible"
          viewport={inView}
          className="mt-10 grid gap-6 md:grid-cols-3"
        >
          {tiers.map((tier) => (
            <motion.div
              key={tier.id}
              variants={fadeUp}
              className="flex flex-col justify-between rounded-xl border border-ink-line/90 bg-ink-panel/80 p-6 backdrop-blur-sm hover:border-amber/40 transition-colors shadow-lg"
            >
              <div>
                <div className="flex items-center gap-2">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: dotFor[tier.id] }}
                    aria-hidden="true"
                  />
                  <span className="font-mono text-xs font-semibold tracking-label uppercase text-amber">
                    Tier {tier.index} · {tier.medium}
                  </span>
                </div>

                <h3 className="mt-4 font-display text-2xl text-cream font-medium">
                  {tier.name}
                </h3>

                <p className="mt-3 font-mono text-xs leading-relaxed text-cream/80">
                  {tier.note}
                </p>
              </div>

              <div className="mt-6 border-t border-ink-line/80 pt-4 grid grid-cols-3 gap-3 text-left">
                <div>
                  <span className="font-mono text-[10px] uppercase tracking-label text-cream/50 block">
                    Bandwidth
                  </span>
                  <span className="font-display text-base text-cream font-medium">{tier.bandwidth}</span>
                </div>
                <div>
                  <span className="font-mono text-[10px] uppercase tracking-label text-cream/50 block">
                    Latency
                  </span>
                  <span className="font-display text-base text-cream font-medium">{tier.latency}</span>
                </div>
                <div>
                  <span className="font-mono text-[10px] uppercase tracking-label text-cream/50 block">
                    Capacity
                  </span>
                  <span className="font-mono text-xs text-cream/90 font-medium">{tier.capacity}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Memory Efficiency & CXL Architecture Callout */}
        <div className="mt-10 rounded-xl border border-ink-line/80 bg-ink-soft/80 p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-md">
          <div className="flex items-center gap-3">
            <span className="h-3 w-1 bg-amber rounded-full" />
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-amber">
              {harness.platform}
            </span>
          </div>
          <p className="font-mono text-xs md:text-sm text-cream/90 leading-relaxed max-w-[80ch]">
            {harness.detail}
          </p>
        </div>
      </div>
    </section>
  );
}
