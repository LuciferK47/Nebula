import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { ProvenanceBadge } from './Provenance';
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
    <section id="architecture" className="on-ink bg-ink py-14 md:py-20" aria-labelledby="arch-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-ink-line pb-4">
          <div>
            <Kicker as="p" className="text-amber">
              01 — Architecture
            </Kicker>
            <h2
              id="arch-title"
              className="mt-2 font-display text-[clamp(2.2rem,4.2vw,3.5rem)] leading-[0.95] text-cream"
            >
              Three Tiers. <span className="italic">One Residency Hierarchy.</span>
            </h2>
          </div>
          <p className="max-w-[48ch] font-mono text-[0.78rem] leading-relaxed text-khaki/75">
            Real GPU GDDR6 and host DRAM memory with asynchronous CUDA DMA transfers, coupled with a
            software-emulated CXL far memory pool.
          </p>
        </div>

        {/* 3-Tier Concise Grid */}
        <motion.div
          variants={stagger()}
          initial="hidden"
          whileInView="visible"
          viewport={inView}
          className="mt-8 grid gap-5 md:grid-cols-3"
        >
          {tiers.map((tier) => (
            <motion.div
              key={tier.id}
              variants={fadeUp}
              className="flex flex-col justify-between rounded-xl border border-ink-line/80 bg-ink-panel/70 p-5 backdrop-blur-sm hover:border-amber/30 transition-colors"
            >
              <div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: dotFor[tier.id] }}
                      aria-hidden="true"
                    />
                    <span className="font-mono text-10 font-semibold tracking-label uppercase text-cream/50">
                      Tier {tier.index} · {tier.medium}
                    </span>
                  </div>
                  <ProvenanceBadge value={tier.provenance} dark />
                </div>

                <h3 className="mt-3 font-display text-xl text-cream font-medium">
                  {tier.name}
                </h3>

                <p className="mt-2 font-mono text-[0.75rem] leading-relaxed text-khaki/70">
                  {tier.note}
                </p>
              </div>

              <div className="mt-5 border-t border-ink-line/60 pt-3 grid grid-cols-3 gap-2 text-left">
                <div>
                  <span className="font-mono text-[9px] uppercase tracking-label text-cream/40 block">
                    Bandwidth
                  </span>
                  <span className="font-display text-sm text-cream">{tier.bandwidth}</span>
                </div>
                <div>
                  <span className="font-mono text-[9px] uppercase tracking-label text-cream/40 block">
                    Latency
                  </span>
                  <span className="font-display text-sm text-cream">{tier.latency}</span>
                </div>
                <div>
                  <span className="font-mono text-[9px] uppercase tracking-label text-cream/40 block">
                    Capacity
                  </span>
                  <span className="font-mono text-[11px] text-cream/80">{tier.capacity}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Compact Physical Hardware Baseline Callout */}
        <div className="mt-8 rounded-lg border border-ink-line/60 bg-ink-soft/60 px-5 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="font-mono text-10 font-bold uppercase tracking-wider text-amber">
              Hardware Platform
            </span>
            <span className="text-cream/40">|</span>
            <span className="font-mono text-11 text-cream/90 font-medium">
              {harness.platform}
            </span>
          </div>
          <span className="font-mono text-10 text-khaki/60 max-w-[65ch]">
            {harness.detail}
          </span>
        </div>
      </div>
    </section>
  );
}
