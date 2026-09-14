import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { WobbleRule } from './WobbleRule';
import { ProvenanceBadge } from './Provenance';
import { tiers } from '../data/tiers';
import { harness } from '../data/benchmarks';
import { fadeUp, inView, stagger } from '../utils/motion';
import { PackagingCrossSection } from './PackagingCrossSection';

const dotFor: Record<string, string> = {
  hbm: '#e4512b',
  dram: '#e5b52f',
  cxl: '#8fa3b8'
};

export function Architecture() {
  return (
    <section id="architecture" className="on-ink bg-ink py-20 md:py-28" aria-labelledby="arch-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <Kicker as="p" className="text-amber">
          02 — Architecture
        </Kicker>
        <h2
          id="arch-title"
          className="mt-5 max-w-[24ch] font-display text-[clamp(2.5rem,5.6vw,5rem)] leading-[0.92] text-cream">

          Three Tiers. <span className="italic">One Address Space.</span>
        </h2>
        <p className="mt-6 max-w-[62ch] font-mono text-[0.8rem] leading-relaxed text-khaki/80">
          Only the tier at the bottom is emulated — there is no CXL hardware attached to the test
          machine. HBM and DRAM are the physical GPU and host memory, moved between with real CUDA
          transfers.
        </p>

        <motion.ol
          variants={stagger()}
          initial="hidden"
          whileInView="visible"
          viewport={inView}
          className="mt-14 w-full">

          {tiers.map((tier, i) =>
          <motion.li key={tier.id} variants={fadeUp}>
              {i > 0 && <WobbleRule tone="dark" seed={50 + i} />}
              <div className="grid gap-6 py-7 md:grid-cols-[auto_minmax(0,1fr)] md:gap-10">
                <div className="flex items-start gap-4">
                  <span
                  className="flex h-12 w-12 items-center justify-center rounded-full bg-ink-panel font-mono text-11 font-semibold tracking-wide text-cream"
                  aria-hidden="true">

                    {tier.index}
                  </span>
                  <span
                  className="mt-5 h-[7px] w-[7px] shrink-0 rounded-full"
                  style={{ backgroundColor: dotFor[tier.id] }}
                  aria-hidden="true" />

                </div>

                <div>
                  <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
                    <h3 className="font-display text-[1.75rem] leading-none text-cream md:text-[2rem]">
                      {tier.name}
                    </h3>
                    <Kicker className="text-cream/45">{tier.medium}</Kicker>
                    <ProvenanceBadge value={tier.provenance} dark />
                  </div>

                  <p className="mt-3 max-w-[62ch] font-mono text-[0.78rem] leading-relaxed text-khaki/80">
                    {tier.note}
                  </p>

                  <dl className="mt-5 flex flex-wrap gap-x-10 gap-y-3">
                    <div>
                      <dt className="font-mono text-10 uppercase tracking-label text-cream/35">
                        Bandwidth
                      </dt>
                      <dd className="mt-1 font-display text-xl text-cream">{tier.bandwidth}</dd>
                    </div>
                    <div>
                      <dt className="font-mono text-10 uppercase tracking-label text-cream/35">
                        Latency
                      </dt>
                      <dd className="mt-1 font-display text-xl text-cream">{tier.latency}</dd>
                    </div>
                    <div>
                      <dt className="font-mono text-10 uppercase tracking-label text-cream/35">
                        Capacity
                      </dt>
                      <dd className="mt-1 font-mono text-11 text-cream/80">{tier.capacity}</dd>
                    </div>
                  </dl>
                </div>
              </div>
            </motion.li>
          )}
        </motion.ol>

        <PackagingCrossSection />

        <div className="mt-16 border-t border-ink-line pt-12">
          <Kicker as="p" className="text-cream/40">
            Where it ran
          </Kicker>
          <p className="mt-4 max-w-[48ch] font-display text-[1.6rem] leading-snug text-cream md:text-[2rem]">
            {harness.platform}
          </p>
          <p className="mt-5 max-w-[70ch] font-mono text-[0.78rem] leading-relaxed text-khaki/75">
            {harness.detail}
          </p>
        </div>
      </div>
    </section>);

}
