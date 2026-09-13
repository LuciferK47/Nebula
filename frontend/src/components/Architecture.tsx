import React from 'react';
import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { WobbleRule } from './WobbleRule';
import { tiers } from '../data/tiers';
import { harness } from '../data/benchmarks';
import { fadeUp, inView, stagger } from '../utils/motion';

const TIER_ART = "/83347f14-4a63-4353-be11-17f741c2e148.jpg";


const DATACENTER = "/90148012-209c-4d54-a267-30cdecc4df05.jpg";


const dotFor: Record<string, string> = {
  hot: '#e4512b',
  warm: '#e5b52f',
  cold: '#8fa3b8'
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

        <div className="mt-14 grid gap-14 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)] lg:gap-20">
          <div className="lg:pt-2">
            <img
              src={TIER_ART}
              alt="Three fused glossy chrome forms stacked as a hierarchy, glowing hot orange, amber and cool blue against black."
              className="w-full max-w-[26rem]"
              loading="lazy" />
            
            <Kicker className="mt-4 block text-cream/40">
              fig. 03 — the stack, as one object
            </Kicker>
          </div>

          <motion.ol
            variants={stagger()}
            initial="hidden"
            whileInView="visible"
            viewport={inView}
            className="w-full">
            
            {tiers.map((tier, i) =>
            <motion.li key={tier.id} variants={fadeUp}>
                {i > 0 && <WobbleRule tone="dark" seed={50 + i} />}
                <div className="grid gap-6 py-7 md:grid-cols-[auto_minmax(0,1fr)] md:gap-10">
                  <div className="flex items-start gap-4">
                    {/* The cold tier sits on the exposed checkerboard: raw,
                       not resident, nothing mapped yet. */}
                    <span
                    className={`flex h-12 w-12 items-center justify-center rounded-full font-mono text-11 font-semibold tracking-wide ${
                    tier.id === 'cold' ?
                    'checker-dark checker-sm text-cream/70' :
                    'bg-ink-panel text-cream'}`
                    }
                    aria-hidden="true">
                    
                      {tier.index}
                    </span>
                    <span
                    className="mt-5 h-[7px] w-[7px] shrink-0 rounded-full"
                    style={{ backgroundColor: dotFor[tier.id] }}
                    aria-hidden="true" />
                  
                  </div>

                  <div>
                    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                      <h3 className="font-display text-[1.75rem] leading-none text-cream md:text-[2rem]">
                        {tier.name}
                      </h3>
                      <Kicker className="text-cream/45">{tier.medium}</Kicker>
                      {tier.id === 'cold' &&
                    <Kicker className="rounded-full border border-ink-line px-2.5 py-1 text-cream/50">
                          not yet resident
                        </Kicker>
                    }
                    </div>

                    <p className="mt-3 max-w-[62ch] font-mono text-[0.78rem] leading-relaxed text-khaki/80">
                      {tier.note}
                    </p>

                    <dl className="mt-5 flex flex-wrap gap-x-10 gap-y-3">
                      <div>
                        <dt className="font-mono text-10 uppercase tracking-label text-cream/35">
                          Fetch
                        </dt>
                        <dd className="mt-1 font-display text-xl text-cream">{tier.fetch}</dd>
                      </div>
                      <div>
                        <dt className="font-mono text-10 uppercase tracking-label text-cream/35">
                          Holds
                        </dt>
                        <dd className="mt-1 font-mono text-11 text-cream/80">{tier.capacity}</dd>
                      </div>
                      <div>
                        <dt className="font-mono text-10 uppercase tracking-label text-cream/35">
                          Residency rule
                        </dt>
                        <dd className="mt-1 font-mono text-11 text-cream/80">{tier.residency}</dd>
                      </div>
                    </dl>
                  </div>
                </div>
              </motion.li>
            )}
          </motion.ol>
        </div>

        <div className="mt-16 grid gap-10 border-t border-ink-line pt-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)] lg:gap-16">
          <div>
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
          <img
            src={DATACENTER}
            alt="Data center aisle of server racks and cable bundles lit in blue and violet ambient light."
            className="h-56 w-full object-cover lg:h-full"
            loading="lazy" />
          
        </div>
      </div>
    </section>);

}