import React from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Kicker } from './Kicker';
import { WobbleCircle, WobbleRule } from './WobbleRule';
import { benchBars, benchStats, mechanisms, oracleNote } from '../data/benchmarks';
import { expo, fadeUp, inView, stagger } from '../utils/motion';

const SCALE_MAX = 48;

export function Benchmark() {
  const reduced = useReducedMotion();

  return (
    <section id="benchmark" className="on-ink bg-ink-soft pb-20 pt-4 md:pb-28" aria-labelledby="bench-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="grid gap-10 border-t border-ink-line pt-16 lg:grid-cols-[minmax(0,1fr)_minmax(0,34rem)] lg:gap-20">
          <div>
            <Kicker as="p" className="text-amber">
              03 — Benchmark
            </Kicker>
            <h2
              id="bench-title"
              className="mt-5 max-w-[16ch] font-display text-[clamp(2.75rem,6.4vw,5.75rem)] leading-[0.9] text-cream">
              
              It Beats The <span className="italic">Oracle.</span>
            </h2>
          </div>
          <p className="max-w-[68ch] font-mono text-[0.8rem] leading-relaxed text-khaki/85 lg:pt-8">
            {oracleNote}
          </p>
        </div>

        {/* Numbered stat cards — the metric carries the weight, the card just holds it. */}
        <motion.ul
          variants={stagger()}
          initial="hidden"
          whileInView="visible"
          viewport={inView}
          className="mt-14 grid gap-px border border-ink-line bg-ink-line sm:grid-cols-2 lg:grid-cols-4">
          
          {benchStats.map((stat, i) =>
          <motion.li key={stat.label} variants={fadeUp} className="bg-ink p-6 md:p-7">
              <WobbleCircle tone="dark" size={34} seed={9 + i}>
                <span className="font-mono text-10 font-medium tracking-wide text-amber">
                  {String(i + 1).padStart(2, '0')}
                </span>
              </WobbleCircle>
              <p className="mt-6 font-display text-[clamp(2.5rem,4vw,3.5rem)] leading-none text-cream">
                {stat.value}
              </p>
              <h3 className="mt-4 font-mono text-11 font-medium uppercase tracking-label text-cream/85">
                {stat.label}
              </h3>
              <p className="mt-2 font-mono text-10 leading-relaxed text-khaki/60">
                {stat.baseline}
              </p>
            </motion.li>
          )}
        </motion.ul>

        {/* Head-to-head */}
        <div className="mt-16 grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,24rem)] lg:gap-20">
          <div>
            <Kicker as="h3" className="text-cream/40">
              Decode throughput — tokens/sec/replica
            </Kicker>
            <ul className="mt-7 space-y-6">
              {benchBars.map((bar, i) =>
              <li key={bar.name}>
                  <div className="flex items-baseline justify-between gap-4">
                    <span
                    className={`font-mono text-11 uppercase tracking-wide ${
                    bar.isSubject ? 'text-amber' : 'text-cream/60'}`
                    }>
                    
                      {bar.name}
                    </span>
                    <span
                    className={`font-display text-xl ${
                    bar.isSubject ? 'text-amber' : 'text-cream/70'}`
                    }>
                    
                      {bar.tokensPerSecond.toFixed(1)}
                    </span>
                  </div>
                  <div className="mt-2 h-[10px] w-full bg-ink-panel">
                    <motion.div
                    className={`h-full ${bar.isSubject ? 'bg-amber' : 'bg-khaki/35'}`}
                    initial={reduced ? undefined : { scaleX: 0 }}
                    whileInView={reduced ? undefined : { scaleX: 1 }}
                    viewport={inView}
                    transition={{ duration: 0.3, ease: expo, delay: 0.05 * i }}
                    style={{
                      width: `${bar.tokensPerSecond / SCALE_MAX * 100}%`,
                      transformOrigin: 'left'
                    }} />
                  
                  </div>
                </li>
              )}
            </ul>
          </div>

          <div className="lg:pt-9">
            <WobbleRule tone="dark" seed={77} />
            <p className="py-6 font-mono text-[0.78rem] leading-relaxed text-khaki/75">
              The oracle is given the entire future routing trace in advance and still loses, because
              it fetches on demand at the moment of perfect knowledge. Knowing which expert is next
              is worth nothing if the link is busy when you need it.
            </p>
            <WobbleRule tone="dark" seed={88} />
          </div>
        </div>

        {/* The seven mechanisms */}
        <div className="mt-20">
          <Kicker as="h3" className="text-cream/40">
            How — seven mechanisms
          </Kicker>
          <motion.ol
            variants={stagger()}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.1 }}
            className="mt-8 grid gap-x-16 md:grid-cols-2">
            
            {mechanisms.map((m, i) =>
            <motion.li key={m.n} variants={fadeUp}>
                <WobbleRule tone="dark" seed={100 + i} />
                <div className="flex gap-5 py-6">
                  <WobbleCircle tone="dark" size={40} seed={30 + i}>
                    <span className="font-mono text-11 font-medium text-amber">{m.n}</span>
                  </WobbleCircle>
                  <div>
                    <h4 className="font-display text-[1.35rem] leading-snug text-cream">
                      {m.title}
                    </h4>
                    <p className="mt-2 font-mono text-[0.76rem] leading-relaxed text-khaki/75">
                      {m.body}
                    </p>
                  </div>
                </div>
              </motion.li>
            )}
          </motion.ol>
        </div>
      </div>
    </section>);

}