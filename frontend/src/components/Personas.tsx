import React from 'react';
import { motion } from 'framer-motion';
import { PlusIcon } from 'lucide-react';
import { Kicker } from './Kicker';
import { PillButton } from './PillButton';
import { personas } from '../data/personas';
import { fadeUp, inView, stagger } from '../utils/motion';

export function Personas() {
  return (
    <section id="audience" className="on-ink bg-ink py-20 md:py-28" aria-labelledby="audience-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <Kicker as="p" className="text-amber">
          06 — Same system, three readings
        </Kicker>
        <h2
          id="audience-title"
          className="mt-5 max-w-[22ch] font-display text-[clamp(2.5rem,5.4vw,4.75rem)] leading-[0.92] text-cream">
          
          Pick Your <span className="italic">Depth.</span>
        </h2>

        <motion.ul
          variants={stagger()}
          initial="hidden"
          whileInView="visible"
          viewport={inView}
          className="mt-14 grid gap-px bg-ink-line md:grid-cols-3">
          
          {personas.map((persona, i) =>
          <motion.li
            key={persona.audience}
            variants={fadeUp}
            className="flex flex-col bg-ink p-7 md:p-8">
            
              <Kicker className="text-cream/40">{persona.audience}</Kicker>
              <h3 className="mt-5 font-display text-[1.75rem] leading-snug text-cream md:text-[2rem]">
                {persona.headline}
              </h3>
              <p className="mt-4 font-mono text-[0.78rem] leading-relaxed text-khaki/80">
                {persona.body}
              </p>
              <ul className="mt-6 space-y-3">
                {persona.bullets.map((bullet) =>
              <li key={bullet} className="flex gap-3">
                    <PlusIcon
                  size={13}
                  strokeWidth={2}
                  className="mt-[3px] shrink-0 text-amber"
                  aria-hidden="true" />
                
                    <span className="font-mono text-11 leading-relaxed text-cream/70">
                      {bullet}
                    </span>
                  </li>
              )}
              </ul>
              <div className="mt-auto pt-9">
                <PillButton
                href="#top"
                variant={i === 1 ? 'solid-ink' : 'outline-ink'}
                className="w-full md:w-auto">
                
                  {persona.cta}
                </PillButton>
              </div>
            </motion.li>
          )}
        </motion.ul>
      </div>
    </section>);

}