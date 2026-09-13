import React from 'react';
import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { WobbleRule } from './WobbleRule';
import { fieldNotes } from '../data/fieldNotes';
import { fadeUp, inView, stagger } from '../utils/motion';

export function FieldNotes() {
  const [featured, ...rest] = fieldNotes;

  return (
    <section aria-labelledby="notes-title" className="bg-offwhite py-20 md:py-28">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex flex-wrap items-baseline justify-between gap-4">
          <Kicker as="h2" id="notes-title" className="text-ink/45">
            05 — From the pilot deployments
          </Kicker>
          <Kicker className="text-ink/35">4 of 11 notes</Kicker>
        </div>
        <WobbleRule seed={140} className="mt-5" />

        <motion.div
          variants={stagger()}
          initial="hidden"
          whileInView="visible"
          viewport={inView}
          className="mt-10 grid gap-x-16 gap-y-10 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
          
          <motion.figure variants={fadeUp}>
            <Kicker className="text-clay">{featured.context}</Kicker>
            <blockquote className="mt-5 font-display text-[clamp(1.75rem,3.2vw,2.75rem)] leading-[1.12] text-ink">
              “{featured.quote}”
            </blockquote>
            <figcaption className="mt-6 font-mono text-11 uppercase tracking-wide text-ink/55">
              {featured.author} — {featured.role}
            </figcaption>
          </motion.figure>

          <div>
            {rest.map((note, i) =>
            <motion.figure key={note.author} variants={fadeUp}>
                {i > 0 && <WobbleRule seed={150 + i} />}
                <div className="py-6">
                  <Kicker className="text-ink/35">{note.context}</Kicker>
                  <blockquote className="mt-3 max-w-[52ch] font-mono text-[0.8rem] leading-relaxed text-ink/80">
                    “{note.quote}”
                  </blockquote>
                  <figcaption className="mt-3 font-mono text-10 uppercase tracking-label text-ink/45">
                    {note.author} — {note.role}
                  </figcaption>
                </div>
              </motion.figure>
            )}
          </div>
        </motion.div>
      </div>
    </section>);

}