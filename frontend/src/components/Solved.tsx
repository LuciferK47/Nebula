import { motion } from 'framer-motion';
import { CheckIcon } from 'lucide-react';
import { Kicker } from './Kicker';
import { WobbleRule } from './WobbleRule';
import { ProvenanceBadge } from './Provenance';
import { guarantees } from '../data/guarantees';
import { fadeUp, inView, stagger } from '../utils/motion';

export function Solved() {
  return (
    <section aria-labelledby="solved-title" className="bg-cream py-20 md:py-28">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="grid gap-12 md:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] md:gap-16">
          <div>
            <Kicker as="p" className="text-ink/45">
              01 — What&rsquo;s real
            </Kicker>
            <h2
              id="solved-title"
              className="mt-5 font-display text-[clamp(2.75rem,5vw,4.5rem)] leading-[0.92] text-ink">

              Measured, not claimed.
            </h2>
            <p className="mt-6 max-w-[34ch] font-mono text-[0.78rem] leading-relaxed text-ink/60">
              Every line below traces to a real run — including the two bugs an edge-case suite
              found and the negative result that made the cut.
            </p>
          </div>

          <motion.ul
            variants={stagger()}
            initial="hidden"
            whileInView="visible"
            viewport={inView}
            className="md:pt-2">
            
            {guarantees.map((item, i) =>
            <motion.li key={item.title} variants={fadeUp}>
                {i > 0 && <WobbleRule seed={40 + i} className="opacity-80" />}
                <div className="flex gap-5 py-6">
                  <CheckIcon
                  size={18}
                  strokeWidth={2.4}
                  className="mt-0.5 shrink-0 text-ink"
                  aria-hidden="true" />
                
                  <div>
                    <div className="flex flex-wrap items-center gap-3">
                      <h3 className="font-display text-2xl leading-snug text-ink md:text-[1.75rem]">
                        {item.title}
                      </h3>
                      <ProvenanceBadge value={item.provenance} />
                    </div>
                    <p className="mt-2 max-w-[58ch] font-mono text-[0.78rem] leading-relaxed text-ink/65">
                      {item.detail}
                    </p>
                  </div>
                </div>
              </motion.li>
            )}
          </motion.ul>
        </div>
      </div>
    </section>);

}