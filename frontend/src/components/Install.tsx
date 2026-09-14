import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { CommandBlock } from './CommandBlock';
import { SwatchChips } from './SwatchChips';
import { WobbleRule } from './WobbleRule';
import { installCommand, packages, usageSnippet } from '../data/packages';
import { fadeUp, inView, stagger } from '../utils/motion';

export function Install() {
  return (
    <section id="install" className="bg-cream py-20 md:py-28" aria-labelledby="install-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,32rem)] lg:gap-20">
          <div>
            <Kicker as="p" className="text-ink/45">
              07 — Install
            </Kicker>
            <h2
              id="install-title"
              className="mt-5 max-w-[16ch] font-display text-[clamp(2.75rem,6vw,5.25rem)] leading-[0.9] text-ink">

              Clone It. <span className="italic">Run It Yourself.</span>
            </h2>
            <div className="mt-9 flex flex-wrap items-center gap-4">
              <CommandBlock command={installCommand} />
              <SwatchChips colors={['#141414', '#e5b52f', '#77784f']} width={18} />
            </div>
            <Kicker as="p" className="mt-5 text-ink/40">
              Python 3.10+ · Torch 2.1+ · Not on PyPI yet
            </Kicker>
          </div>

          <pre
            aria-label="Usage example"
            className="overflow-x-auto rounded-2xl bg-ink p-6 font-mono text-[0.75rem] leading-relaxed text-cream md:p-7">
            
            <code>{usageSnippet}</code>
          </pre>
        </div>

        <motion.ul
          variants={stagger()}
          initial="hidden"
          whileInView="visible"
          viewport={inView}
          className="mt-16 grid gap-x-12 md:grid-cols-3">
          
          {packages.map((pkg, i) =>
          <motion.li key={pkg.name} variants={fadeUp}>
              <WobbleRule seed={120 + i} tone="ink" className="opacity-60" />
              <div className="py-7">
                <h3 className="font-mono text-11 font-semibold uppercase tracking-label text-ink">
                  {pkg.name}
                </h3>
                <p className="mt-3 min-h-[4.5rem] max-w-[38ch] font-mono text-[0.76rem] leading-relaxed text-ink/60">
                  {pkg.blurb}
                </p>
                <CommandBlock command={pkg.command} size="sm" className="mt-1" />
              </div>
            </motion.li>
          )}
        </motion.ul>
      </div>
    </section>);

}