import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { PillButton } from './PillButton';
import { SwatchChips } from './SwatchChips';
import { HeroChip } from './HeroChip';
import { fadeUp, stagger } from '../utils/motion';

export function Hero() {
  return (
    <section className="relative overflow-hidden bg-cream pb-4 pt-16 md:pt-24">
      <SwatchChips
        colors={['#141414', '#77784f', '#e5b52f']}
        className="absolute left-[7%] top-[13%] rotate-[-8deg]" />

      <SwatchChips
        colors={['#e5b52f', '#141414']}
        className="absolute right-[9%] top-[9%] rotate-[12deg]"
        width={18} />


      <motion.div
        variants={stagger()}
        initial="hidden"
        animate="visible"
        className="mx-auto max-w-site px-5 text-center md:px-8">

        <motion.div variants={fadeUp} className="flex items-center justify-center gap-3">
          <SwatchChips colors={['#c8c8be']} width={26} />
          <Kicker as="p" className="text-ink/70">
            MemTier-MoE
          </Kicker>
          <SwatchChips colors={['#c8c8be']} width={26} />
        </motion.div>

        <motion.h1
          variants={fadeUp}
          className="mx-auto mt-6 max-w-[20ch] font-display text-[clamp(2.75rem,7.6vw,7.25rem)] font-normal leading-[0.9] tracking-[-0.01em] text-ink">

          A 14B MoE, decoding{' '}
          <span className="italic">on a 6 GB laptop GPU.</span>
        </motion.h1>

        <motion.p
          variants={fadeUp}
          className="mx-auto mt-8 max-w-[62ch] font-mono text-[0.8rem] leading-relaxed text-ink/70 md:text-[0.875rem]">

          A three-tier residency layer for Mixture-of-Experts inference — HBM, DRAM, CXL — that
          keeps a model too large for the card resident by moving the right expert at the right
          time, and moving activations instead of weights whenever it can.
        </motion.p>

        <motion.div
          variants={fadeUp}
          className="mt-10 flex flex-wrap items-center justify-center gap-3">

          <PillButton href="#benchmark">See the measured results</PillButton>
          <PillButton href="#explorer" variant="outline">
            Explore the memory hierarchy
          </PillButton>
        </motion.div>
      </motion.div>

      <div className="relative mx-auto mt-10 max-w-site px-5 md:mt-6 md:px-8">
        <HeroChip />
      </div>
    </section>);

}
