import React from 'react';
import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { PillButton } from './PillButton';
import { SwatchChips } from './SwatchChips';
import { fadeUp, stagger } from '../utils/motion';

const HERO_ART = "/c7a74e53-728f-4f05-821f-a0c2a8bfdc4f.jpg";


const tierLabels: {label: string;color: string;position: string;}[] = [
{ label: 'Tier 00 — HBM', color: '#e4512b', position: 'left-[5%] top-[24%]' },
{ label: 'Tier 01 — CXL', color: '#e5b52f', position: 'left-[5%] top-[44%]' },
{ label: 'Tier 02 — NVMe', color: '#8fa3b8', position: 'left-[5%] top-[64%]' }];


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
            Strata
          </Kicker>
          <SwatchChips colors={['#c8c8be']} width={26} />
        </motion.div>

        <motion.h1
          variants={fadeUp}
          className="mx-auto mt-6 max-w-[18ch] font-display text-[clamp(3rem,8.6vw,8.25rem)] font-normal leading-[0.88] tracking-[-0.01em] text-ink">
          
          Your Model.{' '}
          <span className="italic">Unstalled.</span>
        </motion.h1>

        <motion.p
          variants={fadeUp}
          className="mx-auto mt-8 max-w-[62ch] font-mono text-[0.8rem] leading-relaxed text-ink/70 md:text-[0.875rem]">
          
          A three-tier expert residency layer for mixture-of-experts inference — HBM, CXL, NVMe —
          that moves the weights across the fabric before the router asks for them.
        </motion.p>

        <motion.div
          variants={fadeUp}
          className="mt-10 flex flex-wrap items-center justify-center gap-3">
          
          <PillButton href="#benchmark">Read the benchmark</PillButton>
          <PillButton href="#architecture" variant="outline">
            View architecture
          </PillButton>
        </motion.div>
      </motion.div>

      {/* One rendered object, asymmetrically placed, floating on the exposed
           alpha-channel checkerboard: the asset is raw, still editable. */}
      <div className="relative mx-auto mt-10 max-w-site px-5 md:mt-4 md:px-8">
        <motion.figure
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1], delay: 0.1 }}
          className="relative ml-auto w-full max-w-[46rem] lg:mr-[6%]">
          
          <div className="checker">
            <img
              src={HERO_ART}
              alt="A single glossy liquid-chrome sculpture of three fused forms, tinted hot orange, amber and cool blue, representing the three memory tiers."
              className="block w-full" />
            
          </div>

          {tierLabels.map((tier) =>
          <span
            key={tier.label}
            className={`absolute hidden items-center gap-2 lg:flex ${tier.position}`}
            aria-hidden="true">
            
              <span
              className="h-[7px] w-[7px] rounded-full"
              style={{ backgroundColor: tier.color }} />
            
              <span className="font-mono text-10 font-medium uppercase tracking-label text-ink/70">
                {tier.label}
              </span>
            </span>
          )}

          <figcaption className="mt-3 flex items-center justify-between font-mono text-10 uppercase tracking-label text-ink/40">
            <span>fig. 01 — residency, rendered</span>
            <span>hot / warm / cold</span>
          </figcaption>
        </motion.figure>

        <SwatchChips
          colors={['#b4623c', '#e5b52f', '#141414', '#77784f']}
          className="absolute bottom-[12%] left-[4%] hidden rotate-[-90deg] lg:flex"
          width={16} />
        
      </div>
    </section>);

}