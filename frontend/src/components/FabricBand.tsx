import React from 'react';
import { Kicker } from './Kicker';

const MODULE_MACRO = "/4a83aaee-a8ea-4334-8a53-9830839fd6f9.jpg";


/** The deliberate switch from the warm-cream narrative half into the
 *  near-black technical half. Photography, not illustration. */
export function FabricBand() {
  return (
    <section aria-label="The fabric" className="relative bg-ink">
      <img
        src={MODULE_MACRO}
        alt="Macro photograph of an accelerator module's gold edge contacts and memory packages, lit in cool blue and violet light."
        className="h-56 w-full object-cover md:h-72"
        loading="lazy" />
      
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between px-5 pb-5 md:px-8 md:pb-7">
        <Kicker className="text-cream/75">
          The weights have to live somewhere. The fabric decides how fast they arrive.
        </Kicker>
        <Kicker className="hidden text-cream/40 md:block">fig. 02 — cxl 3.1 link</Kicker>
      </div>
    </section>);

}