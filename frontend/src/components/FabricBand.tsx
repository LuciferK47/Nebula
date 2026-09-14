import { Kicker } from './Kicker';

/** The deliberate switch from the warm-cream narrative half into the
 *  near-black technical half. Used to be a stock macro photo of PCB gold
 *  contacts — replaced with the same hairline-grid language the rest of
 *  the site's diagrams use, since a stock photo of an unrelated board
 *  wasn't a real representation of anything this project built. */
export function FabricBand() {
  return (
    <section aria-label="Section divider" className="relative flex h-40 items-end overflow-hidden bg-ink md:h-48">
      <svg
        className="pointer-events-none absolute inset-0 h-full w-full opacity-[0.14]"
        aria-hidden="true">

        <defs>
          <pattern id="fabric-grid" width="28" height="28" patternUnits="userSpaceOnUse">
            <path d="M 28 0 L 0 0 0 28" fill="none" stroke="#f4f3ed" strokeWidth={0.75} />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#fabric-grid)" />
      </svg>
      <div className="pointer-events-none relative flex w-full items-end justify-between px-5 pb-5 md:px-8 md:pb-7">
        <Kicker className="max-w-[38ch] text-cream/75">
          The weights have to live somewhere. This is how the fabric decides how fast they arrive.
        </Kicker>
        <Kicker className="hidden text-cream/40 md:block">fig. 02 — the boundary</Kicker>
      </div>
    </section>);

}
