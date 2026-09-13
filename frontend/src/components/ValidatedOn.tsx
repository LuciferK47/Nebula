import React from 'react';
import { Kicker } from './Kicker';
import { WobbleRule } from './WobbleRule';
import { platforms } from '../data/platforms';

export function ValidatedOn() {
  return (
    <section aria-label="Validated platforms" className="bg-cream pt-14">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <WobbleRule seed={21} />
        <div className="flex flex-col gap-5 py-6 md:flex-row md:items-center md:gap-10">
          <Kicker as="p" className="shrink-0 text-ink/45">
            Measured on
          </Kicker>
          <ul className="flex flex-wrap items-center gap-x-7 gap-y-3">
            {platforms.map((platform) =>
            <li
              key={platform}
              className="font-mono text-11 font-medium uppercase tracking-wide text-ink/80">
              
                {platform}
              </li>
            )}
          </ul>
        </div>
        <WobbleRule seed={34} />
      </div>
    </section>);

}