import React from 'react';
import { Kicker } from './Kicker';
import { Logo } from './Logo';
import { SwatchChips } from './SwatchChips';
import { WobbleRule } from './WobbleRule';
import { navLinks } from '../data/platforms';

export function SiteFooter() {
  return (
    <footer className="on-ink bg-ink pb-7">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <WobbleRule tone="dark" seed={200} />
        <div className="flex flex-col gap-8 py-9 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-5">
            <Logo tone="cream" />
            <SwatchChips colors={['#e4512b', '#e5b52f', '#8fa3b8']} width={16} />
          </div>
          <ul className="flex flex-wrap items-center gap-x-8 gap-y-3">
            {navLinks.map((link) =>
            <li key={link.href}>
                <a
                href={link.href}
                className="font-mono text-10 uppercase tracking-label text-cream/55 transition-colors duration-150 ease-out hover:text-amber">
                
                  {link.label}
                </a>
              </li>
            )}
          </ul>
        </div>

        <WobbleRule tone="dark" seed={210} />
        <div className="flex flex-col gap-3 pt-5 md:flex-row md:items-center md:justify-between">
          <Kicker className="text-cream/45">Strata is under active development</Kicker>
          <a
            href="#architecture"
            className="font-mono text-10 uppercase tracking-label text-cream/70 underline decoration-cream/30 underline-offset-4 transition-colors duration-150 ease-out hover:text-amber hover:decoration-amber">
            
            Learn more about what we are building
          </a>
        </div>
      </div>
    </footer>);

}