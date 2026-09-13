import React from 'react';
import { Kicker } from './Kicker';
import { Logo } from './Logo';
import { WobbleRule } from './WobbleRule';
import { navLinks } from '../data/platforms';

const TICKER =
'Astera Labs Connectivity Challenge 2026 · Track 03 — Memory Fabric · Finalist submission';

interface TopBarProps {
  showTicker?: boolean;
}

export function TopBar({ showTicker = true }: TopBarProps) {
  return (
    <>
      {showTicker ?
      <div id="top" className="overflow-hidden bg-ink py-2">
          <div className="marquee-track flex w-max">
            {[0, 1].map((copy) =>
          <div key={copy} className="flex shrink-0" aria-hidden={copy === 1}>
                {Array.from({ length: 4 }).map((_, i) =>
            <Kicker key={i} className="flex items-center gap-3 px-6 text-cream/70">
                    <span className="h-1 w-1 rounded-full bg-amber" aria-hidden="true" />
                    {TICKER}
                  </Kicker>
            )}
              </div>
          )}
          </div>
        </div> :

      <div id="top" className="h-1.5 w-full bg-ink" />
      }

      <header className="sticky top-0 z-40 bg-cream/90 backdrop-blur">
        <nav
          aria-label="Primary"
          className="mx-auto flex max-w-site items-center justify-between gap-6 px-5 py-3.5 md:px-8">
          
          <Logo />

          <ul className="hidden items-center gap-8 md:flex">
            {navLinks.map((link) =>
            <li key={link.href}>
                <a
                href={link.href}
                className="font-mono text-10 font-medium uppercase tracking-label text-ink/60 transition-colors duration-150 ease-out hover:text-ink">
                
                  {link.label}
                </a>
              </li>
            )}
          </ul>

          <a
            href="#audience"
            className="rounded-full border border-ink/25 px-4 py-2 font-mono text-10 font-medium uppercase tracking-label text-ink transition-colors duration-150 ease-out hover:border-ink hover:bg-ink hover:text-cream">
            
            Reproduction kit
          </a>
        </nav>
        <WobbleRule seed={11} />
      </header>
    </>);

}