import { Logo } from './Logo';
import { WobbleRule } from './WobbleRule';
import { LiveModePill } from './LiveModePill';
import { navLinks } from '../data/platforms';

export function TopBar() {
  return (
    <>
      <div id="top" className="h-1.5 w-full bg-ink" />

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

          <LiveModePill />
        </nav>
        <WobbleRule seed={11} />
      </header>
    </>);

}