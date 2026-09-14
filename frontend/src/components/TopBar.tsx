import { useEffect, useState } from 'react';
import { Logo } from './Logo';
import { WobbleRule } from './WobbleRule';
import { LiveModePill } from './LiveModePill';
import { navLinks } from '../data/platforms';

export function TopBar() {
  const [visible, setVisible] = useState(true);
  const [prevScrollY, setPrevScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      if (currentScrollY < 60) {
        setVisible(true);
      } else if (currentScrollY > prevScrollY && currentScrollY > 100) {
        // Scrolling DOWN -> collapse navbar
        setVisible(false);
      } else if (currentScrollY < prevScrollY) {
        // Scrolling UP -> reveal navbar
        setVisible(true);
      }
      setPrevScrollY(currentScrollY);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [prevScrollY]);

  return (
    <>
      <div id="top" className="h-1.5 w-full bg-ink" />

      <header
        className={`sticky top-0 z-40 bg-cream/95 backdrop-blur border-b border-ink/10 transition-transform duration-300 ease-in-out ${
          visible ? 'translate-y-0' : '-translate-y-full shadow-none'
        }`}
      >
        <nav
          aria-label="Primary"
          className="mx-auto flex max-w-site items-center justify-between gap-6 px-5 py-3 md:px-8"
        >
          <div className="flex items-center gap-3.5">
            <Logo />

            {/* Subtle separator */}
            <div className="h-4 w-px bg-ink/20" />

            {/* Official Astera Labs Logo */}
            <div
              className="flex items-center group cursor-pointer"
              title="Astera Labs · PCIe & CXL Connectivity Architecture"
            >
              <img
                src="/astera-labs-logo.png"
                alt="Astera Labs"
                className="h-6 w-auto object-contain transition-transform group-hover:scale-105"
              />
            </div>
          </div>

          <ul className="hidden items-center gap-7 md:flex">
            {navLinks.map((link) => (
              <li key={link.href}>
                <a
                  href={link.href}
                  className="font-mono text-xs font-semibold uppercase tracking-label text-ink/70 transition-colors duration-150 ease-out hover:text-ink hover:text-amber-700"
                >
                  {link.label}
                </a>
              </li>
            ))}
          </ul>

          <LiveModePill />
        </nav>
        <WobbleRule seed={11} />
      </header>
    </>
  );
}