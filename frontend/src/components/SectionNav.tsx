import React, { useEffect, useState } from 'react';
import { ChevronDown, ArrowUp } from 'lucide-react';

export interface SectionMeta {
  id: string;
  num: string;
  name: string;
  href: string;
}

export const SECTIONS: SectionMeta[] = [
  { id: 'hero', num: '00', name: 'Overview', href: '#hero' },
  { id: 'architecture', num: '01', name: 'Architecture', href: '#architecture' },
  { id: 'explorer', num: '02', name: 'Interactive Hierarchy', href: '#explorer' },
  { id: 'benchmark', num: '03', name: 'Empirical Benchmarks', href: '#benchmark' },
  { id: 'limits', num: '04', name: 'Robustness & Limits', href: '#limits' },
  { id: 'live-lab', num: '05', name: 'Live Hardware Studio', href: '#live-lab' },
  { id: 'install', num: '06', name: 'Reproduction & Deploy', href: '#install' },
];

/**
 * Header Arrow: Compact down-arrow jump button placed next to section title/kicker
 */
export function SectionHeaderArrow({
  nextId,
  nextNum,
  nextLabel,
  isDark = true,
}: {
  nextId: string;
  nextNum: string;
  nextLabel: string;
  isDark?: boolean;
}) {
  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    const target = document.getElementById(nextId);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <a
      href={`#${nextId}`}
      onClick={handleClick}
      title={`Jump to Section ${nextNum} — ${nextLabel}`}
      aria-label={`Jump to Section ${nextNum}`}
      className={`group flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-mono font-medium transition-all duration-200 border ${
        isDark
          ? 'border-white/15 bg-white/5 text-cream/75 hover:border-amber/60 hover:bg-amber/10 hover:text-amber'
          : 'border-ink/15 bg-ink/5 text-ink/75 hover:border-amber-600/50 hover:bg-amber-50 hover:text-amber-800'
      }`}
    >
      <span>Next: {nextNum}</span>
      <ChevronDown className="h-3.5 w-3.5 transition-transform duration-200 group-hover:translate-y-0.5 text-amber" />
    </a>
  );
}

/**
 * Bottom Jump: Prominent transition button at the end of a section
 */
export function SectionBottomJump({
  nextId,
  nextNum,
  nextLabel,
  isDark = true,
}: {
  nextId: string;
  nextNum: string;
  nextLabel: string;
  isDark?: boolean;
}) {
  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    const target = document.getElementById(nextId);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className={`mt-12 pt-6 border-t flex justify-center ${isDark ? 'border-white/10' : 'border-ink/10'}`}>
      <a
        href={`#${nextId}`}
        onClick={handleClick}
        className={`group inline-flex items-center gap-2.5 rounded-full px-5 py-2.5 font-mono text-xs font-medium transition-all duration-200 border shadow-sm ${
          isDark
            ? 'border-white/15 bg-white/5 text-cream/80 hover:border-amber/60 hover:bg-amber/10 hover:text-amber shadow-black/20'
            : 'border-ink/15 bg-white text-ink/80 hover:border-amber-600/50 hover:bg-amber-50 hover:text-amber-800 shadow-ink/5'
        }`}
      >
        <span>Proceed to {nextNum} — {nextLabel}</span>
        <div
          className={`flex h-5 w-5 items-center justify-center rounded-full transition-transform duration-200 group-hover:translate-y-0.5 ${
            isDark ? 'bg-amber/20 text-amber' : 'bg-amber-100 text-amber-800'
          }`}
        >
          <ChevronDown className="h-3.5 w-3.5 animate-bounce" />
        </div>
      </a>
    </div>
  );
}

/**
 * Floating Section Navigation Pill (Bottom-Right corner)
 */
export function FloatingSectionNav() {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      setVisible(scrollY > 200);

      const sectionElements = SECTIONS.map((s) => ({
        ...s,
        el: s.id === 'hero' ? document.querySelector('section') : document.getElementById(s.id),
      }));

      const scrollPosition = scrollY + window.innerHeight * 0.35;

      for (let i = sectionElements.length - 1; i >= 0; i--) {
        const item = sectionElements[i];
        if (item.el) {
          const top = item.el.offsetTop;
          if (scrollPosition >= top) {
            setCurrentIdx(i);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  if (!visible) return null;

  const isLast = currentIdx >= SECTIONS.length - 1;
  const nextSection = isLast ? SECTIONS[0] : SECTIONS[currentIdx + 1];

  const handleJump = () => {
    if (isLast) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      const el = document.getElementById(nextSection.id);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <button
      type="button"
      onClick={handleJump}
      title={isLast ? 'Back to Top' : `Jump to ${nextSection.num} — ${nextSection.name}`}
      className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full border border-amber/40 bg-[#0f1015]/90 px-3.5 py-2 font-mono text-[11px] font-semibold text-cream shadow-2xl backdrop-blur-md hover:border-amber hover:bg-amber hover:text-ink transition-all duration-200 group active:scale-95"
    >
      <span className="text-amber group-hover:text-ink font-bold">
        {isLast ? '06' : SECTIONS[currentIdx].num}
      </span>
      <span className="text-white/40 group-hover:text-ink/60">→</span>
      <span className="group-hover:text-ink">
        {isLast ? 'Top' : nextSection.num}
      </span>
      <div className="flex h-5 w-5 items-center justify-center rounded-full bg-amber/20 group-hover:bg-ink group-hover:text-amber text-amber transition-colors">
        {isLast ? (
          <ArrowUp className="h-3 w-3" />
        ) : (
          <ChevronDown className="h-3 w-3 animate-bounce" />
        )}
      </div>
    </button>
  );
}
