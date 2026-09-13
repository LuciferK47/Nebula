import React from 'react';

interface LogoProps {
  tone?: 'ink' | 'cream';
  className?: string;
}

/** Small rounded-pill badge: swirl mark + wordmark. */
export function Logo({ tone = 'ink', className = '' }: LogoProps) {
  const shell =
  tone === 'ink' ? 'bg-ink text-cream' : 'bg-cream text-ink';
  const mark = tone === 'ink' ? '#f4f3ed' : '#141414';
  const markFill = tone === 'ink' ? '#141414' : '#f4f3ed';

  return (
    <a
      href="#top"
      className={`inline-flex items-center gap-2 rounded-full py-1.5 pl-1.5 pr-3.5 ${shell} ${className}`}
      aria-label="STRATA — home">
      
      <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <circle cx="12" cy="12" r="11" fill={mark} />
        <path
          d="M12 1a11 11 0 0 1 0 22 5.5 5.5 0 0 1 0-11 5.5 5.5 0 0 0 0-11Z"
          fill={markFill} />
        
        <circle cx="12" cy="6.5" r="1.7" fill={markFill} />
        <circle cx="12" cy="17.5" r="1.7" fill={mark} />
      </svg>
      <span className="font-mono text-11 font-semibold uppercase tracking-label">Strata</span>
    </a>);

}