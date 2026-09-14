
interface LogoProps {
  tone?: 'ink' | 'cream';
  className?: string;
}

/** Small rounded-pill badge: swirl mark + wordmark. */
export function Logo({ tone = 'ink', className = '' }: LogoProps) {
  const shell =
  tone === 'ink' ? 'bg-ink text-cream' : 'bg-cream text-ink';
  const mark = tone === 'ink' ? '#f4f3ed' : '#141414';

  return (
    <a
      href="#top"
      className={`inline-flex items-center gap-2 rounded-full py-1.5 pl-1.5 pr-3.5 ${shell} ${className}`}
      aria-label="MemTier-MoE — home">

      {/* Three stacked tier bars — the same hot/warm/cold mark used
         throughout the site for HBM/DRAM/CXL. */}
      <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <circle cx="12" cy="12" r="11" fill={mark} />
        <rect x="6" y="6.2" width="12" height="3" rx="1" fill="#e4512b" />
        <rect x="6" y="10.5" width="12" height="3" rx="1" fill="#e5b52f" />
        <rect x="6" y="14.8" width="12" height="3" rx="1" fill="#8fa3b8" />
      </svg>
      <span className="font-mono text-11 font-semibold uppercase tracking-label">Nebula</span>
    </a>);

}