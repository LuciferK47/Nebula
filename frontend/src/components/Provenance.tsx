import type { Provenance as ProvenanceType } from '../types/content';

export function ProvenanceBadge({
  value,
  className = '',
  dark = false,
}: {
  value?: ProvenanceType;
  className?: string;
  dark?: boolean;
}) {
  if (value !== 'measured') return null;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 font-mono text-10 font-semibold uppercase tracking-wider ${
        dark
          ? 'border border-emerald-500/40 bg-emerald-500/10 text-emerald-400'
          : 'border border-emerald-600/30 bg-emerald-600/10 text-emerald-700'
      } ${className}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" aria-hidden="true" />
      Verified
    </span>
  );
}
