import type { Provenance as ProvenanceType } from '../types/content';

/**
 * The visible marker distinguishing a measured number from a modeled one —
 * the project's own methodology notes (Nebula/GPU_RUNBOOK.md) insist a
 * closed-form/analytical figure must never share an axis or a claim with a
 * measured wall-clock one. This is that rule made visible rather than a
 * convention someone has to remember to follow.
 */
export function ProvenanceBadge({ value, className = '', dark = false }: { value: ProvenanceType; className?: string; dark?: boolean }) {
  const isMeasured = value === 'measured';
  const base = 'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-mono text-10 font-medium uppercase tracking-label';
  const tone = isMeasured ?
  dark ? 'border border-olive/40 bg-olive/10 text-olive' : 'border border-olive/30 bg-olive/10 text-olive' :
  dark ? 'checker-dark checker-sm text-cream/70' : 'checker checker-sm text-ink/60';

  return (
    <span className={`${base} ${tone} ${className}`}>
      <span
        className={`h-1.5 w-1.5 rounded-full ${isMeasured ? 'bg-olive' : 'bg-current opacity-50'}`}
        aria-hidden="true" />

      {isMeasured ? 'Measured' : 'Modeled'}
    </span>);

}
