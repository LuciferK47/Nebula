import type { Provenance as ProvenanceType } from '../types/content';

/**
 * ProvenanceBadge component - returns null as requested by user to remove all tags.
 */
export function ProvenanceBadge(_props: {
  value?: ProvenanceType;
  className?: string;
  dark?: boolean;
}) {
  return null;
}
