export type TierId = 'hbm' | 'dram' | 'cxl';

/** How a number on this page was produced — surfaced as a visible badge
 * everywhere a metric is shown, per the project's own methodology notes
 * (Nebula/GPU_RUNBOOK.md): a measured wall-clock number must never share
 * an axis or a claim with a modeled/analytical one. */
export type Provenance = 'measured' | 'modeled';

export interface Tier {
  id: TierId;
  index: string;
  name: string;
  medium: string;
  capacity: string;
  bandwidth: string;
  latency: string;
  provenance: Provenance;
  note: string;
}

export interface Guarantee {
  title: string;
  detail: string;
  provenance: Provenance;
}

export interface Mechanism {
  n: string;
  title: string;
  body: string;
}

export interface PackageCard {
  name: string;
  command: string;
  blurb: string;
}
