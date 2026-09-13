export type TierId = 'hot' | 'warm' | 'cold';

export interface Tier {
  id: TierId;
  index: string;
  name: string;
  medium: string;
  capacity: string;
  fetch: string;
  residency: string;
  note: string;
}

export interface Guarantee {
  title: string;
  detail: string;
}

export interface BenchStat {
  value: string;
  label: string;
  baseline: string;
}

export interface BenchBar {
  name: string;
  tokensPerSecond: number;
  isSubject?: boolean;
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

export interface FieldNote {
  quote: string;
  author: string;
  role: string;
  context: string;
}

export interface Persona {
  audience: string;
  headline: string;
  body: string;
  bullets: string[];
  cta: string;
}