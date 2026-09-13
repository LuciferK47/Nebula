import type { Tier } from '../types/content';

export const tiers: Tier[] = [
{
  id: 'hot',
  index: '00',
  name: 'Hot',
  medium: 'On-package HBM3e',
  capacity: '8–16 experts resident',
  fetch: '0.4 µs',
  residency: 'Predicted next two layers',
  note: 'Reserved for the experts the router is about to select. Nothing sits here on merit — only on prediction.'
},
{
  id: 'warm',
  index: '01',
  name: 'Warm',
  medium: 'CXL 3.1 attached DRAM',
  capacity: 'Full expert set, all layers',
  fetch: '2.1 µs',
  residency: 'Everything not hot',
  note: 'Zero-copy mapped. A promotion out of this tier is a descriptor handoff, not a memcpy.'
},
{
  id: 'cold',
  index: '02',
  name: 'Cold',
  medium: 'NVMe namespace',
  capacity: 'Every checkpoint revision',
  fetch: '180 µs',
  residency: 'Archival, first-touch only',
  note: 'Read once on the first token of an unseen routing regime, then promoted and never read again.'
}];