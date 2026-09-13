import type { Guarantee } from '../types/content';

export const guarantees: Guarantee[] = [
{
  title: 'Bit-exact outputs.',
  detail:
  'No quantization, no expert dropping, no approximate routing. Logits match the unmodified baseline to the last bit.'
},
{
  title: 'No host paging, ever.',
  detail:
  'A cold miss executes in place off the warm tier. The kernel never takes a page fault on the decode path.'
},
{
  title: 'Deterministic replay.',
  detail:
  'Every promotion, demotion and fault is written to a trace you can replay step-for-step on another machine.'
},
{
  title: 'Drop-in at the block.',
  detail:
  'One wrapper around your existing MoE layer. No kernel rewrites, no custom allocator, no fork of the serving stack.'
}];