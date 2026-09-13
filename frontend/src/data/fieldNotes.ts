import type { FieldNote } from '../types/content';

export const fieldNotes: FieldNote[] = [
{
  quote:
  'Took our replica count from nine down to five with no output diff. The trace replay is the part that actually got it past our SREs.',
  author: 'K. Ostrowski',
  role: 'Platform Engineer',
  context: 'PILOT · 5-NODE CLUSTER'
},
{
  quote:
  'We had written off expert offload after the LRU experiments. The scheduling piece is what everyone else skipped.',
  author: 'D. Raghunathan',
  role: 'Inference Lead',
  context: 'PILOT · CXL TESTBED'
},
{
  quote: 'I ran the oracle baseline myself expecting it to win. It did not.',
  author: 'M. Alvarez',
  role: 'Performance Engineer',
  context: 'REPRODUCTION KIT'
},
{
  quote:
  'First offload layer I have used where the p99 did not quietly fall apart at 400 concurrent streams.',
  author: 'S. Ilves',
  role: 'Staff Engineer, Serving',
  context: 'PILOT · 8×H100'
}];