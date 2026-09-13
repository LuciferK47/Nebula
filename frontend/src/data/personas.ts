import type { Persona } from '../types/content';

export const personas: Persona[] = [
{
  audience: 'For judges',
  headline: 'The claim, in one page.',
  body:
  'What is new here is not a cache. It is the decision of when to move weight across the fabric, made against measured link headroom.',
  bullets: [
  'Novelty stated in a single paragraph',
  'Measured against an offline oracle, not a strawman',
  'Runs on the Interop Lab reference platform'],

  cta: 'Read the submission'
},
{
  audience: 'For technical reviewers',
  headline: 'Every number, reproducible.',
  body:
  'The harness, the baselines and the raw traces ship together. Nothing on this page is a figure you cannot regenerate.',
  bullets: [
  'Full harness plus the Belady oracle baseline',
  'Ablations for all seven mechanisms',
  'Raw decode traces, 41 GB, downloadable'],

  cta: 'Open the reproduction kit'
},
{
  audience: 'For everyone else',
  headline: 'Why memory is the wall.',
  body:
  'A mixture-of-experts model only uses a sliver of itself per token — but you still have to keep all of it somewhere fast.',
  bullets: [
  'What an expert actually is',
  'Why the weights do not fit',
  'Ninety seconds, no mathematics'],

  cta: 'Watch the explainer'
}];