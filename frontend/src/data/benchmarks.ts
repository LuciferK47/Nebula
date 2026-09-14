import type { Mechanism } from '../types/content';

// The stat cards and the throughput bars are NOT hardcoded here — they are
// derived at render time in the Benchmark component from
// src/lib/results.ts, which reads the actual synced JSON. That is
// deliberate: a hand-typed number is exactly how this page ended up with
// "94.2% hit rate" and "1.84x" in an earlier draft, neither of which
// existed in any measured file.

export const harness = {
  platform: 'One RTX 4050 Laptop GPU. Nothing exotic.',
  detail:
  'Every number on this page was measured on a single NVIDIA GeForce RTX 4050 Laptop GPU ' +
  '(6.0 GB GDDR6, 96-bit bus), 16 GB host RAM, Windows 11, CUDA 12.4, PyTorch 2.6.0 — the ' +
  'kind of machine a student or an indie researcher actually owns, not a rented 8-GPU node.'
};

export const oracleNote =
'Hybrid mode doesn’t beat weight-swapping by fetching faster — it beats it by not fetching ' +
'the 16.5 MB expert at all. A cold expert’s activation (a few KB) goes to the CPU, gets computed ' +
'there, and comes back. The weight never moves, so there is nothing to evict and nothing to wait on.';

export const mechanisms: Mechanism[] = [
{
  n: '01',
  title: 'Three real tiers, one emulated link',
  body: 'HBM and DRAM are physically measured: real CUDA async DMA, real CPU compute, real overlap timing via CUDA events. Only the CXL tier is software-emulated — there is no CXL hardware attached.'
},
{
  n: '02',
  title: 'A sensitivity sweep instead of a simulator',
  body: 'Rather than validate the CXL model against DRAMSim3/gem5, we swept its bandwidth assumption 4–64 GB/s and disabled emulation entirely. Placement never changed. That bounds sensitivity to the whole plausible range, not one simulator’s assumptions.'
},
{
  n: '03',
  title: 'Activation offload, not weight swapping',
  body: 'When an expert is cold, hybrid mode ships its activation to the CPU and computes there — instead of promoting the 16.5 MB expert into HBM and evicting something else to make room.'
},
{
  n: '04',
  title: 'Decayed-LFU eviction with pinning',
  body: 'Expert popularity decays over time so the cache adapts to a shifting routing distribution, and a layer’s active top-k experts are pinned against mutual eviction while they’re needed.'
},
{
  n: '05',
  title: 'Router-predictive prefetching',
  body: 'The next layer’s gate is peeked before it is needed, and a speculative fetch is issued for experts the router is likely to pick — gated adaptively so it doesn’t thrash a tight budget.'
},
{
  n: '06',
  title: 'A dedicated transfer stream',
  body: 'DRAM→HBM copies run on their own CUDA stream, separate from the compute stream, so a prefetch can genuinely overlap with the current layer’s FFN instead of serializing behind it.'
},
{
  n: '07',
  title: 'Edge cases, reported honestly',
  body: 'A 10-case robustness suite deliberately tries to break the system — degenerate budgets, VRAM over-commit, prefetch abuse. It has already found and fixed two real bugs; see “What we found” below.'
}];
