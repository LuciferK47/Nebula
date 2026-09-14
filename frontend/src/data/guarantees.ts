import type { Guarantee } from '../types/content';

// Renamed in spirit from "guarantees" to "what's actually true" — the
// previous copy claimed "bit-exact outputs" and "deterministic replay",
// both contradicted by the project's own S10 edge-case suite
// (results/scenarios/s10.json records real failures). These four are
// what the measured data and the test suite actually support.

export const guarantees: Guarantee[] = [
{
  title: 'Real transfers, not a simulation.',
  detail:
  'DRAM→HBM copies are genuine torch.cuda.Stream async DMA, timed with CUDA events. Only the CXL tier (no hardware attached) is emulated — and a 16× bandwidth sweep shows placement never depends on that emulation’s calibration.',
  provenance: 'measured'
},
{
  title: 'Overlap you can see in a trace.',
  detail:
  'Compute and the next transfer run on separate CUDA streams. verify_overlap.py exports a Chrome trace and CUDA-event deltas showing the overlap directly, not inferred from a throughput number.',
  provenance: 'measured'
},
{
  title: 'A negative result, shown anyway.',
  detail:
  'Adding lookahead prefetching on top of hybrid mode makes it slower, not faster (7.23 → 4.94 tok/s at a 600 MB budget) — it re-introduces the transfers hybrid mode exists to avoid. We show that chart, not just the ones that flatter the system.',
  provenance: 'measured'
},
{
  title: 'Edge cases are tested, and failures are public.',
  detail:
  'A 10-case robustness suite deliberately tries to break the system. It found two real bugs — a crash in the DRAM staging hop, and a silent-weight-loss path under an oversubscribed budget — both now fixed, both disclosed here rather than quietly dropped from the results.',
  provenance: 'measured'
}];
