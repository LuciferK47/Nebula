import type { Tier } from '../types/content';

// `provenance` badges the Bandwidth/Latency figures shown below, NOT the
// transfer mechanism — those two things differ for HBM and DRAM. The
// mechanism moving DRAM<->HBM data is genuinely measured (real
// torch.cuda.Stream async DMA, timed with CUDA events), but the 16 GB/s
// and ~100 ns figures themselves are memtier_moe/core/config.py's
// MemTierConfig defaults — the assumed capability of the link, not an
// achieved-bandwidth number independently benchmarked on this card. HBM
// has no transfer cost at all (a hit just executes), so its cells are
// "—" rather than a number either way. Both get 'modeled': neither cell
// is an empirical measurement, even though DRAM's underlying transfers
// are real.
export const tiers: Tier[] = [
{
  id: 'hbm',
  index: '00',
  name: 'HBM',
  medium: 'GPU VRAM — resident experts',
  capacity: 'the expert cache budget (200 MB–2.5 GB in our sweeps)',
  bandwidth: '— (already local)',
  latency: '— (already local)',
  provenance: 'modeled',
  note: 'A hit here means the expert’s weights are already on-GPU: the FFN just runs — measured real execution, but nothing to badge as bandwidth or latency since no transfer happens. This is why the capacity-cliff chart below is flat until the budget runs out.'
},
{
  id: 'dram',
  index: '01',
  name: 'DRAM',
  medium: 'Pinned host memory',
  capacity: 'configurable per run',
  bandwidth: '16 GB/s (assumed)',
  latency: '~100 ns (assumed)',
  provenance: 'modeled',
  note: 'The figures at right are config defaults, not an independently benchmarked achieved bandwidth. The transfer itself is real, though: a torch.cuda.Stream async DMA copy, overlapped with compute and timed with CUDA events — not a simulated hop. This is the hop the hybrid mode below mostly avoids by moving a 4 KB activation instead of the 16.5 MB expert.'
},
{
  id: 'cxl',
  index: '02',
  name: 'CXL',
  medium: 'Emulated CXL-attached memory',
  capacity: 'configurable per run',
  bandwidth: '8 GB/s (assumed)',
  latency: '350 ns (assumed)',
  provenance: 'modeled',
  note: 'The only emulated tier: a token-bucket bandwidth limiter plus a busy-wait latency injection, since no physical CXL hardware is attached. Latency contributes ~0.02% of the modeled cost — bandwidth dominates — and a 4×–16× sweep of the bandwidth assumption changes throughput but never which placement wins (see the sensitivity chart below).'
}];
