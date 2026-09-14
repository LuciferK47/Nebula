import type { Mechanism } from '../types/content';

export const harness = {
  platform: 'CXL-Enabled Heterogeneous Memory Testbed',
  detail:
    'Evaluated under a constrained 6 GB accelerator memory envelope, demonstrating that disaggregated CXL tiering and asymmetric activation routing enable full-scale MoE inference without requiring multi-GPU enterprise clusters.',
};

export const oracleNote =
  'Hybrid mode does not attempt to swap 16.5 MB expert weights across the PCIe bus on every token miss. ' +
  'Instead, cold expert activations (only a few KB) route to host CPU/CXL memory for evaluation. ' +
  'Weights remain stationary in their tiers, eliminating bus thrashing and cache evictions.';

export const mechanisms: Mechanism[] = [
  {
    n: '01',
    title: 'Disaggregated CXL Far Memory Tier',
    body: 'Pools large MoE parameter sets across host DRAM and disaggregated CXL memory with calibrated bandwidth and latency modeling.',
  },
  {
    n: '02',
    title: 'Asymmetric Activation Offload',
    body: 'Routes lightweight token activations over PCIe to stationary weights rather than thrashing 16.5 MB expert weights into VRAM.',
  },
  {
    n: '03',
    title: 'Decayed-LFU with Active Pinning',
    body: 'Dynamically adapts expert residency to routing shifts over time, while pinning active layer experts against intra-layer eviction races.',
  },
  {
    n: '04',
    title: 'Decoupled Asynchronous CUDA DMA',
    body: 'DRAM-to-GPU transfers execute on a dedicated CUDA stream isolated from compute kernels, enabling speculative prefetching without stalls.',
  },
];
