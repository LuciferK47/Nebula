import type { Tier } from '../types/content';

export const tiers: Tier[] = [
  {
    id: 'hbm',
    index: '00',
    name: 'GPU VRAM',
    medium: 'High-Speed Accelerator Memory',
    capacity: 'Configurable cache budget (200 MB – 2.5 GB)',
    bandwidth: '> 192 GB/s',
    latency: '< 10 ns',
    provenance: 'measured',
    note: 'Primary fast tier holding hot expert weights directly in GPU VRAM. Zero PCIe bus transfer overhead during forward execution.',
  },
  {
    id: 'dram',
    index: '01',
    name: 'Host DRAM',
    medium: 'Pinned System Memory',
    capacity: '16 GB Host Memory Pool',
    bandwidth: '32–64 GB/s (PCIe Gen4/5)',
    latency: '~100 ns',
    provenance: 'measured',
    note: 'Warm tier holding secondary experts. Serves as staging for async CUDA DMA transfers and hosts CPU activation computation in hybrid mode.',
  },
  {
    id: 'cxl',
    index: '02',
    name: 'CXL Far Memory',
    medium: 'Disaggregated Coherent Memory Pool',
    capacity: 'Configurable pooled capacity',
    bandwidth: '8–32 GB/s',
    latency: '~350 ns',
    provenance: 'measured',
    note: 'Disaggregated memory tier enabling large MoE parameter offloading without host OS page thrashing. Rate-limited and latency-calibrated for evaluation.',
  },
];
