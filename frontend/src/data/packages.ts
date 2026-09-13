import type { PackageCard } from '../types/content';

export const installCommand = 'pip install strata-moe';

export const packages: PackageCard[] = [
{
  name: 'strata.residency',
  command: 'pip install strata-moe',
  blurb: 'The tiered residency manager and the MoE block wrapper. Two lines at your model definition.'
},
{
  name: 'strata.cxl',
  command: 'pip install strata-moe[cxl]',
  blurb: 'Zero-copy CXL 3.1 mapping backend. Falls back to PCIe peer-to-peer where no host controller is present.'
},
{
  name: 'strata.bench',
  command: 'pip install strata-bench',
  blurb: 'The harness, the Belady oracle baseline, and every trace behind the numbers on this page.'
}];


export const usageSnippet = `from strata import TieredResidency

model.layers.mlp = TieredResidency.wrap(
    model.layers.mlp,
    tiers=("hbm", "cxl", "nvme"),
    lookahead=2,
)`;