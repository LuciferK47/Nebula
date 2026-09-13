import type { BenchBar, BenchStat, Mechanism } from '../types/content';

export const benchStats: BenchStat[] = [
{
  value: '94.2%',
  label: 'Hot-tier hit rate',
  baseline: 'Offline Belady oracle: 91.7%'
},
{
  value: '1.84×',
  label: 'Decode throughput',
  baseline: 'vs. HBM-only offload at equal memory'
},
{
  value: '0.9 ms',
  label: 'p99 expert-fault stall',
  baseline: 'Down from 14.3 ms unprefetched'
},
{
  value: '61%',
  label: 'Less HBM per replica',
  baseline: 'Mixtral 8x7B served from 24 GB'
}];


export const benchBars: BenchBar[] = [
{ name: 'STRATA', tokensPerSecond: 41.6, isSubject: true },
{ name: 'Belady oracle prefetch', tokensPerSecond: 38.2 },
{ name: 'LRU tier cache', tokensPerSecond: 26.4 },
{ name: 'On-demand fetch', tokensPerSecond: 22.6 }];


export const oracleNote =
'An offline oracle maximises hit rate. It does not schedule transfers. STRATA trades away “perfect” hits to keep the CXL link saturated ahead of the router — so the hits it does take never land in the critical path.';

export const mechanisms: Mechanism[] = [
{
  n: '1',
  title: 'Router logit lookahead',
  body: 'Read the gate before it commits. Layer N activations predict layer N+1 top-k selection at 96.4% recall.'
},
{
  n: '2',
  title: 'Tier-aware admission',
  body: 'An expert enters HBM only when its predicted reuse outlives the transfer it would displace.'
},
{
  n: '3',
  title: 'Speculative promotion',
  body: 'The top two runners-up move alongside the winners. A wrong guess costs bandwidth, never correctness.'
},
{
  n: '4',
  title: 'Bandwidth-scheduled transfer',
  body: 'Promotions queue against measured CXL headroom, so prefetch never contends with activation traffic.'
},
{
  n: '5',
  title: 'Decay-based demotion',
  body: 'Residency is a half-life, not an LRU stack. A regime change drains the hot tier in two decode steps.'
},
{
  n: '6',
  title: 'Zero-copy CXL mapping',
  body: 'Warm experts are mapped, not copied. The GPU walks a descriptor table the host never rewrites.'
},
{
  n: '7',
  title: 'Fault-path fallback',
  body: 'A true miss executes in place from the warm tier. The cost is 2.1 µs, not a host page fault.'
}];


export const harness = {
  platform: 'Astera Labs Interop Lab reference platform',
  detail:
  '8× H100 SXM · Leo CXL 3.1 memory controller · 2 TB attached DRAM · Scorpio PCIe 6 fabric switch · 512 concurrent decode streams · ShareGPT + LMSYS-Chat routing traces, 40k prompts, three seeds.'
};