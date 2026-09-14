import s1Data from '../data/generated/s1.json';
import s2Data from '../data/generated/s2.json';
import s3Data from '../data/generated/s3.json';
import s7Data from '../data/generated/s7.json';
import s10Data from '../data/generated/s10.json';
import qwenLiveData from '../data/generated/qwen14b_live_metrics.json';

export interface ScenarioRun {
  hbm_budget_mb: number;
  dram_budget_mb: number;
  cxl_budget_mb: number;
  execution_mode: string;
  enable_prefetch: boolean;
  enable_lookahead: boolean;
  cxl_bandwidth_gbps?: number | null;
  cxl_emulation_mode?: string | null;
  tokens_per_second: number;
  tokens_per_second_std?: number;
  wall_time_seconds: number;
  wall_time_seconds_std?: number;
  hit_rate: number;
  hbm_hit_rate: number;
  dram_hit_rate: number;
  cxl_hit_rate: number;
  cache_hits: number;
  cache_misses: number;
  evictions: number;
  transfer_mb: number;
  prefetch_precision?: number;
  peak_vram_mb?: number;
  mode_label?: string;
  repeats?: ScenarioRun[];
}

export interface S3Scenario {
  scenario: string;
  description: string;
  placement_stable_across_bandwidth_sweep: boolean;
  runs: ScenarioRun[];
}

export interface S7Run {
  batch_size: number;
  tokens_per_second: number;
  hit_rate: number;
  hbm_hit_rate: number;
  dram_hit_rate: number;
  cxl_hit_rate: number;
  execution_mode: 'hybrid' | 'weight_transfer';
  evictions: number;
  transfer_mb: number;
  wall_time_seconds: number;
}

export interface S7Scenario {
  scenario: string;
  description: string;
  runs: S7Run[];
}

export interface S10Case {
  case: string;
  description: string;
  expected_to_fail: boolean;
  status: string;
  error?: string;
  result?: Record<string, any>;
}

export interface S10Scenario {
  scenario: string;
  description: string;
  summary: {
    total: number;
    passed: number;
    expected_fail: number;
    failed: number;
    unexpected_pass: number;
  };
  model_contaminated: boolean;
  contamination_note: string | null;
  cases: S10Case[];
}

export interface QwenLiveMetrics {
  model_id: string;
  total_parameters: string;
  active_parameters: string;
  weight_footprint_fp16_gb: number;
  hardware: {
    gpu: string;
    vram_gb: number;
    host_ram_gb: number;
    os: string;
  };
  tier_configuration: {
    hbm_cache_mb: number;
    hbm_capacity_experts: number;
    dram_cache_mb: number;
    dram_capacity_experts: number;
    storage_experts: number;
    gpu_vram_allocated_gb: number;
  };
  live_generation_run: {
    prompt: string;
    prompt_tokens: number;
    tokens_generated: number;
    total_wall_time_s: number;
    end_to_end_throughput_tok_s: number;
    warmed_steady_state_tok_s_range: [number, number];
    warmed_steady_state_latency_ms_range: [number, number];
    hbm_cache_hits: number;
    dram_cache_hits: number;
    disk_fetches: number;
    combined_memory_hit_rate: number;
    total_pcie_transfer_mb: number;
    token_by_token_latency_ms: Array<{
      token: number;
      text: string;
      latency_ms: number;
      hbm_hits: number;
      dram_hits: number;
      disk_fetches: number;
    }>;
  };
}

export function getS1(): ScenarioRun[] {
  return ((s1Data as any)?.runs as ScenarioRun[]) ?? [];
}

export function getS2(): ScenarioRun[] {
  return ((s2Data as any)?.runs as ScenarioRun[]) ?? [];
}

export function getS3(): S3Scenario | null {
  return (s3Data as unknown as S3Scenario) ?? null;
}

export function getS7(): S7Scenario | null {
  return (s7Data as unknown as S7Scenario) ?? null;
}

export function getS10(): S10Scenario | null {
  return (s10Data as unknown as S10Scenario) ?? null;
}

export function getQwenLive(): QwenLiveMetrics | null {
  return (qwenLiveData as unknown as QwenLiveMetrics) ?? null;
}

export function crossoverAt(budgetMb: number) {
  const runs = getS2();
  const baseline = runs.find(
    (r) =>
      r.hbm_budget_mb === budgetMb &&
      r.execution_mode === 'weight_transfer' &&
      !r.enable_prefetch &&
      !r.enable_lookahead
  );
  const hybrid = runs.find(
    (r) =>
      r.hbm_budget_mb === budgetMb &&
      r.execution_mode === 'hybrid' &&
      !r.enable_prefetch &&
      !r.enable_lookahead
  );
  if (!baseline || !hybrid) return null;
  const speedup = hybrid.tokens_per_second / Math.max(baseline.tokens_per_second, 0.001);
  const byteRatio = baseline.transfer_mb / Math.max(hybrid.transfer_mb, 0.001);
  return { hybrid, baseline, speedup, byteRatio };
}

export function s1ResidentFractionAt(budgetMb: number): number | null {
  // Total model expert size for 4x0.5B: 24 layers * 4 experts * 17.3 MB = 1660.9 MB
  const totalExpertMb = 1660.9;
  return budgetMb / totalExpertMb;
}

export function qwenLocalityRatio(): number | null {
  const q = getQwenLive();
  if (!q || !q.tier_configuration || !q.live_generation_run) return null;
  const hbmCap = q.tier_configuration.hbm_capacity_experts ?? 60;
  const dramCap = q.tier_configuration.dram_capacity_experts ?? 121;
  const storageCap = q.tier_configuration.storage_experts ?? 1259;
  const total = hbmCap + dramCap + storageCap;
  if (total === 0) return null;
  const residentFraction = (hbmCap + dramCap) / total;
  const hitRate = q.live_generation_run.combined_memory_hit_rate ?? 0.6688;
  return hitRate / Math.max(residentFraction, 0.001);
}

export function qwenPerTokenDeltas(): Array<{
  token: number;
  text: string;
  latencyMs: number;
  hbm: number;
  dram: number;
  disk: number;
}> {
  const q = getQwenLive();
  const tokens = q?.live_generation_run?.token_by_token_latency_ms;
  if (!Array.isArray(tokens)) return [];
  return tokens.map((step, i) => {
    const prev = i > 0 ? tokens[i - 1] : null;
    const hbm = prev ? Math.max(0, step.hbm_hits - prev.hbm_hits) : step.hbm_hits;
    const dram = prev ? Math.max(0, step.dram_hits - prev.dram_hits) : step.dram_hits;
    const disk = step.disk_fetches;
    return {
      token: step.token,
      text: step.text,
      latencyMs: step.latency_ms,
      hbm,
      dram,
      disk,
    };
  });
}
