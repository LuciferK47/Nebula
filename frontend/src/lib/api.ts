const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000';

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export interface Baseline {
  id: string;
  name: string;
  badge: string;
  color: string;
  hbm_budget_mb: number;
  dram_budget_mb: number;
  cxl_budget_mb: number;
  execution_mode: string;
  enable_prefetch: boolean;
  enable_lookahead: boolean;
  description: string;
}

export interface SystemInfo {
  device_name: string;
  vram_total_gb: number;
  cuda_available: boolean;
  default_model: string;
  torch_version: string;
}

export interface RunResult {
  baseline: Baseline;
  prompt: string;
  generated_text: string;
  full_text: string;
  token_ids: number[];
  generated_tokens: number;
  wall_time_seconds: number;
  tokens_per_second: number;
  hit_rate: number;
  cache_hits: number;
  cache_misses: number;
  evictions: number;
  transfer_bytes: number;
  transfer_mb: number;
  peak_vram_mb: number;
  prefetch_precision: number;
  memory_constraint_mb: number;
  timestamp: string;
}

export interface RunParams {
  prompt: string;
  baseline_id: string;
  max_tokens?: number;
  memory_constraint_mb?: number;
  temperature?: number;
  top_p?: number;
  do_sample?: boolean;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${path}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
    });
    if (!res.ok) {
      let errMsg = `Request failed (${res.status})`;
      try {
        const body = await res.json();
        if (body.error) errMsg = body.error;
      } catch {
        // ignore parse error
      }
      throw new ApiError(errMsg, res.status);
    }
    return (await res.json()) as T;
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(err.message || 'Network error');
  }
}

export const api = {
  async systemInfo(): Promise<SystemInfo> {
    return request<SystemInfo>('/api/system_info');
  },

  async baselines(): Promise<Baseline[]> {
    return request<Baseline[]>('/api/baselines');
  },

  async run(params: RunParams): Promise<RunResult> {
    return request<RunResult>('/api/run', {
      method: 'POST',
      body: JSON.stringify({
        prompt: params.prompt,
        baseline_id: params.baseline_id,
        max_tokens: params.max_tokens ?? 128,
        max_new_tokens: params.max_tokens ?? 128,
        memory_constraint_mb: params.memory_constraint_mb,
        do_sample: params.do_sample ?? false,
        temperature: params.temperature ?? 0.7,
        top_p: params.top_p ?? 0.9,
      }),
    });
  },

  async historicalResults(): Promise<any> {
    return request<any>('/api/historical_results');
  },
};
