import { useEffect, useState } from 'react';
import { Kicker } from '../Kicker';
import { api, ApiError, type Baseline, type RunResult, type SystemInfo } from '../../lib/api';

const DEFAULT_PROMPTS = [
  'Explain the fundamental principles of hierarchical memory tiering in high-performance computing.',
  'How does dynamic expert routing in Mixture-of-Experts models reduce inference computational cost?',
  'Compare compute-bound vs memory-bound kernel execution on modern GPU accelerators.',
];

const MEMORY_PRESETS = [300, 600, 900, 1500, 2500];

interface HistoryEntry {
  baselineId: string;
  baselineName: string;
  tokS: number;
  hitRate: number;
  transferMb: number;
  evictions: number;
  wallTime: number;
}

export function PromptStudio({ systemInfo }: { systemInfo: SystemInfo }) {
  const [baselines, setBaselines] = useState<Baseline[] | null>(null);
  const [selectedBaseline, setSelectedBaseline] = useState<string>('hybrid_sota');
  const [prompt, setPrompt] = useState(DEFAULT_PROMPTS[0]);
  const [maxTokens, setMaxTokens] = useState(48);
  const [memoryMb, setMemoryMb] = useState(600);
  const [running, setRunning] = useState(false);
  const [activeResult, setActiveResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    let alive = true;
    api
      .baselines()
      .then((list) => {
        if (alive && list.length > 0) {
          setBaselines(list);
          setSelectedBaseline(list[0].id);
        }
      })
      .catch((err) => {
        if (alive) setError(err instanceof ApiError ? err.message : 'Could not load baselines.');
      });
    return () => {
      alive = false;
    };
  }, []);

  async function handleRun(baselineIdToUse?: string) {
    if (running) return;
    const targetId = baselineIdToUse || selectedBaseline;
    setRunning(true);
    setError(null);

    try {
      const res = await api.run({
        prompt,
        baseline_id: targetId,
        max_tokens: maxTokens,
        memory_constraint_mb: memoryMb,
      });
      setActiveResult(res);
      setHistory((prev) => [
        ...prev.filter((h) => h.baselineId !== targetId),
        {
          baselineId: targetId,
          baselineName: res.baseline.name,
          tokS: res.tokens_per_second,
          hitRate: res.hit_rate,
          transferMb: res.transfer_mb,
          evictions: res.evictions,
          wallTime: res.wall_time_seconds,
        },
      ]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Inference request failed.');
    } finally {
      setRunning(false);
    }
  }

  const activeBaselineObj = baselines?.find((b) => b.id === selectedBaseline);

  return (
    <div className="space-y-6">
      <div className="grid gap-8 lg:grid-cols-12 items-start">
        {/* Left Column (7 cols): Interactive Prompt & Generation Console */}
        <div className="lg:col-span-7 space-y-5">
          <div className="rounded-xl border border-ink-line/80 bg-ink-panel/80 p-6 shadow-xl backdrop-blur-sm space-y-4">
            <div className="flex items-center justify-between">
              <Kicker className="text-amber">Generation Prompt</Kicker>
              <div className="flex items-center gap-1.5 font-mono text-xs text-cream/70">
                <span>Model:</span>
                <span className="text-amber font-semibold">{systemInfo.default_model}</span>
              </div>
            </div>

            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              disabled={running}
              className="w-full rounded-lg border border-ink-line bg-ink-soft/90 p-3.5 font-mono text-xs md:text-sm leading-relaxed text-cream outline-none focus:border-amber transition-colors disabled:opacity-50 resize-none shadow-inner"
              placeholder="Enter a prompt to dispatch to GPU..."
            />

            {/* Quick Prompt Suggestions */}
            <div className="flex flex-wrap gap-2">
              {DEFAULT_PROMPTS.map((p, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setPrompt(p)}
                  disabled={running}
                  className="rounded-md border border-ink-line/70 bg-ink px-2.5 py-1 font-mono text-[11px] text-cream/70 hover:text-amber hover:border-amber/40 transition-colors disabled:opacity-40"
                >
                  Prompt {idx + 1}
                </button>
              ))}
            </div>

            {/* Baseline Configuration Mode */}
            {baselines && (
              <div>
                <Kicker className="text-cream/60 mb-2">Execution Strategy</Kicker>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {baselines.map((b) => {
                    const isSel = selectedBaseline === b.id;
                    return (
                      <button
                        key={b.id}
                        type="button"
                        onClick={() => setSelectedBaseline(b.id)}
                        disabled={running}
                        className={`rounded-lg border p-2.5 text-left transition-all ${
                          isSel
                            ? 'border-amber bg-amber/10 text-amber shadow-sm'
                            : 'border-ink-line/70 bg-ink hover:border-cream/30 text-cream/80'
                        }`}
                      >
                        <div className="flex items-center justify-between font-mono text-xs font-semibold">
                          <span>{b.name}</span>
                          <span className="text-[10px] text-cream/50 uppercase">{b.badge}</span>
                        </div>
                        <p className="mt-1 font-mono text-[10px] text-cream/60 line-clamp-1">
                          {b.description}
                        </p>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Sliders for Budget and Tokens */}
            <div className="grid gap-4 sm:grid-cols-2 pt-2 border-t border-ink-line/60">
              <div>
                <div className="flex items-baseline justify-between">
                  <span className="font-mono text-xs uppercase tracking-label text-cream/70">
                    VRAM Cache Budget
                  </span>
                  <span className="font-mono text-xs font-semibold text-amber">{memoryMb} MB</span>
                </div>
                <input
                  type="range"
                  min={300}
                  max={2500}
                  step={50}
                  value={memoryMb}
                  onChange={(e) => setMemoryMb(Number(e.target.value))}
                  disabled={running}
                  className="mt-2 w-full accent-amber"
                />
                <div className="mt-1.5 flex gap-1.5">
                  {MEMORY_PRESETS.map((mb) => (
                    <button
                      key={mb}
                      type="button"
                      onClick={() => setMemoryMb(mb)}
                      disabled={running}
                      className={`rounded px-2 py-0.5 font-mono text-[10px] border transition-colors ${
                        memoryMb === mb
                          ? 'bg-amber text-ink font-bold border-amber'
                          : 'border-ink-line text-cream/60 hover:text-cream'
                      }`}
                    >
                      {mb === 2500 ? 'Full' : `${mb}M`}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="flex items-baseline justify-between">
                  <span className="font-mono text-xs uppercase tracking-label text-cream/70">
                    Tokens to Generate
                  </span>
                  <span className="font-mono text-xs font-semibold text-amber">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min={16}
                  max={128}
                  step={16}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  disabled={running}
                  className="mt-2 w-full accent-amber"
                />
                <span className="mt-1 font-mono text-[10px] text-cream/50 block">
                  Capped for interactive responsiveness
                </span>
              </div>
            </div>

            {/* Big Action Button */}
            <button
              type="button"
              onClick={() => handleRun()}
              disabled={running}
              className={`w-full rounded-lg py-3 font-mono text-xs uppercase tracking-wider font-semibold transition-all shadow-md flex items-center justify-center gap-2 ${
                running
                  ? 'bg-amber/50 text-ink cursor-wait ring-2 ring-amber'
                  : 'bg-amber text-ink hover:bg-amber-400 hover:shadow-lg active:scale-[0.99]'
              }`}
            >
              {running && <span className="h-2 w-2 rounded-full bg-ink animate-ping" />}
              {running
                ? `Executing on GPU (${activeBaselineObj?.name || 'Inference'})...`
                : `Dispatch Token Generation (${activeBaselineObj?.name || 'GPU'})`}
            </button>
          </div>

          {/* Terminal Output Console */}
          <div className="rounded-xl border border-ink-line/90 bg-[#0c0d10] p-5 shadow-2xl space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full bg-rose-500/80 inline-block" />
                <span className="h-3 w-3 rounded-full bg-amber-500/80 inline-block" />
                <span className="h-3 w-3 rounded-full bg-emerald-500/80 inline-block" />
                <span className="ml-2 font-mono text-xs font-medium text-cream/70">
                  GPU Execution Stream
                </span>
              </div>
              {activeResult && (
                <span className="font-mono text-xs text-emerald-400 font-semibold">
                  {activeResult.tokens_per_second.toFixed(2)} tok/s · {activeResult.wall_time_seconds.toFixed(2)}s
                </span>
              )}
            </div>

            {error && (
              <div className="rounded bg-rose-500/10 border border-rose-500/30 p-3 font-mono text-xs text-rose-400">
                {error}
              </div>
            )}

            <div className="min-h-[140px] font-mono text-xs md:text-sm text-cream/90 leading-relaxed overflow-x-auto whitespace-pre-wrap selection:bg-amber selection:text-ink">
              {activeResult ? (
                <span>
                  <span className="text-amber font-semibold">&gt; {activeResult.prompt}</span>
                  {'\n\n'}
                  {activeResult.generated_text}
                </span>
              ) : running ? (
                <span className="text-amber animate-pulse">
                  &gt; Dispatching CUDA kernels and evaluating routing gates on physical GPU...
                </span>
              ) : (
                <span className="text-cream/40 italic">
                  Press &ldquo;Dispatch Token Generation&rdquo; above to run the live model and observe token decode...
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Right Column (5 cols): Compressed Telemetry & Dynamic Metrics Plot */}
        <div className="lg:col-span-5 space-y-5">
          {/* Compressed Hardware Metrics Panel */}
          <div className="rounded-xl border border-ink-line/80 bg-ink-panel/80 p-5 shadow-xl backdrop-blur-sm space-y-4">
            <div className="flex items-center justify-between border-b border-ink-line pb-3">
              <span className="font-mono text-xs font-bold uppercase tracking-wider text-amber">
                Hardware Telemetry
              </span>
              <span className="font-mono text-xs text-cream/60">{systemInfo.device_name}</span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg bg-ink-soft p-3 border border-ink-line/50">
                <span className="font-mono text-[10px] uppercase text-cream/50 block">Throughput</span>
                <span className="font-display text-2xl text-amber font-semibold">
                  {activeResult ? activeResult.tokens_per_second.toFixed(2) : '—'}
                </span>
                <span className="font-mono text-[10px] text-cream/60 block mt-0.5">tokens / second</span>
              </div>

              <div className="rounded-lg bg-ink-soft p-3 border border-ink-line/50">
                <span className="font-mono text-[10px] uppercase text-cream/50 block">Cache Hit Rate</span>
                <span className="font-display text-2xl text-cream font-semibold">
                  {activeResult ? `${(activeResult.hit_rate * 100).toFixed(1)}%` : '—'}
                </span>
                <span className="font-mono text-[10px] text-cream/60 block mt-0.5">
                  {activeResult ? `${activeResult.cache_hits} hits / ${activeResult.cache_misses} miss` : 'resident experts'}
                </span>
              </div>

              <div className="rounded-lg bg-ink-soft p-3 border border-ink-line/50">
                <span className="font-mono text-[10px] uppercase text-cream/50 block">PCIe Data Moved</span>
                <span className="font-display text-xl text-cream font-medium">
                  {activeResult ? `${activeResult.transfer_mb.toFixed(1)} MB` : '—'}
                </span>
                <span className="font-mono text-[10px] text-emerald-400 block mt-0.5">
                  {activeResult ? (activeResult.evictions === 0 ? '0 Evictions (Zero Thrash)' : `${activeResult.evictions} evictions`) : 'transfer volume'}
                </span>
              </div>

              <div className="rounded-lg bg-ink-soft p-3 border border-ink-line/50">
                <span className="font-mono text-[10px] uppercase text-cream/50 block">Accelerator VRAM</span>
                <span className="font-display text-xl text-cream font-medium">
                  {activeResult ? `${activeResult.peak_vram_mb.toFixed(0)} MB` : '—'}
                </span>
                <span className="font-mono text-[10px] text-cream/60 block mt-0.5">
                  allocated on {systemInfo.vram_total_gb.toFixed(0)} GB VRAM
                </span>
              </div>
            </div>
          </div>

          {/* Dynamic Metrics Comparison Plot */}
          <div className="rounded-xl border border-ink-line/80 bg-ink-panel/80 p-5 shadow-xl backdrop-blur-sm space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold uppercase tracking-wider text-cream/80">
                Dynamic Strategy Comparison
              </span>
              <span className="font-mono text-[10px] text-amber">
                {history.length > 0 ? `${history.length} runs executed` : 'Run multiple baselines'}
              </span>
            </div>

            {history.length > 0 ? (
              <div className="space-y-3 pt-2">
                {history.map((h) => {
                  const maxTokS = Math.max(...history.map((item) => item.tokS), 12);
                  const barWidth = Math.min(100, Math.max(10, (h.tokS / maxTokS) * 100));
                  const isHybrid = h.baselineId.includes('hybrid');

                  return (
                    <div key={h.baselineId} className="space-y-1">
                      <div className="flex items-center justify-between font-mono text-xs">
                        <span className={isHybrid ? 'text-amber font-semibold' : 'text-cream/70'}>
                          {h.baselineName}
                        </span>
                        <div className="flex items-center gap-3">
                          <span className="text-cream/50">{(h.hitRate * 100).toFixed(0)}% hit</span>
                          <span className={isHybrid ? 'text-amber font-display font-semibold' : 'text-cream/80 font-display'}>
                            {h.tokS.toFixed(2)} tok/s
                          </span>
                        </div>
                      </div>
                      <div className="h-2 w-full rounded bg-ink-soft overflow-hidden border border-ink-line/40">
                        <div
                          className={`h-full rounded transition-all duration-500 ${
                            isHybrid
                              ? 'bg-gradient-to-r from-amber-400 to-amber-600'
                              : 'bg-gradient-to-r from-slate-500 to-slate-400'
                          }`}
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-ink-line/60 p-6 text-center font-mono text-xs text-cream/50">
                Run different execution strategies to populate live comparative throughput and hit-rate curves.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
