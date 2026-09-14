import { useEffect, useRef, useState } from 'react';
import { api, ApiError, type Baseline, type RunResult, type SystemInfo } from '../../lib/api';

const DEFAULT_PROMPTS = [
  'Explain the fundamental principles of hierarchical memory tiering in high-performance computing.',
  'How does dynamic expert routing in Mixture-of-Experts models reduce inference computational cost?',
  'Compare compute-bound vs memory-bound kernel execution on modern GPU accelerators.',
];

const MEMORY_PRESETS = [300, 600, 900, 1500, 2500];
const TOKEN_PRESETS = [
  { label: '32', val: 32 },
  { label: '64', val: 64 },
  { label: '128', val: 128 },
  { label: '256', val: 256 },
  { label: 'Complete (EOS)', val: 0 },
];

interface HistoryEntry {
  baselineId: string;
  baselineName: string;
  tokS: number;
  hitRate: number;
  transferMb: number;
  evictions: number;
  wallTime: number;
  tokens: number;
  text: string;
}

export function PromptStudio({ systemInfo }: { systemInfo: SystemInfo }) {
  const [baselines, setBaselines] = useState<Baseline[] | null>(null);
  const [selectedBaseline, setSelectedBaseline] = useState<string>('hybrid_sota');
  const [prompt, setPrompt] = useState(DEFAULT_PROMPTS[0]);
  const [maxTokens, setMaxTokens] = useState(64);
  const [memoryMb, setMemoryMb] = useState(600);
  const [running, setRunning] = useState(false);
  const [activeResult, setActiveResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
  const [copied, setCopied] = useState(false);

  const chatBottomRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeResult, running]);

  async function executeRun(baselineId: string): Promise<RunResult | null> {
    const res = await api.run({
      prompt,
      baseline_id: baselineId,
      max_tokens: maxTokens,
      memory_constraint_mb: memoryMb,
    });
    setActiveResult(res);
    setHistory((prev) => [
      ...prev.filter((h) => h.baselineId !== baselineId),
      {
        baselineId,
        baselineName: res.baseline.name,
        tokS: res.tokens_per_second,
        hitRate: res.hit_rate,
        transferMb: res.transfer_mb,
        evictions: res.evictions,
        wallTime: res.wall_time_seconds,
        tokens: res.generated_tokens,
        text: res.generated_text,
      },
    ]);
    return res;
  }

  async function handleRunSingle() {
    if (running) return;
    setRunning(true);
    setError(null);
    try {
      await executeRun(selectedBaseline);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Inference request failed.');
    } finally {
      setRunning(false);
    }
  }

  async function handleRunAllBaselines() {
    if (running || !baselines || baselines.length === 0) return;
    setRunning(true);
    setError(null);
    setBatchProgress({ current: 0, total: baselines.length });

    try {
      for (let i = 0; i < baselines.length; i++) {
        setBatchProgress({ current: i + 1, total: baselines.length });
        setSelectedBaseline(baselines[i].id);
        await executeRun(baselines[i].id);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Batch evaluation encountered an error.');
    } finally {
      setRunning(false);
      setBatchProgress(null);
    }
  }

  const handleCopy = () => {
    if (!activeResult) return;
    navigator.clipboard.writeText(activeResult.generated_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const activeBaselineObj = baselines?.find((b) => b.id === selectedBaseline);

  return (
    <div className="rounded-xl border border-white/10 bg-[#0c0d12] p-4 md:p-6 shadow-2xl space-y-4">
      {/* Top Header: Model & GPU Status */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-3.5">
        <div className="flex items-center gap-2.5">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-amber text-ink font-bold font-mono text-xs">
            N
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-semibold text-cream">
                Nebula MoE Inference Assistant
              </span>
              <span className="rounded-full bg-emerald-500/20 text-emerald-400 px-2 py-0.2 font-mono text-[9px] font-medium border border-emerald-500/30">
                GPU Connected
              </span>
            </div>
            <p className="font-mono text-[10px] text-cream/50">
              Evaluated under constrained memory scenarios (Qwen1.5-4x0.5B MoE)
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleRunAllBaselines}
            disabled={running || !baselines}
            className="rounded-lg border border-amber/60 bg-amber/10 px-3 py-1.5 font-mono text-[11px] font-bold uppercase tracking-wider text-amber hover:bg-amber/20 transition-all disabled:opacity-40"
          >
            {batchProgress
              ? `Evaluating ${batchProgress.current}/${batchProgress.total}…`
              : '⚡ Run All 5 Baselines'}
          </button>
          <button
            type="button"
            onClick={handleRunSingle}
            disabled={running}
            className="rounded-lg bg-amber px-3.5 py-1.5 font-mono text-[11px] font-bold uppercase tracking-wider text-ink hover:bg-amber-400 transition-all shadow-md active:scale-95 disabled:opacity-40 flex items-center gap-1.5"
          >
            {running && <span className="h-2 w-2 rounded-full bg-ink animate-ping" />}
            {running ? 'Generating…' : 'Dispatch Prompt'}
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 font-mono text-xs text-rose-300">
          {error}
        </div>
      )}

      {/* Main Split Interface (Claude / Codex Layout) */}
      <div className="grid gap-4 lg:grid-cols-12 items-start">
        {/* Left Column (7 cols): Claude/Codex Chatbot Interface */}
        <div className="lg:col-span-7 space-y-3">
          {/* Scrollable Conversation Console */}
          <div className="rounded-xl border border-white/10 bg-[#111319] p-3.5 space-y-3.5 max-h-[260px] md:max-h-[290px] overflow-y-auto scrollbar-thin scrollbar-thumb-white/20">
            {/* User Bubble */}
            <div className="flex gap-2.5 items-start">
              <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-white/10 text-cream font-mono text-[9px] font-semibold">
                U
              </div>
              <div className="rounded-lg bg-white/5 border border-white/10 p-2.5 text-xs font-mono text-cream/90 leading-relaxed max-w-[92%]">
                {prompt}
              </div>
            </div>

            {/* Assistant Bubble (Codex/Claude Style) */}
            <div className="flex gap-2.5 items-start">
              <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-amber text-ink font-mono text-[9px] font-bold">
                AI
              </div>
              <div className="w-full rounded-lg bg-white/[0.02] border border-white/10 p-3 text-xs font-mono leading-relaxed space-y-2">
                {/* Assistant Bubble Header Bar */}
                <div className="flex items-center justify-between border-b border-white/5 pb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-amber text-[11px]">
                      {activeResult ? activeResult.baseline.name : activeBaselineObj?.name || 'Hybrid SOTA'}
                    </span>
                    {activeResult && (
                      <span className="text-emerald-400 text-[10px] font-medium">
                        {activeResult.tokens_per_second.toFixed(1)} tok/s · {activeResult.wall_time_seconds.toFixed(2)}s
                      </span>
                    )}
                  </div>
                  {activeResult && (
                    <button
                      type="button"
                      onClick={handleCopy}
                      className="text-[10px] font-mono text-cream/60 hover:text-amber transition-colors flex items-center gap-1"
                    >
                      {copied ? <span className="text-emerald-400">Copied ✓</span> : 'Copy'}
                    </button>
                  )}
                </div>

                {/* Assistant Output Content */}
                <div className="text-cream/90 whitespace-pre-wrap selection:bg-amber selection:text-ink max-h-[140px] overflow-y-auto pr-1">
                  {activeResult ? (
                    activeResult.generated_text
                  ) : running ? (
                    <span className="text-amber animate-pulse">
                      Generating tokens autoregressively on GPU accelerator…
                    </span>
                  ) : (
                    <span className="text-cream/40 italic">
                      Ready. Click &ldquo;Dispatch Prompt&rdquo; or &ldquo;Run All 5 Baselines&rdquo; to execute.
                    </span>
                  )}
                </div>

                {/* Assistant Telemetry Mini-Bar */}
                {activeResult && (
                  <div className="pt-2 border-t border-white/5 flex flex-wrap gap-3 text-[10px] text-cream/60">
                    <span>Generated: <strong className="text-cream">{activeResult.generated_tokens} tok</strong></span>
                    <span>Hit Rate: <strong className="text-cream">{(activeResult.hit_rate * 100).toFixed(1)}%</strong></span>
                    <span>Evictions: <strong className={activeResult.evictions === 0 ? 'text-emerald-400' : 'text-rose-400'}>{activeResult.evictions === 0 ? '0 (Zero Thrash)' : activeResult.evictions}</strong></span>
                    <span>PCIe Data: <strong className="text-cream">{activeResult.transfer_mb.toFixed(1)} MB</strong></span>
                  </div>
                )}
              </div>
            </div>
            <div ref={chatBottomRef} />
          </div>

          {/* Prompt Input & Presets */}
          <div className="space-y-2">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={2}
              disabled={running}
              className="w-full rounded-lg border border-white/15 bg-white/5 p-2.5 font-mono text-xs text-cream outline-none focus:border-amber transition-colors disabled:opacity-50 resize-none"
              placeholder="Type a custom prompt to test expert routing..."
            />

            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              {/* Prompt Presets */}
              <div className="flex items-center gap-1">
                <span className="font-mono text-[9px] uppercase text-cream/40 mr-1">Presets:</span>
                {DEFAULT_PROMPTS.map((p, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setPrompt(p)}
                    disabled={running}
                    className="rounded bg-white/5 border border-white/10 px-2 py-0.5 font-mono text-[10px] text-cream/70 hover:text-amber hover:border-amber/40 transition-colors"
                  >
                    P{idx + 1}
                  </button>
                ))}
              </div>

              {/* VRAM Envelope & Token Limits */}
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-1">
                  <span className="font-mono text-[9px] uppercase text-cream/50">VRAM:</span>
                  <div className="flex gap-0.5">
                    {MEMORY_PRESETS.map((mb) => (
                      <button
                        key={mb}
                        type="button"
                        onClick={() => setMemoryMb(mb)}
                        disabled={running}
                        className={`rounded px-1.5 py-0.5 font-mono text-[9px] border transition-colors ${
                          memoryMb === mb
                            ? 'bg-amber text-ink font-bold border-amber'
                            : 'border-white/10 text-cream/60 hover:text-cream'
                        }`}
                      >
                        {mb === 2500 ? 'Full' : `${mb}M`}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex items-center gap-1">
                  <span className="font-mono text-[9px] uppercase text-cream/50">Tokens:</span>
                  <div className="flex gap-0.5">
                    {TOKEN_PRESETS.map((t) => (
                      <button
                        key={t.val}
                        type="button"
                        onClick={() => setMaxTokens(t.val)}
                        disabled={running}
                        className={`rounded px-1.5 py-0.5 font-mono text-[9px] border transition-colors ${
                          maxTokens === t.val
                            ? 'bg-amber text-ink font-bold border-amber'
                            : 'border-white/10 text-cream/60 hover:text-cream'
                        }`}
                      >
                        {t.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Baseline Strategy Buttons */}
            {baselines && (
              <div className="flex flex-wrap gap-1.5 pt-0.5">
                {baselines.map((b) => {
                  const isSel = selectedBaseline === b.id;
                  return (
                    <button
                      key={b.id}
                      type="button"
                      onClick={() => setSelectedBaseline(b.id)}
                      disabled={running}
                      className={`rounded-md border px-2 py-1 font-mono text-[10px] font-semibold transition-all ${
                        isSel
                          ? 'border-amber bg-amber/15 text-amber shadow-sm'
                          : 'border-white/10 bg-white/5 text-cream/60 hover:text-cream hover:border-white/20'
                      }`}
                    >
                      {b.name}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Right Column (5 cols): Codex-Style Metrics Table & Comparative Leaderboard */}
        <div className="lg:col-span-5 space-y-3">
          {/* Telemetry Overview Cards */}
          <div className="rounded-xl border border-white/10 bg-[#111319] p-3 space-y-2.5">
            <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-amber block border-b border-white/10 pb-1.5">
              Live Hardware Telemetry
            </span>

            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg bg-white/[0.02] p-2 border border-white/5">
                <span className="font-mono text-[9px] uppercase text-cream/50 block">Throughput</span>
                <span className="font-display text-lg text-amber font-semibold">
                  {activeResult ? activeResult.tokens_per_second.toFixed(2) : '—'}
                </span>
                <span className="font-mono text-[8px] text-cream/50 block">tok / sec</span>
              </div>

              <div className="rounded-lg bg-white/[0.02] p-2 border border-white/5">
                <span className="font-mono text-[9px] uppercase text-cream/50 block">Cache Hit Rate</span>
                <span className="font-display text-lg text-cream font-semibold">
                  {activeResult ? `${(activeResult.hit_rate * 100).toFixed(1)}%` : '—'}
                </span>
                <span className="font-mono text-[8px] text-cream/50 block">
                  {activeResult ? `${activeResult.cache_hits} hit / ${activeResult.cache_misses} miss` : 'resident cache'}
                </span>
              </div>

              <div className="rounded-lg bg-white/[0.02] p-2 border border-white/5">
                <span className="font-mono text-[9px] uppercase text-cream/50 block">PCIe Bus Traffic</span>
                <span className="font-display text-sm text-cream font-medium">
                  {activeResult ? `${activeResult.transfer_mb.toFixed(1)} MB` : '—'}
                </span>
                <span className="font-mono text-[8px] text-emerald-400 block">
                  {activeResult?.evictions === 0 ? '0 Evictions (Zero Thrash)' : `${activeResult?.evictions ?? 0} evictions`}
                </span>
              </div>

              <div className="rounded-lg bg-white/[0.02] p-2 border border-white/5">
                <span className="font-mono text-[9px] uppercase text-cream/50 block">Peak VRAM</span>
                <span className="font-display text-sm text-cream font-medium">
                  {activeResult ? `${activeResult.peak_vram_mb.toFixed(0)} MB` : '—'}
                </span>
                <span className="font-mono text-[8px] text-cream/50 block">stable memory</span>
              </div>
            </div>
          </div>

          {/* Comparative Baseline Matrix */}
          <div className="rounded-xl border border-white/10 bg-[#111319] p-3 space-y-2">
            <div className="flex items-center justify-between border-b border-white/10 pb-1.5">
              <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-cream/90">
                Strategy Comparison Matrix
              </span>
              <span className="font-mono text-[9px] text-amber font-semibold">
                {history.length} evaluated
              </span>
            </div>

            {history.length > 0 ? (
              <div className="space-y-2">
                {history.map((h) => {
                  const maxTokS = Math.max(...history.map((item) => item.tokS), 12);
                  const barWidth = Math.min(100, Math.max(12, (h.tokS / maxTokS) * 100));
                  const isHybrid = h.baselineId.includes('hybrid');

                  return (
                    <div key={h.baselineId} className="rounded bg-white/[0.02] p-1.5 border border-white/5 space-y-1">
                      <div className="flex items-center justify-between font-mono text-[11px]">
                        <span className={isHybrid ? 'text-amber font-semibold' : 'text-cream/80'}>
                          {h.baselineName}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="text-cream/50 text-[9px]">{(h.hitRate * 100).toFixed(0)}% hit</span>
                          <span className="text-cream/50 text-[9px]">{h.transferMb.toFixed(0)} MB</span>
                          <span className={isHybrid ? 'text-amber font-bold' : 'text-cream/90 font-semibold'}>
                            {h.tokS.toFixed(1)} tok/s
                          </span>
                        </div>
                      </div>
                      <div className="h-1.5 w-full rounded bg-white/5 overflow-hidden">
                        <div
                          className={`h-full rounded transition-all duration-300 ${
                            isHybrid ? 'bg-amber' : 'bg-slate-400'
                          }`}
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-white/10 p-3.5 text-center font-mono text-[11px] text-cream/40">
                Execute runs or click &ldquo;Run All 5 Baselines&rdquo; to populate comparative matrix.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
