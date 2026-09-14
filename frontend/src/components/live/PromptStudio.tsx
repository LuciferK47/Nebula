import { useEffect, useState } from 'react';
import { CommandBlock } from '../CommandBlock';
import { Kicker } from '../Kicker';
import { ProvenanceBadge } from '../Provenance';
import { WobbleCircle } from '../WobbleRule';
import { api, ApiError, type Baseline, type RunResult } from '../../lib/api';
import type { SystemInfo } from '../../lib/api';

const DEFAULT_PROMPT = 'Explain the fundamental principles of hierarchical memory tiering in high-performance computing.';
const MAX_TOKENS_CEILING = 256; // see api.ts / GPU_RUNBOOK.md: no server-side run timeout,
// no request queue — a long run just holds the global model lock. Capping
// here keeps every run comfortably under the client's own 300s timeout
// instead of exposing the old UI's "unlimited (0)" option, which could
// leave the lock held by an aborted fetch.
const MEMORY_PRESETS = [300, 600, 900, 1500, 2500];

type RunState = Record<string, { result?: RunResult; error?: string }>;

export function PromptStudio({ systemInfo }: { systemInfo: SystemInfo }) {
  const [baselines, setBaselines] = useState<Baseline[] | null>(null);
  const [baselinesError, setBaselinesError] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxTokens, setMaxTokens] = useState(64);
  const [memoryMb, setMemoryMb] = useState(600);
  const [runs, setRuns] = useState<RunState>({});
  const [runningId, setRunningId] = useState<string | null>(null);
  const [sequential, setSequential] = useState<{ current: number; total: number } | null>(null);

  useEffect(() => {
    let alive = true;
    api.baselines().
    then((list) => {
      if (alive) setBaselines(list);
    }).
    catch((err) => {
      if (alive) setBaselinesError(err instanceof ApiError ? err.message : 'Could not load baselines.');
    });
    return () => {
      alive = false;
    };
  }, []);

  const modelNotBuilt = systemInfo.default_model.startsWith('not built');
  const busy = runningId !== null;

  async function runOne(baselineId: string) {
    if (busy) return; // single-flight: the server holds one global model
    // lock and has no queue, so a second concurrent request just hangs.
    setRunningId(baselineId);
    setRuns((prev) => ({ ...prev, [baselineId]: {} }));
    try {
      const result = await api.run({
        prompt,
        baseline_id: baselineId,
        max_tokens: maxTokens,
        memory_constraint_mb: memoryMb,
      });
      setRuns((prev) => ({ ...prev, [baselineId]: { result } }));
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Request failed.';
      setRuns((prev) => ({ ...prev, [baselineId]: { error: message } }));
    } finally {
      setRunningId(null);
    }
  }

  async function runAllSequential() {
    if (busy || !baselines) return;
    setSequential({ current: 0, total: baselines.length });
    for (let i = 0; i < baselines.length; i++) {
      setSequential({ current: i + 1, total: baselines.length });
      // Deliberately sequential /api/run calls, not /api/compare: compare
      // is all-or-nothing server-side (one failing baseline discards every
      // result in the batch), so a partial matrix here still shows what
      // succeeded.
      await runOne(baselines[i].id);
    }
    setSequential(null);
  }

  return (
    <div>
      {modelNotBuilt &&
      <div className="mb-8 rounded-lg border border-hot/40 bg-hot/10 p-5">
          <p className="font-mono text-[0.78rem] leading-relaxed text-cream/85">
            The server is reachable, but no model is built yet. Every run below will fail until
            this is run on the machine hosting <code className="text-cream">scripts/serve.py</code>:
          </p>
          <div className="mt-3">
            <CommandBlock command="python scripts/build_chat_moe.py" tone="cream" size="sm" />
          </div>
        </div>
      }

      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div>
          <Kicker className="text-cream/40">Prompt</Kicker>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            disabled={busy}
            className="mt-3 w-full rounded-lg border border-ink-line bg-ink-soft p-4 font-mono text-[0.8rem] leading-relaxed text-cream outline-none focus-visible:border-amber disabled:opacity-60" />


          <div className="mt-6 grid gap-6 sm:grid-cols-2">
            <div>
              <div className="flex items-baseline justify-between">
                <Kicker className="text-cream/40">Expert cache budget</Kicker>
                <span className="font-mono text-11 text-cream/70">{memoryMb} MB</span>
              </div>
              <input
                type="range"
                min={300}
                max={2500}
                step={50}
                value={memoryMb}
                onChange={(e) => setMemoryMb(Number(e.target.value))}
                disabled={busy}
                className="mt-3 w-full accent-amber" />

              <div className="mt-2 flex flex-wrap gap-1.5">
                {MEMORY_PRESETS.map((mb) =>
                <button
                  key={mb}
                  type="button"
                  onClick={() => setMemoryMb(mb)}
                  disabled={busy}
                  className={`rounded-full border px-2.5 py-1 font-mono text-10 uppercase tracking-label transition-colors disabled:opacity-50 ${
                  memoryMb === mb ? 'border-amber text-amber' : 'border-ink-line text-cream/50 hover:text-cream'}`
                  }>

                    {mb === 2500 ? 'full' : mb}
                  </button>
                )}
              </div>
              <p className="mt-2 font-mono text-10 text-cream/35">
                Not VRAM — the base model, KV cache and allocator sit outside this budget.
              </p>
            </div>

            <div>
              <div className="flex items-baseline justify-between">
                <Kicker className="text-cream/40">Tokens to generate</Kicker>
                <span className="font-mono text-11 text-cream/70">{maxTokens}</span>
              </div>
              <input
                type="range"
                min={16}
                max={MAX_TOKENS_CEILING}
                step={16}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                disabled={busy}
                className="mt-3 w-full accent-amber" />

              <p className="mt-2 font-mono text-10 text-cream/35">
                Capped at {MAX_TOKENS_CEILING} — the server holds one global lock with no queue and
                no timeout, so a long unconstrained run would block every later request.
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-ink-line p-5">
          <Kicker className="text-cream/40">System</Kicker>
          <dl className="mt-3 space-y-2 font-mono text-11">
            <div className="flex justify-between text-cream/70">
              <dt>Device</dt>
              <dd className="text-right text-cream/90">{systemInfo.device_name}</dd>
            </div>
            <div className="flex justify-between text-cream/70">
              <dt>VRAM</dt>
              <dd>{systemInfo.vram_total_gb.toFixed(1)} GB</dd>
            </div>
            <div className="flex justify-between text-cream/70">
              <dt>Torch</dt>
              <dd>{systemInfo.torch_version}</dd>
            </div>
          </dl>
          <button
            type="button"
            onClick={runAllSequential}
            disabled={busy || !baselines || modelNotBuilt}
            className="mt-5 w-full rounded-full bg-cream px-4 py-2.5 font-mono text-10 font-medium uppercase tracking-label text-ink transition-colors hover:bg-amber disabled:cursor-not-allowed disabled:opacity-40">

            {sequential ? `Running ${sequential.current}/${sequential.total}…` : 'Run all 5 baselines'}
          </button>
        </div>
      </div>

      <div className="mt-10 border-t border-ink-line pt-8">
        {baselinesError &&
        <p className="font-mono text-[0.78rem] text-hot">{baselinesError}</p>
        }
        {!baselines && !baselinesError &&
        <p className="font-mono text-[0.78rem] text-cream/40">Loading baselines…</p>
        }
        {baselines &&
        <ul className="grid gap-4 md:grid-cols-2">
            {baselines.map((b, i) => {
            const run = runs[b.id];
            const isRunning = runningId === b.id;
            return (
              <li key={b.id} className="rounded-lg border border-ink-line p-5">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start gap-3">
                      <WobbleCircle tone="dark" size={30} seed={40 + i}>
                        <span className="font-mono text-10 font-medium text-amber">{i + 1}</span>
                      </WobbleCircle>
                      <div>
                        <h4 className="font-display text-lg leading-tight text-cream">{b.name}</h4>
                        <p className="mt-1 font-mono text-10 uppercase tracking-label text-cream/40">{b.badge}</p>
                      </div>
                    </div>
                    <button
                    type="button"
                    onClick={() => runOne(b.id)}
                    disabled={busy || modelNotBuilt}
                    className="shrink-0 rounded-full border border-ink-line px-3 py-1.5 font-mono text-10 uppercase tracking-label text-cream/70 transition-colors hover:border-amber hover:text-amber disabled:cursor-not-allowed disabled:opacity-40">

                      {isRunning ? 'Running…' : 'Run'}
                    </button>
                  </div>
                  <p className="mt-3 font-mono text-[0.72rem] leading-relaxed text-khaki/70">{b.description}</p>

                  {run?.error &&
                <p className="mt-4 whitespace-pre-line rounded-md bg-hot/10 p-3 font-mono text-[0.7rem] leading-relaxed text-hot">
                      {run.error}
                    </p>
                }

                  {run?.result &&
                <div className="mt-4 border-t border-ink-line pt-4">
                      <div className="flex flex-wrap items-center gap-2">
                        <ProvenanceBadge value="measured" dark />
                        <span className="font-mono text-10 text-cream/35">{run.result.timestamp}</span>
                      </div>
                      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 font-mono text-11 sm:grid-cols-3">
                        <div>
                          <dt className="text-cream/35">tok/s</dt>
                          <dd className="text-cream/90">{run.result.tokens_per_second.toFixed(2)}</dd>
                        </div>
                        <div>
                          <dt className="text-cream/35">hit rate</dt>
                          <dd className="text-cream/90">{(run.result.hit_rate * 100).toFixed(1)}%</dd>
                        </div>
                        <div>
                          <dt className="text-cream/35">evictions</dt>
                          <dd className={run.result.evictions === 0 ? 'text-olive' : 'text-cream/90'}>
                            {run.result.evictions}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-cream/35">transfer</dt>
                          <dd className="text-cream/90">{run.result.transfer_mb.toFixed(2)} MB</dd>
                        </div>
                        <div>
                          <dt className="text-cream/35">peak VRAM</dt>
                          <dd className="text-cream/90">{run.result.peak_vram_mb.toFixed(0)} MB</dd>
                        </div>
                        <div>
                          <dt className="text-cream/35">wall time</dt>
                          <dd className="text-cream/90">{run.result.wall_time_seconds.toFixed(2)}s</dd>
                        </div>
                      </dl>
                      <pre className="mt-4 overflow-x-auto whitespace-pre-wrap rounded-md bg-ink-soft p-3 font-mono text-[0.72rem] leading-relaxed text-cream/85">
                        {run.result.generated_text}
                      </pre>
                      {run.result.generated_tokens > run.result.token_ids.length &&
                    <p className="mt-2 font-mono text-10 text-cream/35">
                          showing text for all {run.result.generated_tokens} tokens; token-id list is
                          truncated to the first {run.result.token_ids.length}
                        </p>
                    }
                    </div>
                }
                </li>);

          })}
          </ul>
        }
      </div>
    </div>);

}
