import { Kicker } from './Kicker';
import { CommandBlock } from './CommandBlock';
import { ArrowUp } from 'lucide-react';

export function Install() {
  return (
    <section id="install" className="scroll-mt-20 bg-cream py-12 md:py-16 border-t border-ink/10" aria-labelledby="install-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-ink/15 pb-4">
          <div>
            <div className="flex items-center gap-3">
              <Kicker as="p" className="text-ink/60 font-semibold">
                06 — Reproduction & Deployment
              </Kicker>
              <button
                type="button"
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                title="Back to Top — Overview (00)"
                aria-label="Back to Top"
                className="group flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-mono font-medium transition-all duration-200 border border-ink/15 bg-ink/5 text-ink/75 hover:border-amber-600/50 hover:bg-amber-50 hover:text-amber-800"
              >
                <span>Top: 00</span>
                <ArrowUp className="h-3.5 w-3.5 transition-transform duration-200 group-hover:-translate-y-0.5 text-amber-700" />
              </button>
            </div>
            <h2
              id="install-title"
              className="mt-2 font-display text-[clamp(2.4rem,4.5vw,3.8rem)] leading-[0.95] text-ink"
            >
              Reproduce & Deploy the <span className="italic">Evaluation Suite.</span>
            </h2>
          </div>
          <p className="max-w-[54ch] font-mono text-sm leading-relaxed text-ink/75">
            Requirements: Python 3.10+, PyTorch 2.1+, CUDA 12.0+ on NVIDIA GPUs (evaluated under constrained memory scenarios like RTX 4050 6GB).
          </p>
        </div>

        <div className="mt-8 grid gap-5 md:grid-cols-3">
          <div className="rounded-xl border border-ink/12 bg-white/80 p-5 shadow-sm">
            <span className="font-mono text-10 uppercase tracking-label text-ink/50 block">Step 1</span>
            <h3 className="mt-1 font-display text-base text-ink font-medium">Install Dependencies</h3>
            <p className="mt-1 font-mono text-[0.72rem] text-ink/60 mb-3">Install PyTorch and core requirements.</p>
            <CommandBlock command="pip install -e . -r requirements.txt" size="sm" />
          </div>

          <div className="rounded-xl border border-ink/12 bg-white/80 p-5 shadow-sm">
            <span className="font-mono text-10 uppercase tracking-label text-ink/50 block">Step 2</span>
            <h3 className="mt-1 font-display text-base text-ink font-medium">Run Benchmark Suite</h3>
            <p className="mt-1 font-mono text-[0.72rem] text-ink/60 mb-3">Execute S1–S3 sweeps and export metrics.</p>
            <CommandBlock command="python scripts/run_scenarios.py --scenarios s1,s2,s3" size="sm" />
          </div>

          <div className="rounded-xl border border-ink/12 bg-white/80 p-5 shadow-sm">
            <span className="font-mono text-10 uppercase tracking-label text-ink/50 block">Step 3</span>
            <h3 className="mt-1 font-display text-base text-ink font-medium">Launch Presentation</h3>
            <p className="mt-1 font-mono text-[0.72rem] text-ink/60 mb-3">Start GPU backend & Vite frontend server.</p>
            <CommandBlock command="python scripts/serve.py --port 8000" size="sm" />
          </div>
        </div>

        <div className="mt-12 pt-6 border-t border-ink/10 flex justify-center">
          <button
            type="button"
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="group inline-flex items-center gap-2.5 rounded-full px-5 py-2.5 font-mono text-xs font-medium transition-all duration-200 border shadow-sm border-ink/15 bg-white text-ink/80 hover:border-amber-600/50 hover:bg-amber-50 hover:text-amber-800 shadow-ink/5"
          >
            <span>Back to Top — Overview (00)</span>
            <div className="flex h-5 w-5 items-center justify-center rounded-full bg-amber-100 text-amber-800 transition-transform duration-200 group-hover:-translate-y-0.5">
              <ArrowUp className="h-3.5 w-3.5" />
            </div>
          </button>
        </div>
      </div>
    </section>
  );
}