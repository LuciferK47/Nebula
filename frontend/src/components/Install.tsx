import { Kicker } from './Kicker';
import { CommandBlock } from './CommandBlock';

export function Install() {
  return (
    <section id="install" className="bg-cream py-12 md:py-16 border-t border-ink/10" aria-labelledby="install-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-ink/15 pb-4">
          <div>
            <Kicker as="p" className="text-ink/60 font-semibold">
              06 — Reproduction & Deployment
            </Kicker>
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
      </div>
    </section>
  );
}