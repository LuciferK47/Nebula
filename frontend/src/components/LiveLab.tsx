import { Kicker } from './Kicker';
import { CommandBlock } from './CommandBlock';
import { PromptStudio } from './live/PromptStudio';
import { useLiveMode } from '../lib/useLiveMode';
import { SectionHeaderArrow, SectionBottomJump } from './SectionNav';

/**
 * The one thing this site does that isn't reading cached JSON: submit a
 * real prompt to a real GPU. Gated on useLiveMode() actually reaching
 * scripts/serve.py — the studio component itself is only ever mounted
 * once that's confirmed, so it never has to handle the "backend vanished"
 * case mid-render.
 */
export function LiveLab() {
  const state = useLiveMode();

  return (
    <section id="live-lab" className="on-ink bg-ink py-14 md:py-20 border-t border-ink-line/60 scroll-mt-20" aria-labelledby="live-lab-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex items-center gap-3">
          <Kicker as="p" className="text-amber">
            05 — Live Hardware Inference Studio
          </Kicker>
          <SectionHeaderArrow nextId="install" nextNum="06" nextLabel="Reproduction & Deployment" isDark={true} />
        </div>
        <h2
          id="live-lab-title"
          className="mt-3 max-w-[24ch] font-display text-[clamp(2.4rem,4.5vw,3.8rem)] leading-[0.95] text-cream"
        >
          Interactive Generation Terminal & <span className="italic">Telemetry.</span>
        </h2>

        {state.status === 'live' &&
        <div className="mt-12">
            <PromptStudio systemInfo={state.info} />
          </div>
        }

        {state.status === 'static' &&
        <div className="mt-12 max-w-[60ch]">
            <p className="font-mono text-[0.8rem] leading-relaxed text-khaki/80">
              This page can't reach a GPU right now, so everything above is drawn from the cached
              results in <code className="text-cream">results/*.json</code>. Start the server on a
              machine with the model built, then reload this page:
            </p>
            <div className="mt-4">
              <CommandBlock command="python scripts/serve.py" tone="cream" />
            </div>
            <p className="mt-3 font-mono text-10 text-cream/35">
              Point this page at it with VITE_API_BASE_URL if it isn&rsquo;t on localhost:8000.
            </p>
          </div>
        }

        {state.status === 'probing' &&
        <p className="mt-12 font-mono text-[0.8rem] text-cream/40">Checking for a live backend…</p>
        }

        <SectionBottomJump
          nextId="install"
          nextNum="06"
          nextLabel="Reproduction & Deployment"
          isDark={true}
        />
      </div>
    </section>);

}
