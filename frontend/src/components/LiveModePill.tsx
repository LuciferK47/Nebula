import { useLiveMode } from '../lib/useLiveMode';

/** Replaces the old "Reproduction kit" nav CTA, which pointed at a
 * #audience section that never delivered a kit. This tells the truth
 * about what state the page is actually in instead. */
export function LiveModePill() {
  const state = useLiveMode();

  if (state.status === 'probing') {
    return (
      <span className="flex items-center gap-2 rounded-full border border-ink/15 px-4 py-2 font-mono text-10 font-medium uppercase tracking-label text-ink/40">
        <span className="h-1.5 w-1.5 rounded-full bg-ink/30" aria-hidden="true" />
        Checking
      </span>);

  }

  if (state.status === 'live') {
    return (
      <a
        href="#live-lab"
        className="flex items-center gap-2 rounded-full border border-olive/40 bg-olive/10 px-4 py-2 font-mono text-10 font-medium uppercase tracking-label text-olive transition-colors duration-150 ease-out hover:bg-olive/20"
        title="Live Inference Active · Constrained Memory Evaluation">

        <span className="h-1.5 w-1.5 rounded-full bg-olive animate-pulse" aria-hidden="true" />
        Live · {state.info.cuda_available ? 'Inference Online' : 'CPU Mode'}
      </a>);

  }

  return (
    <span
      className="flex items-center gap-2 rounded-full border border-ink/25 px-4 py-2 font-mono text-10 font-medium uppercase tracking-label text-ink/60"
      title="python scripts/serve.py is not reachable — showing cached results instead.">

      <span className="h-1.5 w-1.5 rounded-full bg-ink/25" aria-hidden="true" />
      Static · cached results
    </span>);

}
