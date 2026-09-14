import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { WobbleRule } from './WobbleRule';
import { ProvenanceBadge } from './Provenance';
import { getS10, getS1, qwenLocalityRatio, s1ResidentFractionAt } from '../lib/results';
import { fadeUp, inView, stagger } from '../utils/motion';

const STATUS_TONE: Record<string, string> = {
  PASS: 'text-olive',
  EXPECTED_FAIL: 'text-cold',
  FAIL: 'text-hot',
  UNEXPECTED_PASS: 'text-hot'
};

export function Limits() {
  const s10 = getS10();
  const s1 = getS1();
  const run600 = s1.find((r) => r.hbm_budget_mb === 600);
  const residentFraction600 = s1ResidentFractionAt(600);
  const locality = qwenLocalityRatio();

  return (
    <section id="limits" className="bg-cream py-20 md:py-28" aria-labelledby="limits-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <Kicker as="p" className="text-ink/45">
          05 — Limits
        </Kicker>
        <h2
          id="limits-title"
          className="mt-5 max-w-[22ch] font-display text-[clamp(2.5rem,5.4vw,4.5rem)] leading-[0.92] text-ink">

          What This Page <span className="italic">Won&rsquo;t Claim.</span>
        </h2>
        <p className="mt-6 max-w-[62ch] font-mono text-[0.78rem] leading-relaxed text-ink/60">
          A results page that only shows what worked is not a research artifact, it&rsquo;s marketing.
          These are the caveats a careful reader would ask about — stated up front instead of found
          later.
        </p>

        <motion.ul
          variants={stagger()}
          initial="hidden"
          whileInView="visible"
          viewport={inView}
          className="mt-14 space-y-10">

          <motion.li variants={fadeUp}>
            <WobbleRule seed={301} className="opacity-70" />
            <div className="py-6">
              <div className="flex flex-wrap items-center gap-3">
                <h3 className="font-display text-2xl leading-snug text-ink md:text-[1.75rem]">
                  The small model has no real expert locality.
                </h3>
                <ProvenanceBadge value="measured" />
              </div>
              <p className="mt-3 max-w-[70ch] font-mono text-[0.78rem] leading-relaxed text-ink/65">
                The 4×0.5B model used for the capacity and mode-crossover charts was built by
                copying one dense feed-forward network into all four expert slots, with a randomly
                initialized, never-trained router. That keeps its output coherent — every expert
                computes the same function, so routing can’t corrupt anything — but it means
                expert selection carries no semantic meaning.
                {run600 && residentFraction600 != null &&
              <> The evidence is in its own numbers: at a 600 MB budget
                  ({(residentFraction600 * 100).toFixed(0)}% of the working set), its hit rate is{' '}
                  {(run600.hit_rate * 100).toFixed(1)}% — almost exactly the resident fraction. A
                  cache tracking capacity that closely is capturing no locality at all.</>
              }
              </p>
              <p className="mt-3 max-w-[70ch] font-mono text-[0.78rem] leading-relaxed text-ink/65">
                That model can support claims about the <em>hybrid activation-offload mechanism</em>{' '}
                (which doesn&rsquo;t depend on locality) — not about profile-guided placement or
                prefetching, which need a model with real specialization to mean anything.
              </p>
            </div>
          </motion.li>

          <motion.li variants={fadeUp}>
            <WobbleRule seed={302} className="opacity-70" />
            <div className="py-6">
              <div className="flex flex-wrap items-center gap-3">
                <h3 className="font-display text-2xl leading-snug text-ink md:text-[1.75rem]">
                  The real MoE is slow, and that&rsquo;s the point.
                </h3>
                <ProvenanceBadge value="measured" />
              </div>
              <p className="mt-3 max-w-[70ch] font-mono text-[0.78rem] leading-relaxed text-ink/65">
                Qwen1.5-MoE-A2.7B (14.3B total / 2.7B active parameters, 26.68 GB in fp16) runs on the
                same 6 GB card at 0.56 tok/s end-to-end, spilling most of its 1,440 experts to disk.
                {locality != null &&
              <> That slowness is exactly what makes its {locality.toFixed(2)}× locality ratio
                  meaningful — it’s a real model under real memory pressure, not a toy.</>
              }{' '}
                It is not the model to look at for throughput.
              </p>
              <p className="mt-3 max-w-[70ch] font-mono text-[0.72rem] leading-relaxed text-ink/45">
                One gap worth naming: the script that produced this specific run isn&rsquo;t in the
                current checkout, so this number can&rsquo;t be re-derived from source today. The
                figures are internally consistent (every token&rsquo;s HBM+DRAM+disk hits sum to
                exactly 96 = 24 layers × top-4, with no exceptions) but treat it as a recorded
                measurement, not a reproducible one, until that script is restored.
              </p>
            </div>
          </motion.li>

          <motion.li variants={fadeUp}>
            <WobbleRule seed={303} className="opacity-70" />
            <div className="py-6">
              <div className="flex flex-wrap items-center gap-3">
                <h3 className="font-display text-2xl leading-snug text-ink md:text-[1.75rem]">
                  A modeled number is not a measured one.
                </h3>
                <ProvenanceBadge value="modeled" />
              </div>
              <p className="mt-3 max-w-[70ch] font-mono text-[0.78rem] leading-relaxed text-ink/65">
                One benchmark file in this project computes throughput from a closed-form model with
                a hardcoded 85 ms/token constant. It is never rendered next to a measured wall-clock
                number on this page — they answer different questions, and putting them on the same
                axis would silently overstate whichever one looks better.
              </p>
            </div>
          </motion.li>

          {s10 &&
          <motion.li variants={fadeUp}>
              <WobbleRule seed={304} className="opacity-70" />
              <div className="py-6">
                <div className="flex flex-wrap items-center gap-3">
                  <h3 className="font-display text-2xl leading-snug text-ink md:text-[1.75rem]">
                    An edge-case suite found real bugs. Here they are.
                  </h3>
                  <ProvenanceBadge value="measured" />
                </div>
                <p className="mt-3 max-w-[70ch] font-mono text-[0.78rem] leading-relaxed text-ink/65">
                  Ten deliberate stress cases — degenerate budgets, VRAM over-commit, prefetch
                  abuse — ran against the live system. {s10.summary.failed > 0 ?
                <>It surfaced {s10.summary.failed} real failure(s), which led to two fixes: a crash in
                    the CXL→DRAM→HBM staging hop, and a silent-weight-loss path when a budget was
                    oversubscribed.</> :

                <>All cases passed or failed for their expected reason after the fixes below were
                    applied.</>
                }
                </p>
                <div className="mt-5 overflow-x-auto">
                  <table className="w-full min-w-[36rem] border-collapse font-mono text-[0.72rem]">
                    <thead>
                      <tr className="border-b border-ink/15 text-left text-ink/40">
                        <th className="py-2 pr-4 font-medium uppercase tracking-label">Case</th>
                        <th className="py-2 pr-4 font-medium uppercase tracking-label">Status</th>
                        <th className="py-2 font-medium uppercase tracking-label">Note</th>
                      </tr>
                    </thead>
                    <tbody>
                      {s10.cases.map((c) =>
                    <tr key={c.case} className="border-b border-ink/8">
                          <td className="py-2 pr-4 text-ink/70">{c.case}</td>
                          <td className={`py-2 pr-4 font-medium ${STATUS_TONE[c.status] ?? 'text-ink/60'}`}>
                            {c.status}
                          </td>
                          <td className="py-2 text-ink/50">{c.error ? c.error.slice(0, 72) : '—'}</td>
                        </tr>
                    )}
                    </tbody>
                  </table>
                </div>
                {s10.model_contaminated &&
              <p className="mt-3 font-mono text-[0.72rem] text-hot">
                    This run flagged model_contaminated — a fix (unpatch() on every exit path) has
                    since landed; see the repository for the current status.
                  </p>
              }
              </div>
            </motion.li>
          }
        </motion.ul>
      </div>
    </section>);

}
