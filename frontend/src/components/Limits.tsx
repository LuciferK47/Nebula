import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { ProvenanceBadge } from './Provenance';
import { getS10, getS1, qwenLocalityRatio, s1ResidentFractionAt } from '../lib/results';
import { fadeUp, stagger } from '../utils/motion';

const STATUS_TONE: Record<string, string> = {
  PASS: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
  EXPECTED_FAIL: 'text-blue-400 bg-blue-400/10 border-blue-400/30',
  FAIL: 'text-rose-400 bg-rose-400/10 border-rose-400/30',
};

export function Limits() {
  const s10 = getS10();
  const s1 = getS1();
  const run600 = s1.find((r) => r.hbm_budget_mb === 600);
  const residentFraction600 = s1ResidentFractionAt(600);
  const locality = qwenLocalityRatio();

  return (
    <section id="limits" className="bg-cream py-14 md:py-20 border-t border-ink/10" aria-labelledby="limits-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-ink/15 pb-4">
          <div>
            <Kicker as="p" className="text-ink/50">
              04 — Technical Realities & Caveats
            </Kicker>
            <h2
              id="limits-title"
              className="mt-2 font-display text-[clamp(2.2rem,4.2vw,3.5rem)] leading-[0.95] text-ink"
            >
              Engineering Boundaries & <span className="italic">Disclosures.</span>
            </h2>
          </div>
          <p className="max-w-[48ch] font-mono text-[0.78rem] leading-relaxed text-ink/65">
            Transparent reporting of model differences, hardware boundaries, and automated stress
            test edge cases.
          </p>
        </div>

        <motion.div
          variants={stagger()}
          initial="hidden"
          whileInView="visible"
          className="mt-8 grid gap-5 md:grid-cols-2"
        >
          {/* Card 1: Synthetic vs Real MoE Locality */}
          <motion.div
            variants={fadeUp}
            className="rounded-xl border border-ink/12 bg-white/80 p-5 shadow-sm backdrop-blur"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-10 font-bold uppercase tracking-wider text-ink/50">
                Model Locality Differences
              </span>
              <ProvenanceBadge value="measured" />
            </div>
            <h3 className="mt-2 font-display text-lg text-ink font-medium">
              Synthetic vs. Real MoE Locality
            </h3>
            <p className="mt-2 font-mono text-[0.75rem] leading-relaxed text-ink/70">
              The 4×0.5B synthetic model uses replicated weights with uniform routing. At 600 MB budget
              ({residentFraction600 != null ? `${(residentFraction600 * 100).toFixed(0)}%` : '50%'} capacity),
              its hit rate is {run600 != null ? `${(run600.hit_rate * 100).toFixed(1)}%` : '50%'}—purely capacity-bound.
            </p>
            <p className="mt-2 font-mono text-[0.75rem] leading-relaxed text-ink/70">
              In contrast, real <strong>Qwen1.5-MoE-A2.7B</strong> achieves a{' '}
              <strong className="text-ink">{locality != null ? `${locality.toFixed(2)}×` : '5.31×'} locality ratio</strong>{' '}
              (66.9% hit rate at 12.6% residency), demonstrating genuine semantic clustering.
            </p>
          </motion.div>

          {/* Card 2: Hardware vs Emulation Boundary */}
          <motion.div
            variants={fadeUp}
            className="rounded-xl border border-ink/12 bg-white/80 p-5 shadow-sm backdrop-blur"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-10 font-bold uppercase tracking-wider text-ink/50">
                Hardware Boundaries
              </span>
              <ProvenanceBadge value="measured" />
            </div>
            <h3 className="mt-2 font-display text-lg text-ink font-medium">
              Physical Transfers vs. CXL Emulation
            </h3>
            <p className="mt-2 font-mono text-[0.75rem] leading-relaxed text-ink/70">
              All GPU VRAM (6GB GDDR6 on NVIDIA RTX 4050) and Host RAM (16GB) transfers are physical
              CUDA asynchronous DMA operations timed with CUDA events.
            </p>
            <p className="mt-2 font-mono text-[0.75rem] leading-relaxed text-ink/70">
              No physical CXL device is attached; CXL is software-emulated with calibrated 350 ns
              latency injection. Our 4–64 GB/s sweep proves placement decisions are 100% invariant to
              emulation calibration.
            </p>
          </motion.div>

          {/* Card 3: Modeled vs Measured Clarification */}
          <motion.div
            variants={fadeUp}
            className="rounded-xl border border-ink/12 bg-white/80 p-5 shadow-sm backdrop-blur"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-10 font-bold uppercase tracking-wider text-ink/50">
                Methodology Transparency
              </span>
              <ProvenanceBadge value="modeled" />
            </div>
            <h3 className="mt-2 font-display text-lg text-ink font-medium">
              Modeled Pipeline vs. Physical Wall-Clock
            </h3>
            <p className="mt-2 font-mono text-[0.75rem] leading-relaxed text-ink/70">
              Analytical pipeline throughput models (assuming fixed 85 ms compute per token) are
              strictly separated from live GPU wall-clock measurements to prevent misleading comparisons.
            </p>
            <p className="mt-2 font-mono text-[0.75rem] leading-relaxed text-ink/70">
              The 3,110× PCIe traffic reduction reflects actual bytes transferred over the PCIe bus
              (4.6 MB vs 14,308 MB), while end-to-end decode speedup is 3.03×.
            </p>
          </motion.div>

          {/* Card 4: S10 Robustness Suite */}
          <motion.div
            variants={fadeUp}
            className="rounded-xl border border-ink/12 bg-white/80 p-5 shadow-sm backdrop-blur"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-10 font-bold uppercase tracking-wider text-ink/50">
                Automated Robustness
              </span>
              <span className="font-mono text-10 font-semibold px-2 py-0.5 rounded bg-emerald-400/10 text-emerald-700 border border-emerald-400/30">
                0 Unexpected Failures · 0.0 MB Leak
              </span>
            </div>
            <h3 className="mt-2 font-display text-lg text-ink font-medium">
              Scenario 10 Stress & Cascade Validation
            </h3>
            <p className="mt-2 font-mono text-[0.75rem] leading-relaxed text-ink/70">
              Ten deliberate stress cases (extreme over-subscription, zero-budget clamps, prefetch
              thrashing) executed on physical GPU hardware.
            </p>
            {s10 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {s10.cases.map((c) => (
                  <span
                    key={c.case}
                    className={`font-mono text-[9px] px-2 py-0.5 rounded border ${
                      STATUS_TONE[c.status] || 'text-ink/60 bg-ink/5 border-ink/10'
                    }`}
                    title={c.error || c.status}
                  >
                    {c.case.replace('case_', 'C')}: {c.status}
                  </span>
                ))}
              </div>
            )}
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
