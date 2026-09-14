import { motion } from 'framer-motion';
import { Kicker } from './Kicker';
import { WobbleCircle, WobbleRule } from './WobbleRule';
import { ProvenanceBadge } from './Provenance';
import { CrossoverChart } from './charts/CrossoverChart';
import { CapacityCliffChart } from './charts/CapacityCliffChart';
import { CXLSensitivityChart } from './charts/CXLSensitivityChart';
import { QwenWarmupChart } from './charts/QwenWarmupChart';
import { mechanisms, oracleNote } from '../data/benchmarks';
import { crossoverAt, getS1, getS3, getQwenLive, qwenLocalityRatio, qwenPerTokenDeltas } from '../lib/results';
import { fadeUp, inView, stagger } from '../utils/motion';

export function Benchmark() {
  const crossover600 = crossoverAt(600);
  const crossover900 = crossoverAt(900);
  const locality = qwenLocalityRatio();
  const s1runs = getS1();
  const s3 = getS3();
  const qwenLive = getQwenLive();
  const qwenSteps = qwenPerTokenDeltas();

  const statCards: { value: string; label: string; baseline: string }[] = [];
  if (crossover600) {
    statCards.push({
      value: `${crossover600.speedup.toFixed(2)}×`,
      label: 'Hybrid vs. weight-transfer throughput',
      baseline: `${crossover600.hybrid.tokens_per_second.toFixed(2)} vs. ${crossover600.baseline.tokens_per_second.toFixed(2)} tok/s @ 600 MB`
    });
    statCards.push({
      value: `${Math.round(crossover600.byteRatio).toLocaleString()}×`,
      label: 'Less GPU↔host data moved',
      baseline: `${crossover600.hybrid.transfer_mb} MB vs. ${crossover600.baseline.transfer_mb.toLocaleString()} MB @ 600 MB`
    });
    statCards.push({
      value: `${crossover600.baseline.evictions} → 0`,
      label: 'Evictions eliminated',
      baseline: 'weight-transfer thrashes the cache; hybrid never evicts because the weight never moves'
    });
  }
  if (locality != null) {
    statCards.push({
      value: `${locality.toFixed(2)}×`,
      label: 'Real cache locality (Qwen1.5-MoE-A2.7B)',
      baseline: '66.9% hit rate at only 12.6% of experts resident — the cache is capturing real structure, not just capacity'
    });
  }

  return (
    <section id="benchmark" className="on-ink bg-ink-soft pb-20 pt-4 md:pb-28" aria-labelledby="bench-title">
      <div className="mx-auto max-w-site px-5 md:px-8">
        <div className="grid gap-10 border-t border-ink-line pt-16 lg:grid-cols-[minmax(0,1fr)_minmax(0,34rem)] lg:gap-20">
          <div>
            <Kicker as="p" className="text-amber">
              04 — Benchmark
            </Kicker>
            <h2
              id="bench-title"
              className="mt-5 max-w-[18ch] font-display text-[clamp(2.75rem,6.4vw,5.75rem)] leading-[0.9] text-cream">

              Moving Less <span className="italic">Beats Moving Faster.</span>
            </h2>
          </div>
          <p className="max-w-[68ch] font-mono text-[0.8rem] leading-relaxed text-khaki/85 lg:pt-8">
            {oracleNote}
          </p>
        </div>

        {statCards.length > 0 &&
        <motion.ul
          variants={stagger()}
          initial="hidden"
          whileInView="visible"
          viewport={inView}
          className="mt-14 grid gap-px border border-ink-line bg-ink-line sm:grid-cols-2 lg:grid-cols-4">

            {statCards.map((stat, i) =>
          <motion.li key={stat.label} variants={fadeUp} className="bg-ink p-6 md:p-7">
                <WobbleCircle tone="dark" size={34} seed={9 + i}>
                  <span className="font-mono text-10 font-medium tracking-wide text-amber">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                </WobbleCircle>
                <p className="mt-6 font-display text-[clamp(2.25rem,3.6vw,3.25rem)] leading-none text-cream">
                  {stat.value}
                </p>
                <h3 className="mt-4 font-mono text-11 font-medium uppercase tracking-label text-cream/85">
                  {stat.label}
                </h3>
                <p className="mt-2 font-mono text-10 leading-relaxed text-khaki/60">
                  {stat.baseline}
                </p>
              </motion.li>
          )}
          </motion.ul>
        }

        {/* Crossover: the headline comparison, at real scale */}
        {crossover600 &&
        <div className="mt-16 border-t border-ink-line pt-14">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <Kicker as="h3" className="text-cream/40">
                Hybrid vs. weight-transfer @ 600 MB budget
              </Kicker>
              <ProvenanceBadge value="measured" dark />
            </div>
            <div className="mt-8">
              <CrossoverChart hybrid={crossover600.hybrid} baseline={crossover600.baseline} />
            </div>
            {crossover900 &&
          <p className="mt-6 font-mono text-[0.74rem] leading-relaxed text-khaki/60">
                At a looser 900 MB budget the gap narrows but doesn&rsquo;t close:{' '}
                {crossover900.hybrid.tokens_per_second.toFixed(2)} vs.{' '}
                {crossover900.baseline.tokens_per_second.toFixed(2)} tok/s
                ({crossover900.speedup.toFixed(2)}×).
              </p>
          }
          </div>
        }

        {/* Capacity cliff */}
        {s1runs.length > 0 &&
        <div className="mt-16 border-t border-ink-line pt-14">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <Kicker as="h3" className="text-cream/40">
                Where tiering stops being free — hybrid mode, budget sweep
              </Kicker>
              <ProvenanceBadge value="measured" dark />
            </div>
            <div className="mt-8 rounded-lg bg-ink p-4 md:p-6">
              <CapacityCliffChart runs={s1runs} />
            </div>
          </div>
        }

        {/* CXL sensitivity — the paper-defense result */}
        {s3 && s3.runs.length > 0 &&
        <div className="mt-16 border-t border-ink-line pt-14">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <Kicker as="h3" className="text-cream/40">
                Is the CXL tier&rsquo;s calibration actually load-bearing?
              </Kicker>
              <ProvenanceBadge value="measured" dark />
            </div>
            <p className="mt-3 max-w-[70ch] font-mono text-[0.74rem] leading-relaxed text-khaki/60">
              Only the CXL link is emulated — no CXL hardware is attached to the test machine. Sweep
              its assumed bandwidth 4–64 GB/s, or disable emulation outright, and hit rate {' '}
              <strong className="text-cream/80">
                {s3.placement_stable_across_bandwidth_sweep ? 'never changes' : 'changed — see note below'}
              </strong>. Placement isn&rsquo;t being steered by a calibration guess.
            </p>
            <div className="mt-8 rounded-lg bg-ink p-4 md:p-6">
              <CXLSensitivityChart runs={s3.runs} />
            </div>
          </div>
        }

        {/* Qwen cache warm-up — the real-MoE locality story, in motion */}
        {qwenLive && qwenSteps.length > 0 &&
        <div className="mt-16 border-t border-ink-line pt-14">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <Kicker as="h3" className="text-cream/40">
                Watching the real cache warm up — Qwen1.5-MoE-A2.7B
              </Kicker>
              <ProvenanceBadge value="measured" dark />
            </div>
            <p className="mt-3 max-w-[70ch] font-mono text-[0.74rem] leading-relaxed text-khaki/60">
              {qwenLive.live_generation_run.token_by_token_latency_ms[0]?.text ? 'The first' : 'A'} token
              on a cold cache pays {qwenSteps[0].latencyMs.toFixed(0)} ms; ten tokens later, with HBM and
              DRAM warmed, the same decode step costs {qwenSteps[qwenSteps.length - 1].latencyMs.toFixed(0)} ms.
            </p>
            <div className="mt-8 rounded-lg bg-ink p-4 md:p-6">
              <QwenWarmupChart steps={qwenSteps} />
            </div>
          </div>
        }

        {/* The seven mechanisms */}
        <div className="mt-20 border-t border-ink-line pt-14">
          <Kicker as="h3" className="text-cream/40">
            How — seven mechanisms
          </Kicker>
          <motion.ol
            variants={stagger()}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.1 }}
            className="mt-8 grid gap-x-16 md:grid-cols-2">

            {mechanisms.map((m, i) =>
            <motion.li key={m.n} variants={fadeUp}>
                <WobbleRule tone="dark" seed={100 + i} />
                <div className="flex gap-5 py-6">
                  <WobbleCircle tone="dark" size={40} seed={30 + i}>
                    <span className="font-mono text-11 font-medium text-amber">{m.n}</span>
                  </WobbleCircle>
                  <div>
                    <h4 className="font-display text-[1.35rem] leading-snug text-cream">
                      {m.title}
                    </h4>
                    <p className="mt-2 font-mono text-[0.76rem] leading-relaxed text-khaki/75">
                      {m.body}
                    </p>
                  </div>
                </div>
              </motion.li>
            )}
          </motion.ol>
        </div>
      </div>
    </section>);

}
