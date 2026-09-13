import React from 'react';
import { TopBar } from './components/TopBar';
import { Hero } from './components/Hero';
import { ValidatedOn } from './components/ValidatedOn';
import { Solved } from './components/Solved';
import { FabricBand } from './components/FabricBand';
import { Architecture } from './components/Architecture';
import { Benchmark } from './components/Benchmark';
import { Install } from './components/Install';
import { FieldNotes } from './components/FieldNotes';
import { Personas } from './components/Personas';
import { SiteFooter } from './components/SiteFooter';

interface AppProps {
  /** The submission ticker above the nav. Off for a neutral, non-competition read. */
  showTicker?: boolean;
  /** Pilot-deployment quote wall. Off when only the measured results should speak. */
  showFieldNotes?: boolean;
}

export function App({ showTicker = true, showFieldNotes = true }: AppProps) {
  return (
    <div className="w-full bg-cream">
      <TopBar showTicker={showTicker} />
      <main>
        <Hero />
        <ValidatedOn />
        <Solved />
        <FabricBand />
        <Architecture />
        <Benchmark />
        <Install />
        {showFieldNotes && <FieldNotes />}
        <Personas />
      </main>
      <SiteFooter />
    </div>);

}