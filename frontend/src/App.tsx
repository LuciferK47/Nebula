import { LiveModeProvider } from './lib/useLiveMode';
import { TopBar } from './components/TopBar';
import { Hero } from './components/Hero';
import { Architecture } from './components/Architecture';
import { ChipExplorer } from './components/explorer/ChipExplorer';
import { Benchmark } from './components/Benchmark';
import { Limits } from './components/Limits';
import { LiveLab } from './components/LiveLab';
import { Install } from './components/Install';
import { SiteFooter } from './components/SiteFooter';
import { FloatingSectionNav } from './components/SectionNav';

export function App() {
  return (
    <LiveModeProvider>
      <div className="w-full bg-cream">
        <TopBar />
        <main>
          <Hero />
          <Architecture />
          <ChipExplorer />
          <Benchmark />
          <Limits />
          <LiveLab />
          <Install />
        </main>
        <FloatingSectionNav />
        <SiteFooter />
      </div>
    </LiveModeProvider>
  );
}
