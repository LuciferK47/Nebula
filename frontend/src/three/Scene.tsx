import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { AdaptiveDpr, OrbitControls } from '@react-three/drei';
import { ExpertField } from './ExpertField';
import { TierPlates } from './TierPlates';
import { SceneLights } from './SceneChrome';
import { ComputeDie } from './ComputeDie';
import { HBMStacks } from './HBMStacks';
import { SiliconInterposer } from './SiliconInterposer';
import { SubstratePackage } from './SubstratePackage';
import { BusLines } from './BusLines';
import { MigrationDemo } from './MigrationDemo';
import { TraceDemo3D } from './TraceDemo3D';
import type { TierId } from './layout';

export interface SceneProps {
  placement: TierId[];
  explode: number;
  mode: 'weight_transfer' | 'hybrid';
  playToken: number;
  focusTier: TierId | null;
  autoRotate: boolean;
  traceReplay?: boolean;
  activeTraceStep?: number;
  tracePlaying?: boolean;
}

export function Scene({
  placement,
  explode,
  mode,
  playToken,
  focusTier,
  autoRotate,
  traceReplay,
  activeTraceStep,
  tracePlaying,
}: SceneProps) {
  return (
    <Canvas
      dpr={[1, 1.8]}
      camera={{ position: [11, 9, 13], fov: 40 }}
      gl={{ antialias: true, powerPreference: 'high-performance' }}
    >
      <color attach="background" args={['#141414']} />
      <Suspense fallback={null}>
        <SceneLights />

        {/* 3D Silicon Packaging Hierarchy (Top Tier - GPU / HBM Accelerator) */}
        <ComputeDie explode={explode} />
        <HBMStacks explode={explode} />
        <SiliconInterposer explode={explode} />
        <SubstratePackage explode={explode} />

        {/* High-speed PCIe Gen5 / CXL 3.0 Interconnect Conduits */}
        <BusLines explode={explode} />

        {/* Physical Tier Chassis Plates & Specifications */}
        <TierPlates explode={explode} focusTier={focusTier} />

        {/* MoE Expert Residency Matrix (24 Layers x 4 Experts) */}
        <ExpertField placement={placement} explode={explode} focusTier={focusTier} />

        {/* Dynamic Weight Promotion vs Activation Offload Simulation OR Slowed Recorded Trace Playback */}
        {traceReplay ? (
          <TraceDemo3D
            explode={explode}
            activeStep={activeTraceStep ?? 0}
            isPlaying={tracePlaying ?? false}
          />
        ) : (
          <MigrationDemo mode={mode} playToken={playToken} explode={explode} />
        )}
      </Suspense>

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.12}
        autoRotate={autoRotate}
        autoRotateSpeed={0.7}
        minDistance={6}
        maxDistance={32}
        maxPolarAngle={Math.PI * 0.49}
      />

      <AdaptiveDpr pixelated />
    </Canvas>
  );
}
