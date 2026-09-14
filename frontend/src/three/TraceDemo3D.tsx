import React, { useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { DIE_Y, TIER_Y } from './layout';
import { RECORDED_S3_TRACE, type RecordedTraceEvent } from '../data/traceEvents';

interface TraceDemo3DProps {
  explode: number;
  activeStep: number;
  isPlaying: boolean;
}

export function TraceDemo3D({ explode, activeStep, isPlaying }: TraceDemo3DProps) {
  const scale = 1 + explode * 0.9;
  const event: RecordedTraceEvent = RECORDED_S3_TRACE[activeStep] || RECORDED_S3_TRACE[0];

  const packetRef = useRef<THREE.Mesh>(null!);
  const pulseRingRef = useRef<THREE.Mesh>(null!);

  const dieY = DIE_Y * scale;
  const targetY = (event.tier === 'hbm' ? TIER_Y.hbm : event.tier === 'dram' ? TIER_Y.dram : TIER_Y.cxl) * scale;

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (packetRef.current) {
      if (event.tier === 'hbm') {
        // Local GPU hit: orbit slightly or pulse in place at compute die
        packetRef.current.position.y = THREE.MathUtils.lerp(dieY, targetY, 0.5) + Math.sin(t * 8) * 0.1;
        packetRef.current.scale.setScalar(1 + Math.sin(t * 10) * 0.2);
      } else {
        // Interconnect transit: travel between Compute Die and target memory plate
        const cycle = (t * 2.5) % 1;
        packetRef.current.position.y = THREE.MathUtils.lerp(dieY, targetY, cycle);
        packetRef.current.scale.setScalar(1);
      }
    }

    if (pulseRingRef.current) {
      const ringScale = 1 + (t * 2 % 1) * 0.8;
      pulseRingRef.current.scale.set(ringScale, 1, ringScale);
    }
  });

  const tierColor = event.tierColor;

  return (
    <group>
      {/* Laser Transfer Beam between Die and Target Tier */}
      {event.tier !== 'hbm' && (
        <mesh position={[0, (dieY + targetY) / 2, 0]}>
          <cylinderGeometry args={[0.04, 0.04, Math.abs(dieY - targetY), 12]} />
          <meshStandardMaterial
            color={tierColor}
            emissive={tierColor}
            emissiveIntensity={1.2}
            transparent
            opacity={0.7}
          />
        </mesh>
      )}

      {/* Traveling Data Packet (Sphere) */}
      <mesh ref={packetRef} position={[0, (dieY + targetY) / 2, 0]}>
        <sphereGeometry args={[event.sizeBytes > 1000000 ? 0.32 : 0.2, 16, 16]} />
        <meshStandardMaterial
          color={tierColor}
          emissive={tierColor}
          emissiveIntensity={1.8}
        />
      </mesh>

      {/* Target Tier Target Highlight Pulse Ring */}
      <group position={[0, targetY + 0.1, 0]}>
        <mesh ref={pulseRingRef} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[1.2, 1.4, 32]} />
          <meshBasicMaterial
            color={tierColor}
            transparent
            opacity={0.8}
            side={THREE.DoubleSide}
          />
        </mesh>
      </group>

      {/* Floating 3D Telemetry Tooltip on Active Hardware Layer */}
      <Html position={[3.2, targetY + 0.4, 0]} center style={{ pointerEvents: 'none' }}>
        <div
          className="whitespace-nowrap rounded-lg px-2.5 py-1.5 font-mono text-[10px] shadow-2xl backdrop-blur transition-all duration-200 border"
          style={{
            backgroundColor: 'rgba(10, 11, 15, 0.95)',
            borderColor: `${tierColor}aa`,
            boxShadow: `0 0 16px ${tierColor}44`,
          }}
        >
          <div className="flex items-center gap-2">
            <span
              className="h-2 w-2 rounded-full animate-ping"
              style={{ backgroundColor: tierColor }}
            />
            <strong className="font-bold" style={{ color: tierColor }}>
              {event.tierName.toUpperCase()}
            </strong>
            <span className="text-white/60">·</span>
            <span className="text-cream font-semibold">{event.accessType}</span>
            <span className="rounded bg-white/10 px-1 py-0.2 text-[9px] text-cream/90">
              {event.sizeFormatted}
            </span>
          </div>
          <div className="mt-1 flex items-center justify-between gap-3 text-[9px] text-cream/60 border-t border-white/10 pt-1">
            <span>Addr: <code className="text-white/80">{event.addressHex.slice(0, 8)}…</code></span>
            <span className="font-semibold text-emerald-400">{event.latency}</span>
          </div>
        </div>
      </Html>
    </group>
  );
}
