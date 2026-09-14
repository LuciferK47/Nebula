import React from 'react';
import { Html } from '@react-three/drei';
import { DIE_Y, PLATE_DEPTH, PLATE_WIDTH } from './layout';

interface SubstratePackageProps {
  explode: number;
}

/**
 * High-Density Organic Package Substrate with Ball Grid Array (BGA).
 * Bridges the silicon interposer to the system motherboard/accelerator board.
 */
export function SubstratePackage({ explode }: SubstratePackageProps) {
  // Sits directly below the Silicon Interposer
  const substrateY = (DIE_Y - 0.7) * (1 + explode * 0.9);
  const substrateW = PLATE_WIDTH * 0.92;
  const substrateD = PLATE_DEPTH * 0.78;
  const thickness = 0.16;

  const [hovered, setHovered] = React.useState(false);

  return (
    <group position={[0, substrateY, 0]}>
      {/* Organic Substrate Core */}
      <mesh
        position={[0, -thickness / 2, 0]}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
        }}
        onPointerOut={() => setHovered(false)}
      >
        <boxGeometry args={[substrateW, thickness, substrateD]} />
        <meshStandardMaterial
          color="#0f241a"
          roughness={0.6}
          metalness={0.3}
        />
      </mesh>

      {/* Surface Gold Contact Pads */}
      <mesh position={[0, 0.005, 0]}>
        <boxGeometry args={[substrateW * 0.96, 0.01, substrateD * 0.96]} />
        <meshStandardMaterial
          color="#163828"
          roughness={0.5}
          metalness={0.4}
        />
      </mesh>

      {/* Decoupling Capacitors (SMD components on substrate surface) */}
      {[-substrateW * 0.44, substrateW * 0.44].map((x, i) => (
        <group key={`caps-${i}`} position={[x, 0.03, 0]}>
          {Array.from({ length: 6 }).map((_, c) => (
            <mesh key={`cap-${c}`} position={[0, 0, (c - 2.5) * 0.4]}>
              <boxGeometry args={[0.18, 0.06, 0.24]} />
              <meshStandardMaterial color="#886842" roughness={0.3} metalness={0.7} />
            </mesh>
          ))}
        </group>
      ))}

      {/* Solder Ball Grid Array (BGA) underneath */}
      <mesh position={[0, -thickness - 0.04, 0]}>
        <boxGeometry args={[substrateW * 0.88, 0.06, substrateD * 0.88]} />
        <meshStandardMaterial
          color="#5c6068"
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>

      {hovered && (
        <Html position={[0, 0.5, 0]} center style={{ pointerEvents: 'none' }}>
          <div className="whitespace-nowrap rounded-md bg-ink px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wide text-cream shadow-xl border border-ink-line">
            <span className="text-emerald-400">ORGANIC SUBSTRATE · </span>
            <span>Multilayer Package & BGA Array to Host Accelerator Board</span>
          </div>
        </Html>
      )}

      {/* Label */}
      <Html
        position={[-substrateW / 2 - 0.7, -0.05, 0]}
        center
        style={{ pointerEvents: 'none' }}
      >
        <div className="rounded-full bg-ink/90 px-2 py-0.5 font-mono text-[8px] uppercase tracking-wider text-khaki/70 border border-white/10">
          Package Substrate
        </div>
      </Html>
    </group>
  );
}
