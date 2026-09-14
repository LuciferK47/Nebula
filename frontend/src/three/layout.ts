import * as THREE from 'three';

export type TierId = 'hbm' | 'dram' | 'cxl';

export const TIER_COLOR: Record<TierId, string> = {
  hbm: '#e4512b',
  dram: '#e5b52f',
  cxl: '#8fa3b8'
};

export const TIER_Y: Record<TierId, number> = { hbm: 3.2, dram: 0, cxl: -3.2 };
export const DIE_Y = 5.6;

// The real model behind S1/S2/S3: 24 transformer layers, 4 experts/layer
// (Qwen1.5-4x0.5B-Chat-MoE, per Nebula/Readme.md and
// docs/SYSTEM_DOCUMENTATION.md). One instance per (layer, expert) pair.
export const NUM_LAYERS = 24;
export const EXPERTS_PER_LAYER = 4;
export const TOTAL_EXPERTS = NUM_LAYERS * EXPERTS_PER_LAYER;

const FOOTPRINT_X = 9;
const FOOTPRINT_Z = 9;

export function cellIndex(layer: number, expert: number): number {
  return layer * EXPERTS_PER_LAYER + expert;
}

export function cellX(expert: number): number {
  const pitch = FOOTPRINT_X / EXPERTS_PER_LAYER;
  return (expert - (EXPERTS_PER_LAYER - 1) / 2) * pitch;
}

export function cellZ(layer: number): number {
  const pitch = FOOTPRINT_Z / NUM_LAYERS;
  return (layer - (NUM_LAYERS - 1) / 2) * pitch;
}

export const CELL_SIZE_X = (FOOTPRINT_X / EXPERTS_PER_LAYER) * 0.72;
export const CELL_SIZE_Z = (FOOTPRINT_Z / NUM_LAYERS) * 0.72;
export const CELL_HEIGHT = 0.32;

export const PLATE_WIDTH = FOOTPRINT_X + 1.4;
export const PLATE_DEPTH = FOOTPRINT_Z + 1.4;

/**
 * Deterministic illustrative placement matching the REAL per-tier hit
 * fractions from results/scenarios/s3.json (hbm 39.94%, dram 23.74%, cxl
 * 36.33% — the one scenario config that exercises all three tiers). The
 * exported scenario data reports aggregate hit rates, not which specific
 * expert sits where, so this assigns cells round-robin in that proportion
 * rather than claiming to know the true per-expert residency.
 */
export function illustrativePlacement(hbmFrac: number, dramFrac: number): TierId[] {
  // A pure sequential cut (first N cells HBM, next M DRAM, rest CXL) would
  // cluster every HBM cell in the first few layers and every CXL cell in
  // the last few — the opposite of how an LFU cache actually distributes
  // hot experts across layers. Sort by a low-discrepancy key first so the
  // same three proportions land spread across all 24 layers instead.
  const GOLDEN = 0.6180339887;
  const order = Array.from({ length: TOTAL_EXPERTS }, (_, i) => i).
  sort((a, b) => (a * GOLDEN) % 1 - (b * GOLDEN) % 1);

  const placement: TierId[] = new Array(TOTAL_EXPERTS);
  const hbmCount = Math.round(hbmFrac * TOTAL_EXPERTS);
  const dramCount = Math.round(dramFrac * TOTAL_EXPERTS);
  order.forEach((cellIdx, rank) => {
    placement[cellIdx] = rank < hbmCount ? 'hbm' : rank < hbmCount + dramCount ? 'dram' : 'cxl';
  });
  return placement;
}

export const tmpMatrix = new THREE.Matrix4();
export const tmpPosition = new THREE.Vector3();
export const tmpQuaternion = new THREE.Quaternion();
export const tmpScale = new THREE.Vector3();
export const tmpColor = new THREE.Color();
