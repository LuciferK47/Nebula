/** Key + fill + rim lighting with accent lighting for silicon die, HBM stacks and interconnects. */
export function SceneLights() {
  return (
    <>
      {/* Base uniform ambient light for clear geometry visibility */}
      <ambientLight intensity={0.9} color="#ffffff" />

      {/* Hemisphere light providing natural sky/ground contrast */}
      <hemisphereLight args={['#ffffff', '#18181f', 0.8]} />

      {/* Primary Key Directional Studio Light */}
      <directionalLight position={[10, 18, 12]} intensity={2.2} color="#ffffff" />

      {/* Cool Rim Light highlighting silicon chamfers, gold pins, and heatsink fins */}
      <directionalLight position={[-12, 12, -10]} intensity={1.5} color="#93c5fd" />

      {/* Dedicated Fill Light for lower tiers (Host DRAM & CXL far memory chassis) */}
      <pointLight position={[0, -2, 5]} intensity={2.0} color="#e0e7ff" distance={18} />

      {/* Focused Accelerator Spotlight illuminating Compute Die and HBM stacks */}
      <pointLight position={[0, 6, 3]} intensity={2.2} color="#fef08a" distance={15} />

      {/* Subtle under-chassis bounce light */}
      <directionalLight position={[0, -8, -6]} intensity={0.75} color="#cbd5e1" />
    </>
  );
}
