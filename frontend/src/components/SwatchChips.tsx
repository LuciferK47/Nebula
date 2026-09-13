import React from 'react';

interface SwatchChipsProps {
  colors: string[];
  className?: string;
  /** Rendered chip width in px. Height is fixed thin. */
  width?: number;
  vertical?: boolean;
}

/**
 * A colour-palette strip broken into pieces and scattered as decoration.
 * Purely ornamental — hidden from assistive tech.
 */
export function SwatchChips({
  colors,
  className = '',
  width = 22,
  vertical = false
}: SwatchChipsProps) {
  return (
    <span
      aria-hidden="true"
      className={`pointer-events-none inline-flex ${vertical ? 'flex-col' : 'flex-row'} gap-[3px] ${className}`}>
      
      {colors.map((color, i) =>
      <span
        key={`${color}-${i}`}
        className="block rounded-[1px]"
        style={{
          backgroundColor: color,
          width: vertical ? 6 : width,
          height: vertical ? width : 6
        }} />

      )}
    </span>);

}