import React, { useId } from 'react';

type Tone = 'khaki' | 'ink' | 'dark' | 'amber';

const strokeFor: Record<Tone, string> = {
  khaki: '#c8c8be',
  ink: '#141414',
  dark: '#32322c',
  amber: '#f5b32b'
};

interface WobbleRuleProps {
  tone?: Tone;
  className?: string;
  /** Varies the noise field so no two rules wobble identically. */
  seed?: number;
}

/**
 * A structural hairline rendered as an SVG line pushed through an
 * feTurbulence displacement map — straight, but with a small organic
 * wobble instead of a razor-ruled edge.
 */
export function WobbleRule({ tone = 'khaki', className = '', seed = 3 }: WobbleRuleProps) {
  const raw = useId().replace(/[:]/g, '');
  const filterId = `wobble-${raw}`;

  return (
    <svg
      aria-hidden="true"
      focusable="false"
      className={`block h-[7px] w-full ${className}`}
      viewBox="0 0 1200 7"
      preserveAspectRatio="none">
      
      <defs>
        <filter id={filterId} x="-1%" y="-300%" width="102%" height="700%">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.014 0.8"
            numOctaves={2}
            seed={seed}
            result="noise" />
          
          <feDisplacementMap
            in="SourceGraphic"
            in2="noise"
            scale={3.4}
            xChannelSelector="R"
            yChannelSelector="G" />
          
        </filter>
      </defs>
      <line
        x1="0"
        y1="3.5"
        x2="1200"
        y2="3.5"
        stroke={strokeFor[tone]}
        strokeWidth="1"
        filter={`url(#${filterId})`} />
      
    </svg>);

}

interface WobbleCircleProps {
  children: React.ReactNode;
  tone?: Tone;
  size?: number;
  className?: string;
  seed?: number;
}

/** Circular badge whose border gets the same displaced-hairline treatment. */
export function WobbleCircle({
  children,
  tone = 'ink',
  size = 44,
  className = '',
  seed = 7
}: WobbleCircleProps) {
  const raw = useId().replace(/[:]/g, '');
  const filterId = `wobble-circle-${raw}`;

  return (
    <span
      className={`relative inline-flex shrink-0 items-center justify-center ${className}`}
      style={{ width: size, height: size }}>
      
      <svg
        aria-hidden="true"
        focusable="false"
        className="absolute inset-0"
        viewBox="0 0 100 100">
        
        <defs>
          <filter id={filterId} x="-15%" y="-15%" width="130%" height="130%">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.05"
              numOctaves={2}
              seed={seed}
              result="noise" />
            
            <feDisplacementMap
              in="SourceGraphic"
              in2="noise"
              scale={5}
              xChannelSelector="R"
              yChannelSelector="G" />
            
          </filter>
        </defs>
        <circle
          cx="50"
          cy="50"
          r="47"
          fill="none"
          stroke={strokeFor[tone]}
          strokeWidth="1.4"
          filter={`url(#${filterId})`} />
        
      </svg>
      <span className="relative">{children}</span>
    </span>);

}