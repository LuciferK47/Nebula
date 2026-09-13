import React from 'react';

interface KickerProps {
  children: React.ReactNode;
  className?: string;
  as?: 'span' | 'p' | 'h2' | 'h3' | 'div';
  id?: string;
}

/** Small tracked-out uppercase monospace label. Used for every non-headline. */
export function Kicker({ children, className = '', as = 'span', id }: KickerProps) {
  const Tag = as;
  return (
    <Tag id={id} className={`font-mono text-10 font-medium uppercase tracking-label ${className}`}>
      {children}
    </Tag>);

}