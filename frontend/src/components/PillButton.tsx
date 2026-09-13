import React from 'react';

type Variant = 'solid' | 'outline' | 'solid-ink' | 'outline-ink';

const base =
'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-full px-6 py-3 font-mono text-11 font-medium uppercase tracking-label transition-colors duration-150 ease-out';

const variants: Record<Variant, string> = {
  solid: 'bg-ink text-cream hover:bg-[#2c2c28]',
  outline: 'border border-ink/25 bg-transparent text-ink hover:border-ink hover:bg-ink hover:text-cream',
  // On near-black sections the polarity flips, but the system stays the same:
  // one solid button, one outline, no shadows, no gradients.
  'solid-ink': 'bg-cream text-ink hover:bg-amber',
  'outline-ink':
  'border border-ink-line bg-transparent text-cream hover:border-amber hover:text-amber'
};

interface PillButtonProps {
  href: string;
  children: React.ReactNode;
  variant?: Variant;
  className?: string;
}

export function PillButton({ href, children, variant = 'solid', className = '' }: PillButtonProps) {
  return (
    <a href={href} className={`${base} ${variants[variant]} ${className}`}>
      {children}
    </a>);

}