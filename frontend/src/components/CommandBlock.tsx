import { useCallback, useEffect, useRef, useState } from 'react';
import { CheckIcon, CopyIcon } from 'lucide-react';

interface CommandBlockProps {
  command: string;
  size?: 'lg' | 'sm';
  tone?: 'ink' | 'cream';
  className?: string;
}

/**
 * A real, copyable install command styled as the primary action —
 * not a button that hides the command behind a docs link.
 */
export function CommandBlock({
  command,
  size = 'lg',
  tone = 'ink',
  className = ''
}: CommandBlockProps) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(command);
    } catch {


      // Clipboard unavailable (insecure context or denied permission):
      // the command is still fully visible and selectable.
    }setCopied(true);if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setCopied(false), 1600);
  }, [command]);

  const shell =
  tone === 'ink' ?
  'bg-ink text-cream hover:bg-[#2c2c28]' :
  'bg-cream text-ink hover:bg-offwhite';
  const dim = tone === 'ink' ? 'text-khaki' : 'text-ink/45';
  const pad = size === 'lg' ? 'gap-4 py-4 pl-6 pr-4 text-[0.8rem]' : 'gap-3 py-2.5 pl-4 pr-3 text-11';

  return (
    <button
      type="button"
      onClick={copy}
      className={`group inline-flex items-center rounded-full font-mono transition-colors duration-150 ease-out ${shell} ${pad} ${className}`}
      aria-label={`Copy command: ${command}`}>
      
      <span aria-hidden="true" className={dim}>
        $
      </span>
      <span className="tracking-wide">{command}</span>
      <span
        className={`ml-1 inline-flex h-7 w-7 items-center justify-center rounded-full ${
        tone === 'ink' ? 'bg-white/10' : 'bg-ink/10'}`
        }>
        
        {copied ?
        <CheckIcon size={13} strokeWidth={2.2} aria-hidden="true" /> :

        <CopyIcon size={13} strokeWidth={1.8} aria-hidden="true" />
        }
      </span>
      <span className="sr-only" role="status">
        {copied ? 'Command copied to clipboard' : ''}
      </span>
    </button>);

}