import type { Transition, Variants } from 'framer-motion';

/** A curve with enough authority to read as intentional. */
export const expo = [0.23, 1, 0.32, 1] as const;

export const enter: Transition = { duration: 0.28, ease: expo };

export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: enter }
};

export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.24, ease: expo } }
};

/** Caps total sequence length so the last item never feels late. */
export const stagger = (delayChildren = 0): Variants => ({
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.045, delayChildren }
  }
});

export const inView = { once: true, amount: 0.25 } as const;