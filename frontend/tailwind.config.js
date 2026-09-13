export default {
  content: [
  './index.html',
  './src/**/*.{js,ts,jsx,tsx}'
],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: '#141414',
          soft: '#1b1b19',
          panel: '#1f1f1c',
          line: '#32322c',
        },
        cream: '#f4f3ed',
        paper: '#fbfaf5',
        offwhite: '#f0f0ec',
        khaki: '#c8c8be',
        olive: '#77784f',
        clay: '#b4623c',
        sunflower: '#e5b52f',
        petal: '#e7b6c4',
        flare: '#e4512b',
        amber: '#f5b32b',
        hot: '#e4512b',
        warm: '#e5b52f',
        cold: '#8fa3b8',
        ambient: '#7c6bd6',
      },
      fontFamily: {
        display: ['"Bodoni Moda"', 'Didot', 'Georgia', 'serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      fontSize: {
        '10': ['0.625rem', { lineHeight: '1rem' }],
        '11': ['0.6875rem', { lineHeight: '1.05rem' }],
      },
      letterSpacing: {
        label: '0.16em',
        wide: '0.1em',
      },
      transitionTimingFunction: {
        expo: 'cubic-bezier(0.23, 1, 0.32, 1)',
      },
      maxWidth: {
        site: '78rem',
      },
    },
  },
  plugins: [],
}
