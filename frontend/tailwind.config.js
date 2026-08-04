/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      // Opacity steps the glass surfaces rely on that Tailwind omits by default.
      opacity: { 8: '0.08', 12: '0.12', 15: '0.15' },
    },
  },
  plugins: [],
};
