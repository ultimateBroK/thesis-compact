import { createExternalPackageIconLoader } from '@iconify/utils/lib/loader/external-pkg'
import config from '@slidev/client/uno.config'
import { mergeConfigs, presetAttributify, presetIcons, presetWebFonts, presetWind3 } from 'unocss'

export default mergeConfigs([
  config,
  {
    rules: [
      ['font-mono-tight', { 'font-family': '"Fira Code", ui-monospace, monospace', 'letter-spacing': '-0.01em' }],
      ['font-vn', { 'font-family': '"Be Vietnam Pro", "DM Sans", ui-sans-serif, system-ui, sans-serif' }],
      ['font-display', { 'font-family': '"Noto Serif", "Be Vietnam Pro", ui-serif, Georgia, serif' }],
    ],
    safelist: [
      ...Array.from({ length: 30 }, (_, i) => `delay-${(i + 1) * 100}`),
      'animate-pulse',
      'animate-blink',
    ],
    presets: [
      presetWind3({
        dark: 'class',
      }),
      presetAttributify(),
      presetIcons({
        prefix: 'i-',
        scale: 1.2,
        extraProperties: {
          display: 'inline-block',
          'vertical-align': 'middle',
        },
        warn: true,
        collections: {
          ...createExternalPackageIconLoader('@iconify-json/carbon'),
          ...createExternalPackageIconLoader('@iconify-json/ri'),
          ...createExternalPackageIconLoader('@iconify-json/devicon'),
          ...createExternalPackageIconLoader('@iconify-json/simple-icons'),
          ...createExternalPackageIconLoader('@iconify-json/logos'),
          ...createExternalPackageIconLoader('@iconify-json/ph'),
          ...createExternalPackageIconLoader('@iconify-json/fluent'),
          ...createExternalPackageIconLoader('@iconify-json/icon-park-outline'),
          ...createExternalPackageIconLoader('@iconify-json/twemoji'),
          ...createExternalPackageIconLoader('@iconify-json/svg-spinners'),
          ...createExternalPackageIconLoader('@iconify-json/bi'),
        },
      }),
      presetWebFonts({
        fonts: {
          sans: 'DM Sans',
          vn: 'Be Vietnam Pro',
          hand: 'Playwrite IT Moderna',
          serif: 'Noto Serif',
          mono: 'Fira Code',
        },
      }),
    ],
  },
])
