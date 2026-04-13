// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — Design Tokens
// Centralized theme: colors, fonts, spacing, shadows
// ═══════════════════════════════════════════════════════════════════════

export const colors = {
  bg: {
    primary: '#030508',
    panel: '#080d1c',
    input: '#0d1830',
    border: '#1a2845',
    glass: 'rgba(5,9,20,0.98)',
    overlay: 'rgba(6,10,20,0.97)',
  },
  accent: {
    cyan: '#00e5ff',
    purple: '#d500f9',
    green: '#00e676',
    red: '#ff1744',
    orange: '#ff6d00',
    amber: '#ffab00',
    pink: '#ff4081',
    lime: '#76ff03',
    blue: '#00b0ff',
    orangeDeep: '#ff6e40',
    greenLight: '#69f0ae',
  },
  text: {
    primary: '#ddeeff',
    secondary: '#7a9ac0',
    muted: '#5a7090',
    dim: '#4a6080',
    dimmer: '#3a5070',
    dimmest: '#2a4060',
    label: '#1a3050',
  },
  threat: {
    INSTRUCTION_OVERRIDE: '#ff1744',
    ROLE_OVERRIDE: '#ff6d00',
    DATA_EXFIL: '#ffab00',
    INDIRECT_INJECTION: '#d500f9',
    OBFUSCATED: '#00b0ff',
    CONTEXT_OVERFLOW: '#76ff03',
    MULTI_TURN: '#ff4081',
    SOCIAL_ENG: '#ff6e40',
  } as Record<string, string>,
} as const;

export const fonts = {
  mono: "'Share Tech Mono', monospace",
  display: "'Orbitron', monospace",
  body: "'Barlow', sans-serif",
} as const;

export const spacing = {
  xs: 4,
  sm: 6,
  md: 8,
  lg: 12,
  xl: 16,
} as const;
