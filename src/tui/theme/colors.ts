import chalk from 'chalk';

export type ThemeName = 'ocean-blue' | 'forest-green' | 'sunset-orange' | 'monochrome';

export interface ThemeColors {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  surface: string;
  border: string;
  foreground: string;
  muted: string;
  subtle: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  selection: string;
  syntax: {
    keyword: string;
    string: string;
    number: string;
    comment: string;
    function: string;
  };
}

export const themes: Record<ThemeName, ThemeColors> = {
  'ocean-blue': {
    primary: '#06b6d4', // Cyan 500
    secondary: '#3b82f6', // Blue 500
    accent: '#a855f7', // Purple 500
    background: '#0f172a', // Slate 900
    surface: '#1e293b', // Slate 800
    border: '#334155', // Slate 700
    foreground: '#f8fafc', // Slate 50
    muted: '#94a3b8', // Slate 400
    subtle: '#475569', // Slate 600
    success: '#22c55e',
    warning: '#eab308',
    error: '#ef4444',
    info: '#3b82f6',
    selection: '#1e3a8a',
    syntax: {
      keyword: '#f472b6',
      string: '#4ade80',
      number: '#fbbf24',
      comment: '#64748b',
      function: '#818cf8',
    },
  },
  'forest-green': {
    primary: '#22c55e',
    secondary: '#16a34a',
    accent: '#84cc16',
    background: '#052e16',
    surface: '#14532d',
    border: '#166534',
    foreground: '#f0fdf4',
    muted: '#86efac',
    subtle: '#3f6212',
    success: '#4ade80',
    warning: '#fbbf24',
    error: '#f87171',
    info: '#60a5fa',
    selection: '#064e3b',
    syntax: {
      keyword: '#bef264',
      string: '#4ade80',
      number: '#facc15',
      comment: '#4d7c0f',
      function: '#34d399',
    },
  },
  'sunset-orange': {
    primary: '#fb923c',
    secondary: '#f97316',
    accent: '#f472b6',
    background: '#450a0a',
    surface: '#7f1d1d',
    border: '#991b1b',
    foreground: '#fff7ed',
    muted: '#fdba74',
    subtle: '#b91c1c',
    success: '#4ade80',
    warning: '#fbbf24',
    error: '#ef4444',
    info: '#60a5fa',
    selection: '#7f1d1d',
    syntax: {
      keyword: '#fbcfe8',
      string: '#fbbf24',
      number: '#fda4af',
      comment: '#7f1d1d',
      function: '#fca5a5',
    },
  },
  'monochrome': {
    primary: '#ffffff',
    secondary: '#cccccc',
    accent: '#aaaaaa',
    background: '#000000',
    surface: '#111111',
    border: '#333333',
    foreground: '#ffffff',
    muted: '#888888',
    subtle: '#444444',
    success: '#ffffff',
    warning: '#cccccc',
    error: '#aaaaaa',
    info: '#ffffff',
    selection: '#222222',
    syntax: {
      keyword: '#ffffff',
      string: '#cccccc',
      number: '#cccccc',
      comment: '#888888',
      function: '#ffffff',
    },
  },
};

export function getTheme(name: ThemeName = 'ocean-blue'): ThemeColors {
  return themes[name] || themes['ocean-blue'];
}
