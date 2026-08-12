/**
 * isotope-zero - Main Entry Point
 * Sub-millisecond, local-first cognitive memory layer for AI agents
 */

export { runTUI } from './tui/App.ts';
export { default as App } from './tui/App.ts';

// Re-export core utilities
export * from './logger.ts';
export * from './keyboard.ts';
export * from './constants.ts';

// Re-export TUI components
export * from './tui/components/index.ts';
export * from './tui/theme/index.ts';
export * from './tui/hooks/useKeyboard.ts';
export * from './tui/hooks/useTheme.ts';

// Types
export type { ThemeName, ThemeColors } from './tui/theme/colors.ts';
export type { SpacingValue, PaddingValue, MarginValue } from './tui/theme/spacing.ts';