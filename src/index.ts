/**
 * isotope-zero — library entry point.
 *
 * Re-exports the TUI renderer, the React root component, the CLI dispatcher,
 * and the TUI building blocks. Importing this module has no side effects (it
 * does NOT launch the terminal UI — use the `izero` bin or `runTUI()` for that).
 */
export { runTUI, default as App } from './tui/App.js';
export { main as cli } from './cli.js';
export { runPythonCommand } from './pybridge.js';

// TUI components
export * from './tui/components/index.js';

// TUI hooks
export { useKeyboardShortcuts, useListNavigation, useForm } from './tui/hooks/useKeyboard.js';
export { useTheme } from './tui/hooks/useTheme.js';

// TUI theme
export { getTheme, themes, type ThemeName, type ThemeColors } from './tui/theme/colors.js';
export { Dimensions, flexStyles } from './tui/theme/spacing.js';