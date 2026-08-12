/**
 * Application constants
 * Centralized configuration values for the isotope-zero CLI
 */

export const APP_NAME = 'isotope-zero';
export const VERSION = '1.3.2';

export const DEFAULT_DB_PATH = '~/.isotope-zero/history.db';

export const REFRESH_INTERVALS = {
  FAST: 1000,      // 1 second
  NORMAL: 5000,    // 5 seconds
  SLOW: 30000,     // 30 seconds
  MANUAL: 0,       // No auto-refresh
} as const;

export const MAX_HISTORY = 10000;

export const KEYBINDINGS_DEFAULT = {
  quit: ['q', 'Ctrl+c'],
  help: ['?', 'h'],
  refresh: ['r', 'F5'],
  nextView: ['Tab', '>'],
  prevView: ['Shift+Tab', '<'],
  search: ['/', 'Ctrl+f'],
  clearSearch: ['Escape'],
  select: ['Enter', 'Space'],
  expand: ['Right', 'l'],
  collapse: ['Left', 'h'],
  scrollUp: ['k', 'Up'],
  scrollDown: ['j', 'Down'],
  pageUp: ['Ctrl+u', 'PageUp'],
  pageDown: ['Ctrl+d', 'PageDown'],
  gotoTop: ['g', 'Home'],
  gotoBottom: ['G', 'End'],
  toggleSort: ['s'],
  toggleFilter: ['f'],
  copy: ['y', 'Ctrl+c'],
  open: ['o', 'Enter'],
} as const;

export const VIEW_ORDER = [
  'dashboard',
  'services',
  'deployments',
  'logs',
  'metrics',
  'config',
  'help',
] as const;

export const LOG_LEVELS = [
  'debug',
  'info',
  'warn',
  'error',
] as const;

export const DEFAULT_LOG_LEVEL = 'info';

export const THEMES = [
  'ocean-blue',
  'forest-green',
  'sunset-orange',
  'monochrome',
] as const;

export const DEFAULT_THEME = 'ocean-blue';

export const CONFIG_FILE_NAME = 'config.json';
export const CONFIG_DIR_NAME = '.isotope-zero';

export const PLUGIN_DIR_NAME = 'plugins';

export const HEALTH_CHECK_INTERVAL = 10000;
export const HEALTH_CHECK_TIMEOUT = 5000;

export const MAX_CONCURRENT_OPERATIONS = 10;
export const DEFAULT_TIMEOUT = 30000;

export const STYLE = {
  PRIMARY: '#0077cc',
  SUCCESS: '#28a745',
  WARNING: '#ffc107',
  DANGER: '#dc3545',
  INFO: '#17a2b8',
  MUTED: '#6c757d',
} as const;

export const BORDERS = {
  THIN: '─│┌┐└┘├┤┬┴┼',
  THICK: '━┃┏┓┗┛┣┫┳┻╋',
  DOUBLE: '═║╔╗╚╝╠╣╦╩╬',
  ROUNDED: '─│┌┐└┘├┤┬┴┼',
} as const;

export const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
export const SPINNER_INTERVAL = 80;

export const TABLE_COLUMNS = {
  MIN_WIDTH: 10,
  MAX_WIDTH: 80,
  PADDING: 2,
} as const;

export const SEARCH = {
  MIN_QUERY_LENGTH: 1,
  DEBOUNCE_MS: 150,
  HIGHLIGHT_CLASS: 'search-highlight',
} as const;

export const NOTIFICATION = {
  DURATION_MS: 3000,
  MAX_VISIBLE: 5,
  POSITION: 'top-right',
} as const;

export const MODAL = {
  Z_INDEX: 1000,
  BACKDROP_OPACITY: 0.5,
  ANIMATION_DURATION: 200,
} as const;

export type RefreshInterval = typeof REFRESH_INTERVALS[keyof typeof REFRESH_INTERVALS];
export type LogLevel = typeof LOG_LEVELS[number];
export type Theme = typeof THEMES[number];
export type ViewName = typeof VIEW_ORDER[number];
export type KeybindingAction = keyof typeof KEYBINDINGS_DEFAULT;
export type KeybindingKeys = typeof KEYBINDINGS_DEFAULT[KeybindingAction];

export const isValidView = (view: string): view is ViewName => {
  return VIEW_ORDER.includes(view as ViewName);
};

export const isValidTheme = (theme: string): theme is Theme => {
  return THEMES.includes(theme as Theme);
};

export const isValidLogLevel = (level: string): level is LogLevel => {
  return LOG_LEVELS.includes(level as LogLevel);
};

export const getDefaultConfigPath = (): string => {
  const home = process.env.HOME || process.env.USERPROFILE || '~';
  return `${home}/${CONFIG_DIR_NAME}/${CONFIG_FILE_NAME}`;
};

export const getDefaultDbPath = (): string => {
  const home = process.env.HOME || process.env.USERPROFILE || '~';
  return DEFAULT_DB_PATH.replace('~', home);
};

export const getPluginDir = (): string => {
  const home = process.env.HOME || process.env.USERPROFILE || '~';
  return `${home}/${CONFIG_DIR_NAME}/${PLUGIN_DIR_NAME}`;
};