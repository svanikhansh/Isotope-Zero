/**
 * Keyboard Input Helpers and Shortcut Manager
 * Provides utilities for parsing, formatting, and managing keyboard shortcuts
 */

export interface KeyCombination {
  key: string;
  ctrl?: boolean;
  alt?: boolean;
  shift?: boolean;
  meta?: boolean;
}

export interface ShortcutHandler {
  (event: KeyboardEvent): void | boolean;
}

export interface RegisteredShortcut {
  combination: KeyCombination;
  handler: ShortcutHandler;
  view?: string;
  description?: string;
  preventDefault?: boolean;
  stopPropagation?: boolean;
}

export interface ViewShortcuts {
  [view: string]: RegisteredShortcut[];
}

/**
 * Normalizes a key string to standard format
 */
function normalizeKey(key: string): string {
  const keyMap: Record<string, string> = {
    'escape': 'Escape',
    'esc': 'Escape',
    'enter': 'Enter',
    'tab': 'Tab',
    'backspace': 'Backspace',
    'delete': 'Delete',
    'del': 'Delete',
    'arrowup': 'ArrowUp',
    'arrowdown': 'ArrowDown',
    'arrowleft': 'ArrowLeft',
    'arrowright': 'ArrowRight',
    'up': 'ArrowUp',
    'down': 'ArrowDown',
    'left': 'ArrowLeft',
    'right': 'ArrowRight',
    'home': 'Home',
    'end': 'End',
    'pageup': 'PageUp',
    'pagedown': 'PageDown',
    'insert': 'Insert',
    'space': ' ',
    'spacebar': ' ',
    'f1': 'F1',
    'f2': 'F2',
    'f3': 'F3',
    'f4': 'F4',
    'f5': 'F5',
    'f6': 'F6',
    'f7': 'F7',
    'f8': 'F8',
    'f9': 'F9',
    'f10': 'F10',
    'f11': 'F11',
    'f12': 'F12',
  };

  const lower = key.toLowerCase();
  return keyMap[lower] || key;
}

/**
 * Parses a shortcut string into a KeyCombination object
 * Examples: "Ctrl+P", "Ctrl+Shift+K", "Escape", "Alt+F4"
 */
export function parseKey(shortcut: string): KeyCombination {
  const parts = shortcut.split('+').map(p => p.trim().toLowerCase());
  const combination: KeyCombination = { key: '' };

  for (const part of parts) {
    switch (part) {
      case 'ctrl':
      case 'control':
        combination.ctrl = true;
        break;
      case 'alt':
      case 'option':
        combination.alt = true;
        break;
      case 'shift':
        combination.shift = true;
        break;
      case 'meta':
      case 'cmd':
      case 'command':
      case 'win':
      case 'super':
        combination.meta = true;
        break;
      default:
        combination.key = normalizeKey(part);
    }
  }

  return combination;
}

/**
 * Formats a KeyCombination into a human-readable shortcut string
 */
export function formatShortcut(combination: KeyCombination): string {
  const parts: string[] = [];

  if (combination.ctrl) parts.push('Ctrl');
  if (combination.alt) parts.push('Alt');
  if (combination.shift) parts.push('Shift');
  if (combination.meta) parts.push('Meta');
  if (combination.key) parts.push(combination.key);

  return parts.join('+');
}

/**
 * Checks if a keyboard event matches a key combination
 */
function matchesCombination(event: KeyboardEvent, combination: KeyCombination): boolean {
  if (combination.ctrl !== undefined && event.ctrlKey !== combination.ctrl) return false;
  if (combination.alt !== undefined && event.altKey !== combination.alt) return false;
  if (combination.shift !== undefined && event.shiftKey !== combination.shift) return false;
  if (combination.meta !== undefined && event.metaKey !== combination.meta) return false;

  const eventKey = normalizeKey(event.key);
  const comboKey = combination.key ? normalizeKey(combination.key) : '';

  return eventKey === comboKey;
}

/**
 * Global shortcut registry
 */
const globalShortcuts: RegisteredShortcut[] = [];
const viewShortcuts: ViewShortcuts = {};
let currentView: string | null = null;

/**
 * Registers a keyboard shortcut
 */
export function registerShortcut(
  combination: KeyCombination | string,
  handler: ShortcutHandler,
  options: {
    view?: string;
    description?: string;
    preventDefault?: boolean;
    stopPropagation?: boolean;
  } = {}
): RegisteredShortcut {
  const parsedCombination = typeof combination === 'string' ? parseKey(combination) : combination;

  const shortcut: RegisteredShortcut = {
    combination: parsedCombination,
    handler,
    view: options.view,
    description: options.description,
    preventDefault: options.preventDefault ?? true,
    stopPropagation: options.stopPropagation ?? false,
  };

  if (options.view) {
    if (!viewShortcuts[options.view]) {
      viewShortcuts[options.view] = [];
    }
    viewShortcuts[options.view].push(shortcut);
  } else {
    globalShortcuts.push(shortcut);
  }

  return shortcut;
}

/**
 * Unregisters a keyboard shortcut
 */
export function unregisterShortcut(
  combination: KeyCombination | string,
  view?: string
): boolean {
  const parsedCombination = typeof combination === 'string' ? parseKey(combination) : combination;
  const formatCombo = formatShortcut(parsedCombination);

  if (view && viewShortcuts[view]) {
    const index = viewShortcuts[view].findIndex(
      s => formatShortcut(s.combination) === formatCombo
    );
    if (index !== -1) {
      viewShortcuts[view].splice(index, 1);
      return true;
    }
  } else {
    const index = globalShortcuts.findIndex(
      s => formatShortcut(s.combination) === formatCombo
    );
    if (index !== -1) {
      globalShortcuts.splice(index, 1);
      return true;
    }
  }

  return false;
}

/**
 * Gets all shortcuts registered for a specific view
 */
export function getShortcutsForView(view: string): RegisteredShortcut[] {
  return viewShortcuts[view] || [];
}

/**
 * Gets all global shortcuts
 */
export function getGlobalShortcuts(): RegisteredShortcut[] {
  return [...globalShortcuts];
}

/**
 * Sets the current active view for view-specific shortcuts
 */
export function setCurrentView(view: string | null): void {
  currentView = view;
}

/**
 * Gets the current active view
 */
export function getCurrentView(): string | null {
  return currentView;
}

/**
 * Handles global shortcuts - call this in a global keydown listener
 */
export function handleGlobalShortcuts(event: KeyboardEvent): boolean {
  // Check view-specific shortcuts first
  if (currentView && viewShortcuts[currentView]) {
    for (const shortcut of viewShortcuts[currentView]) {
      if (matchesCombination(event, shortcut.combination)) {
        if (shortcut.preventDefault) event.preventDefault();
        if (shortcut.stopPropagation) event.stopPropagation();
        const result = shortcut.handler(event);
        return result !== false;
      }
    }
  }

  // Check global shortcuts
  for (const shortcut of globalShortcuts) {
    if (matchesCombination(event, shortcut.combination)) {
      if (shortcut.preventDefault) event.preventDefault();
      if (shortcut.stopPropagation) event.stopPropagation();
      const result = shortcut.handler(event);
      return result !== false;
    }
  }

  return false;
}

/**
 * Clears all shortcuts (useful for testing or cleanup)
 */
export function clearAllShortcuts(): void {
  globalShortcuts.length = 0;
  Object.keys(viewShortcuts).forEach(key => delete viewShortcuts[key]);
  currentView = null;
}

/**
 * Common shortcut presets
 */
export const CommonShortcuts = {
  // Navigation
  COMMAND_PALETTE: { combination: parseKey('Ctrl+P'), description: 'Open command palette' },
  QUICK_OPEN: { combination: parseKey('Ctrl+O'), description: 'Quick open file' },
  GO_TO_LINE: { combination: parseKey('Ctrl+G'), description: 'Go to line' },
  GO_TO_DEFINITION: { combination: parseKey('F12'), description: 'Go to definition' },
  GO_BACK: { combination: parseKey('Alt+Left'), description: 'Go back' },
  GO_FORWARD: { combination: parseKey('Alt+Right'), description: 'Go forward' },

  // Editing
  UNDO: { combination: parseKey('Ctrl+Z'), description: 'Undo' },
  REDO: { combination: parseKey('Ctrl+Shift+Z'), description: 'Redo' },
  CUT: { combination: parseKey('Ctrl+X'), description: 'Cut' },
  COPY: { combination: parseKey('Ctrl+C'), description: 'Copy' },
  PASTE: { combination: parseKey('Ctrl+V'), description: 'Paste' },
  SELECT_ALL: { combination: parseKey('Ctrl+A'), description: 'Select all' },
  FIND: { combination: parseKey('Ctrl+F'), description: 'Find' },
  FIND_NEXT: { combination: parseKey('F3'), description: 'Find next' },
  FIND_PREVIOUS: { combination: parseKey('Shift+F3'), description: 'Find previous' },
  REPLACE: { combination: parseKey('Ctrl+H'), description: 'Replace' },

  // File operations
  NEW_FILE: { combination: parseKey('Ctrl+N'), description: 'New file' },
  OPEN_FILE: { combination: parseKey('Ctrl+O'), description: 'Open file' },
  SAVE: { combination: parseKey('Ctrl+S'), description: 'Save' },
  SAVE_AS: { combination: parseKey('Ctrl+Shift+S'), description: 'Save as' },
  SAVE_ALL: { combination: parseKey('Ctrl+K S'), description: 'Save all' },
  CLOSE_TAB: { combination: parseKey('Ctrl+W'), description: 'Close tab' },
  REOPEN_CLOSED_TAB: { combination: parseKey('Ctrl+Shift+T'), description: 'Reopen closed tab' },

  // View/Window
  TOGGLE_SIDEBAR: { combination: parseKey('Ctrl+B'), description: 'Toggle sidebar' },
  TOGGLE_TERMINAL: { combination: parseKey('Ctrl+`'), description: 'Toggle terminal' },
  ZOOM_IN: { combination: parseKey('Ctrl+='), description: 'Zoom in' },
  ZOOM_OUT: { combination: parseKey('Ctrl+-'), description: 'Zoom out' },
  ZOOM_RESET: { combination: parseKey('Ctrl+0'), description: 'Reset zoom' },
  FULLSCREEN: { combination: parseKey('F11'), description: 'Toggle fullscreen' },

  // Application
  QUIT: { combination: parseKey('Ctrl+Q'), description: 'Quit application' },
  PREFERENCES: { combination: parseKey('Ctrl+,'), description: 'Open preferences' },
  RELOAD: { combination: parseKey('Ctrl+R'), description: 'Reload window' },

  // Debug/Developer
  TOGGLE_DEVTOOLS: { combination: parseKey('Ctrl+Shift+I'), description: 'Toggle dev tools' },
  TOGGLE_INSPECTOR: { combination: parseKey('Ctrl+Shift+C'), description: 'Toggle inspector' },

  // Generic
  ESCAPE: { combination: parseKey('Escape'), description: 'Escape/Cancel' },
  ENTER: { combination: parseKey('Enter'), description: 'Confirm/Submit' },
  TAB: { combination: parseKey('Tab'), description: 'Next focus' },
  SHIFT_TAB: { combination: parseKey('Shift+Tab'), description: 'Previous focus' },
} as const;

/**
 * Registers common shortcuts with handlers
 */
export function registerCommonShortcuts(handlers: {
  onCommandPalette?: ShortcutHandler;
  onQuickOpen?: ShortcutHandler;
  onGoToLine?: ShortcutHandler;
  onUndo?: ShortcutHandler;
  onRedo?: ShortcutHandler;
  onCopy?: ShortcutHandler;
  onPaste?: ShortcutHandler;
  onFind?: ShortcutHandler;
  onSave?: ShortcutHandler;
  onCloseTab?: ShortcutHandler;
  onQuit?: ShortcutHandler;
  onEscape?: ShortcutHandler;
  onEnter?: ShortcutHandler;
  [key: string]: ShortcutHandler | undefined;
}): RegisteredShortcut[] {
  const shortcuts: RegisteredShortcut[] = [];

  const register = (preset: typeof CommonShortcuts.COMMAND_PALETTE, handler?: ShortcutHandler) => {
    if (handler) {
      shortcuts.push(registerShortcut(preset.combination, handler, { description: preset.description }));
    }
  };

  register(CommonShortcuts.COMMAND_PALETTE, handlers.onCommandPalette);
  register(CommonShortcuts.QUICK_OPEN, handlers.onQuickOpen);
  register(CommonShortcuts.GO_TO_LINE, handlers.onGoToLine);
  register(CommonShortcuts.UNDO, handlers.onUndo);
  register(CommonShortcuts.REDO, handlers.onRedo);
  register(CommonShortcuts.COPY, handlers.onCopy);
  register(CommonShortcuts.PASTE, handlers.onPaste);
  register(CommonShortcuts.FIND, handlers.onFind);
  register(CommonShortcuts.SAVE, handlers.onSave);
  register(CommonShortcuts.CLOSE_TAB, handlers.onCloseTab);
  register(CommonShortcuts.QUIT, handlers.onQuit);
  register(CommonShortcuts.ESCAPE, handlers.onEscape);
  register(CommonShortcuts.ENTER, handlers.onEnter);

  return shortcuts;
}

/**
 * Creates a keyboard event listener for easy attachment
 */
export function createKeyboardListener(
  target: EventTarget = window,
  options: { capture?: boolean; passive?: boolean } = {}
): { destroy: () => void } {
  const handler = (event: KeyboardEvent) => handleGlobalShortcuts(event);

  target.addEventListener('keydown', handler, options);

  return {
    destroy: () => target.removeEventListener('keydown', handler, options),
  };
}

/**
 * Checks if a key is a modifier key
 */
export function isModifierKey(key: string): boolean {
  const normalized = normalizeKey(key).toLowerCase();
  return ['control', 'alt', 'shift', 'meta', 'cmd', 'command', 'win', 'super'].includes(normalized);
}

/**
 * Gets the display name for a key (platform-aware)
 */
export function getKeyDisplayName(key: string, platform: 'mac' | 'windows' | 'linux' = 'windows'): string {
  const normalized = normalizeKey(key);

  if (platform === 'mac') {
    const macMap: Record<string, string> = {
      'Ctrl': '⌃',
      'Alt': '⌥',
      'Shift': '⇧',
      'Meta': '⌘',
      'Enter': '⏎',
      'Escape': '⎋',
      'Tab': '⇥',
      'Backspace': '⌫',
      'Delete': '⌦',
      'ArrowUp': '↑',
      'ArrowDown': '↓',
      'ArrowLeft': '←',
      'ArrowRight': '→',
    };
    return macMap[normalized] || normalized;
  }

  return normalized;
}

/**
 * Formats a shortcut for display (platform-aware)
 */
export function formatShortcutForDisplay(
  combination: KeyCombination,
  platform: 'mac' | 'windows' | 'linux' = 'windows'
): string {
  const parts: string[] = [];

  if (combination.ctrl) parts.push(getKeyDisplayName('Ctrl', platform));
  if (combination.alt) parts.push(getKeyDisplayName('Alt', platform));
  if (combination.shift) parts.push(getKeyDisplayName('Shift', platform));
  if (combination.meta) parts.push(getKeyDisplayName('Meta', platform));
  if (combination.key) parts.push(getKeyDisplayName(combination.key, platform));

  return platform === 'mac' ? parts.join('') : parts.join('+');
}

export default {
  parseKey,
  formatShortcut,
  registerShortcut,
  unregisterShortcut,
  getShortcutsForView,
  getGlobalShortcuts,
  setCurrentView,
  getCurrentView,
  handleGlobalShortcuts,
  clearAllShortcuts,
  CommonShortcuts,
  registerCommonShortcuts,
  createKeyboardListener,
  isModifierKey,
  getKeyDisplayName,
  formatShortcutForDisplay,
};