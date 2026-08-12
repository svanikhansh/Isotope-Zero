import chalk from 'chalk';
import { inspect } from 'util';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'success';

export interface LogContext {
  [key: string]: unknown;
}

export interface LoggerOptions {
  level?: LogLevel;
  timestamp?: boolean;
  context?: LogContext;
  theme?: 'light' | 'dark' | 'auto';
}

const LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
  success: 1,
};

const DEFAULT_LEVEL: LogLevel = 'info';

function detectTheme(): 'light' | 'dark' {
  if (typeof process !== 'undefined' && process.env.FORCE_COLOR === '0') {
    return 'light';
  }
  if (typeof process !== 'undefined' && process.env.TERM_PROGRAM === 'vscode') {
    return 'dark';
  }
  return 'dark';
}

function getThemeColors(theme: 'light' | 'dark') {
  if (theme === 'light') {
    return {
      debug: chalk.gray,
      info: chalk.blue,
      warn: chalk.hex('#B8860B'),
      error: chalk.red,
      success: chalk.green,
      timestamp: chalk.gray,
      context: chalk.cyan,
      label: chalk.bold,
      reset: chalk.reset,
    };
  }
  return {
    debug: chalk.gray,
    info: chalk.cyan,
    warn: chalk.yellow,
    error: chalk.redBright,
    success: chalk.greenBright,
    timestamp: chalk.gray,
    context: chalk.magenta,
    label: chalk.bold,
    reset: chalk.reset,
  };
}

function formatTimestamp(): string {
  const now = new Date();
  return now.toISOString().replace('T', ' ').replace('Z', '').split('.')[0];
}

function formatContext(context: LogContext): string {
  if (!context || Object.keys(context).length === 0) {
    return '';
  }
  const entries = Object.entries(context)
    .map(([key, value]) => `${key}=${formatValue(value)}`)
    .join(' ');
  return ` {${entries}}`;
}

function formatValue(value: unknown): string {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (typeof value === 'string') return value.includes(' ') ? `"${value}"` : value;
  if (typeof value === 'object') {
    try {
      return inspect(value, { depth: 3, colors: false, compact: true, maxArrayLength: 10 });
    } catch {
      return '[Object]';
    }
  }
  return String(value);
}

function formatMessage(
  level: LogLevel,
  message: string,
  context?: LogContext,
  theme: 'light' | 'dark' = 'dark',
  showTimestamp: boolean = true
): string {
  const colors = getThemeColors(theme);
  const parts: string[] = [];

  if (showTimestamp) {
    parts.push(colors.timestamp(`[${formatTimestamp()}]`));
  }

  const levelLabel = level.toUpperCase().padEnd(5);
  let levelColor: chalk.Chalk;

  switch (level) {
    case 'debug':
      levelColor = colors.debug;
      break;
    case 'info':
      levelColor = colors.info;
      break;
    case 'warn':
      levelColor = colors.warn;
      break;
    case 'error':
      levelColor = colors.error;
      break;
    case 'success':
      levelColor = colors.success;
      break;
    default:
      levelColor = colors.reset;
  }

  parts.push(levelColor(levelLabel));
  parts.push(colors.label(message));

  if (context) {
    parts.push(colors.context(formatContext(context)));
  }

  return parts.join(' ');
}

export class Logger {
  private level: LogLevel;
  private showTimestamp: boolean;
  private defaultContext: LogContext;
  private theme: 'light' | 'dark';

  constructor(options: LoggerOptions = {}) {
    this.level = options.level ?? DEFAULT_LEVEL;
    this.showTimestamp = options.timestamp ?? true;
    this.defaultContext = options.context ?? {};
    this.theme = options.theme ?? detectTheme();
  }

  setLevel(level: LogLevel): void {
    this.level = level;
  }

  setTheme(theme: 'light' | 'dark' | 'auto'): void {
    this.theme = theme === 'auto' ? detectTheme() : theme;
  }

  setContext(context: LogContext): void {
    this.defaultContext = { ...this.defaultContext, ...context };
  }

  clearContext(): void {
    this.defaultContext = {};
  }

  private shouldLog(level: LogLevel): boolean {
    return LEVEL_PRIORITY[level] >= LEVEL_PRIORITY[this.level];
  }

  private log(level: LogLevel, message: string, context?: LogContext): void {
    if (!this.shouldLog(level)) return;

    const mergedContext = { ...this.defaultContext, ...context };
    const formatted = formatMessage(level, message, mergedContext, this.theme, this.showTimestamp);
    const stream = level === 'error' || level === 'warn' ? process.stderr : process.stdout;
    stream.write(`${formatted}\n`);
  }

  debug(message: string, context?: LogContext): void {
    this.log('debug', message, context);
  }

  info(message: string, context?: LogContext): void {
    this.log('info', message, context);
  }

  warn(message: string, context?: LogContext): void {
    this.log('warn', message, context);
  }

  error(message: string, context?: LogContext): void {
    this.log('error', message, context);
  }

  success(message: string, context?: LogContext): void {
    this.log('success', message, context);
  }

  child(context: LogContext): Logger {
    return new Logger({
      level: this.level,
      timestamp: this.showTimestamp,
      context: { ...this.defaultContext, ...context },
      theme: this.theme,
    });
  }
}

export const logger = new Logger();

export function createLogger(options: LoggerOptions = {}): Logger {
  return new Logger(options);
}