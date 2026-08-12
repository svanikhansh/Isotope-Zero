import React from 'react';
import styles from './StatTile.module.css';
import { LucideIcon } from 'lucide-react';

export interface StatTileTrend {
  /** Trend value (e.g., "+12%", "-5%") */
  value: string;
  /** Trend label (e.g., "vs last month") */
  label: string;
  /** Whether trend is positive */
  positive: boolean;
}

export type StatTileVariant = 'default' | 'success' | 'warning' | 'error' | 'info';

export interface StatTileProps {
  /** Label text */
  label: string;
  /** Main value */
  value: string | number;
  /** Trend information */
  trend?: StatTileTrend;
  /** Optional icon */
  icon?: LucideIcon;
  /** Visual variant */
  variant?: StatTileVariant;
  /** Additional CSS classes */
  className?: string;
  /** Optional subtitle */
  subtitle?: string;
}

export function StatTile({
  label,
  value,
  trend,
  icon: Icon,
  variant = 'default',
  className = '',
  subtitle,
}: StatTileProps) {
  const classNames = [
    styles.tile,
    styles[`variant-${variant}`],
    className,
  ]
    .filter(Boolean)
    .join(' ');

  const trendClassNames = [
    styles.trend,
    trend?.positive ? styles.trendPositive : styles.trendNegative,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={classNames} role="region" aria-label={label}>
      <div className={styles.header}>
        <div className={styles.labelGroup}>
          <p className={styles.label}>{label}</p>
          {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
        </div>
        {Icon && (
          <div className={cn(styles.iconWrapper, styles[`icon-${variant}`])} aria-hidden="true">
            <Icon className={styles.icon} />
          </div>
        )}
      </div>
      <div className={styles.valueWrapper}>
        <p className={cn(styles.value, styles[`value-${variant}`])}>
          {value}
        </p>
        {trend && (
          <div className={trendClassNames} role="status" aria-live="polite">
            <span className={styles.trendValue}>{trend.value}</span>
            <span className={styles.trendLabel}>{trend.label}</span>
            <svg
              className={cn(styles.trendIcon, trend.positive ? styles.trendIconUp : styles.trendIconDown)}
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              aria-hidden="true"
            >
              {trend.positive ? (
                <path d="M8 4L4 8M8 4L12 8M4 8H12" />
              ) : (
                <path d="M8 12L4 8M8 12L12 8M4 8H12" />
              )}
            </svg>
          </div>
        )}
      </div>
    </div>
  );
}

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}