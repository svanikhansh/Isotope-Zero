import React from 'react';
import styles from './Badge.module.css';

export type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'info';
export type BadgeSize = 'sm' | 'md' | 'lg';

export interface BadgeProps {
  /** Badge content */
  children: React.ReactNode;
  /** Visual variant */
  variant?: BadgeVariant;
  /** Size variant */
  size?: BadgeSize;
  /** Whether badge is removable */
  removable?: boolean;
  /** Remove handler */
  onRemove?: () => void;
  /** Additional CSS classes */
  className?: string;
  /** Accessibility label for remove button */
  removeLabel?: string;
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  (
    {
      children,
      variant = 'default',
      size = 'md',
      removable = false,
      onRemove,
      className = '',
      removeLabel = 'Remove',
    },
    ref
  ) => {
    const handleRemove = (event: React.MouseEvent) => {
      event.stopPropagation();
      onRemove?.();
    };

    const classNames = [
      styles.badge,
      styles[`variant-${variant}`],
      styles[`size-${size}`],
      removable && styles.removable,
      className,
    ]
      .filter(Boolean)
      .join(' ');

    return (
      <span ref={ref} className={classNames} role="status">
        <span className={styles.content}>{children}</span>
        {removable && (
          <button
            type="button"
            className={styles.removeButton}
            onClick={handleRemove}
            aria-label={removeLabel}
          >
            <svg
              className={styles.removeIcon}
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              aria-hidden="true"
            >
              <line x1="4" y1="4" x2="12" y2="12" />
              <line x1="12" y1="4" x2="4" y2="12" />
            </svg>
          </button>
        )}
      </span>
    );
  }
);

Badge.displayName = 'Badge';