import React from 'react';
import styles from './Card.module.css';

export interface CardProps {
  /** Main heading text */
  title?: string;
  /** Secondary text below title */
  subtitle?: string;
  /** Card content */
  children?: React.ReactNode;
  /** Footer content */
  footer?: React.ReactNode;
  /** Whether card is in selected state */
  selected?: boolean;
  /** Whether card is interactive (clickable) */
  interactive?: boolean;
  /** Click handler when interactive */
  onActivate?: () => void;
  /** Badge text */
  badge?: string;
  /** Badge visual variant */
  badgeVariant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  /** Padding size */
  padding?: 'none' | 'sm' | 'md' | 'lg';
  /** Additional CSS classes */
  className?: string;
  /** HTML element to render as */
  as?: 'div' | 'article' | 'section';
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  (
    {
      title,
      subtitle,
      children,
      footer,
      selected = false,
      interactive = false,
      onActivate,
      badge,
      badgeVariant = 'default',
      padding = 'md',
      className = '',
      as: Component = 'div',
    },
    ref
  ) => {
    const handleKeyDown = (event: React.KeyboardEvent) => {
      if (interactive && onActivate && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        onActivate();
      }
    };

    const handleClick = () => {
      if (interactive && onActivate) {
        onActivate();
      }
    };

    const classNames = [
      styles.card,
      styles[`padding-${padding}`],
      selected && styles.selected,
      interactive && styles.interactive,
      className,
    ]
      .filter(Boolean)
      .join(' ');

    const badgeClassNames = [
      styles.badge,
      styles[`badge-${badgeVariant}`],
    ].join(' ');

    return (
      <Component
        ref={ref}
        className={classNames}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        tabIndex={interactive ? 0 : undefined}
        role={interactive ? 'button' : undefined}
        aria-pressed={selected ? true : undefined}
      >
        {(title || subtitle || badge) && (
          <header className={styles.header}>
            <div className={styles.titleGroup}>
              {title && <h3 className={styles.title}>{title}</h3>}
              {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
            </div>
            {badge && <span className={badgeClassNames}>{badge}</span>}
          </header>
        )}

        <div className={styles.content}>{children}</div>

        {footer && <footer className={styles.footer}>{footer}</footer>}
      </Component>
    );
  }
);

Card.displayName = 'Card';