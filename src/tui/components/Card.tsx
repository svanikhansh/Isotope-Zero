import React, { forwardRef } from 'react';
import { Box, Text } from '.';

export interface CardProps {
  title?: string;
  subtitle?: string;
  children?: React.ReactNode;
  footer?: React.ReactNode;
  selected?: boolean;
  interactive?: boolean;
  onActivate?: () => void;
  badge?: string;
  badgeVariant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  padding?: 'none' | 'xs' | 'sm' | 'md' | 'lg';
  bordered?: boolean;
  flex?: number;
  width?: number | string;
  height?: number | string;
  flexDirection?: 'row' | 'column';
  justifyContent?: 'flex-start' | 'center' | 'flex-end' | 'space-between';
  alignItems?: 'flex-start' | 'center' | 'flex-end' | 'stretch';
  gap?: number;
  margin?: number;
  paddingX?: number;
  paddingY?: number;
}

const paddingMap = { none: 0, xs: 1, sm: 2, md: 3, lg: 4 };

export const Card = forwardRef<any, CardProps>(
  ({
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
    bordered = true,
    flex,
    width,
    height,
    flexDirection = 'column',
    justifyContent,
    alignItems = 'stretch',
    gap,
    margin = 1,
    paddingX,
    paddingY,
    ...props
  },
  ref
) => {
  const pad = paddingMap[padding];

  return (
    <Box
      ref={ref}
      variant={selected ? 'selected' : 'surface'}
      borderStyle={bordered ? 'round' : 'none'}
      borderColor={selected ? 'primary' : 'default'}
      padding={pad}
      margin={margin}
      flexDirection={flexDirection}
      alignItems={alignItems}
      flex={flex}
      width={width}
      height={height}
      justifyContent={justifyContent}
      gap={gap}
      paddingX={paddingX}
      paddingY={paddingY}
      {...props}
    >
      {(title || subtitle || badge) && (
        <Box flexDirection="row" justifyContent="space-between" marginBottom={1}>
          <Box flexDirection="column">
            {title && <Text weight="bold" variant="bright">{title}</Text>}
            {subtitle && <Text variant="muted" fontSize="small">{subtitle}</Text>}
          </Box>
          {badge && <Badge variant={badgeVariant}>{badge}</Badge>}
        </Box>
      )}

      <Box flex={1}>{children}</Box>

      {footer && (
        <Box flexDirection="row" marginTop={1} borderTop="single" borderColor="muted" paddingTop={1}>
          {footer}
        </Box>
      )}
    </Box>
  );
});

Card.displayName = 'Card';

export interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  size?: 'sm' | 'md' | 'lg';
}

export const Badge = ({ children, variant = 'default', size = 'md' }: BadgeProps) => {
  const pad = { sm: 0, md: 1, lg: 2 }[size];
  return (
    <Box
      variant="subtle"
      borderStyle="round"
      borderColor={variant}
      padding={pad}
      minWidth={size === 'sm' ? 4 : 6}
    >
      <Text variant={variant} weight="bold">{children}</Text>
    </Box>
  );
};
