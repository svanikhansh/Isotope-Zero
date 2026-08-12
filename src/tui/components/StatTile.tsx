import React from 'react';
import { Box, Text } from './index.js';

export interface StatTileProps {
  label: string;
  value: string | number;
  trend?: {
    value: string;
    label: string;
    positive: boolean;
  };
  icon?: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  subtitle?: string;
}

export const StatTile = ({
  label,
  value,
  trend,
  icon,
  variant = 'default',
  subtitle,
}: StatTileProps) => {
  return (
    <Box
      variant="surface"
      borderStyle="round"
      borderColor="border"
      padding={2}
      flexDirection="column"
      minWidth={18}
      flex={1}
    >
      <Box flexDirection="row" justifyContent="space-between" marginBottom={1}>
        <Box flexDirection="column">
          <Text variant="muted" fontSize="small" textTransform="uppercase" letterSpacing={1}>
            {label}
          </Text>
          {subtitle && <Text variant="subtle" fontSize="small" marginTop={0}>{subtitle}</Text>}
        </Box>
        {icon && (
          <Box alignItems="center" justifyContent="center" opacity={0.7}>
            {icon}
          </Box>
        )}
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Text weight="bold" variant="bright" fontSize="xlarge" lineHeight={1}>
          {value}
        </Text>
        {trend && (
          <Box flexDirection="row" marginTop={1} alignItems="center">
            <Text
              variant={trend.positive ? 'success' : 'error'}
              weight="bold"
              fontSize="small"
            >
              {trend.positive ? '▲' : '▼'} {trend.value}
            </Text>
            <Text variant="subtle" fontSize="small" marginLeft={1}>
              {trend.label}
            </Text>
          </Box>
        )}
      </Box>
    </Box>
  );
};
