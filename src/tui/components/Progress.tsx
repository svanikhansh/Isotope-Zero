import React, { useState, useEffect } from 'react';
import { Box, Text } from '.';
import { Dimensions } from '../theme';

export interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showPercent?: boolean;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  size?: 'sm' | 'md' | 'lg';
}

export const ProgressBar = ({
  value,
  max = 100,
  label,
  showPercent = true,
  variant = 'default',
  size = 'md',
}: ProgressBarProps) => {
  const percentage = Math.min(Math.max(value / max, 0), 1);
  const filledWidth = Math.round(percentage * 100);
  const heightMap = { sm: 1, md: 2, lg: 3 };
  const height = heightMap[size];

  return (
    <Box flexDirection="column" gap={1}>
      {label && (
        <Box flexDirection="row" justifyContent="space-between">
          <Text variant="muted" fontSize="small">{label}</Text>
          {showPercent && (
            <Text variant="subtle" weight="bold" fontSize="small">
              {Math.round(percentage * 100)}%
            </Text>
          )}
        </Box>
      )}
      <Box
        variant="subtle"
        borderStyle="round"
        width="100%"
        height={height}
        overflow="hidden"
      >
        <Box
          width={`${filledWidth}%`}
          height="100%"
          backgroundColor={variant}
        />
      </Box>
    </Box>
  );
};

export interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info';
  label?: string;
}

export const Spinner = ({
  variant = 'primary',
  label,
}: SpinnerProps) => {
  const frames = Dimensions.spinnerFrames;
  const [frameIndex, setFrameIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setFrameIndex((prev) => (prev + 1) % frames.length);
    }, Dimensions.spinnerInterval);
    return () => clearInterval(interval);
  }, []);

  return (
    <Box flexDirection="row" alignItems="center" gap={1}>
      <Text color={variant} weight="bold">
        {frames[frameIndex]}
      </Text>
      {label && <Text variant="muted">{label}</Text>}
    </Box>
  );
};
