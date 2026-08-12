import React from 'react';
import { Box, Text } from '.';

export interface FooterProps {
  left?: React.ReactNode;
  center?: React.ReactNode;
  right?: React.ReactNode;
  hints?: Array<{ key: string; action: string }>;
}

export const Footer = ({
  left,
  center,
  right,
  hints = [],
}: FooterProps) => {
  return (
    <Box
      variant="surface"
      borderStyle="single"
      borderColor="border"
      padding={1}
      flexDirection="row"
      justifyContent="space-between"
      alignItems="center"
      width="100%"
    >
      {left && <Box flexDirection="row">{left}</Box>}

      {center && <Box flex={1} alignItems="center" justifyContent="center">{center}</Box>}

      <Box flexDirection="row" justifyContent="flex-end" gap={2}>
        {hints.map((hint, i) => (
          <Box key={i} flexDirection="row" marginLeft={2}>
            <Text variant="primary" weight="bold">{hint.key}</Text>
            <Text variant="muted" marginLeft={1}>{hint.action}</Text>
          </Box>
        ))}
        {right && <Box flexDirection="row">{right}</Box>}
      </Box>
    </Box>
  );
};
