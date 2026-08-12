import React from 'react';
import { Box, Text } from './index.js';
import { Dimensions } from '../theme/index.js';

export interface HeaderProps {
  title: string;
  subtitle?: string;
  version?: string;
  connected?: boolean;
  lastUpdate?: string;
}

export const Header = ({
  title,
  subtitle,
  version,
  connected = false,
  lastUpdate,
}: HeaderProps) => {
  return (
    <Box
      variant="surface"
      borderStyle="single"
      borderColor="border"
      padding={2}
      flexDirection="row"
      justifyContent="space-between"
      alignItems="center"
      width="100%"
    >
      <Box flexDirection="row" alignItems="center">
        <Box
          width={6}
          height={3}
          backgroundColor="primary"
          marginRight={2}
          flexDirection="row"
          alignItems="center"
          justifyContent="center"
        >
          <Text variant="bright" weight="bold">◆</Text>
        </Box>
        <Box flexDirection="column">
          <Box flexDirection="row" alignItems="center">
            <Text weight="bold" variant="bright" fontSize="large">{title}</Text>
            {version && <Text variant="muted" weight="normal" marginLeft={1}>v{version}</Text>}
          </Box>
          {subtitle && <Text variant="muted" fontSize="small">{subtitle}</Text>}
        </Box>
      </Box>

      <Box flexDirection="row" gap={2}>
        <Box flexDirection="row" alignItems="center">
          <Box
            width={2}
            height={2}
            backgroundColor={connected ? 'success' : 'error'}
            borderRadius={1}
          />
          <Text variant={connected ? 'success' : 'error'} marginLeft={1}>
            {connected ? 'Connected' : 'Disconnected'}
          </Text>
        </Box>

        {lastUpdate && (
          <Box flexDirection="row">
            <Text variant="subtle" fontSize="small">Updated {lastUpdate}</Text>
          </Box>
        )}
      </Box>
    </Box>
  );
};
