import React from 'react';
import { Box, Text } from './index.js';

export interface SidebarProps {
  items: Array<{
    id: string;
    label: string;
    icon?: React.ReactNode;
    badge?: string;
    badgeVariant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  }>;
  activeId: string;
  onSelect: (id: string) => void;
  width?: number;
}

export const Sidebar = ({
  items,
  activeId,
  onSelect,
  width = 24,
}: SidebarProps) => {
  return (
    <Box
      variant="surface"
      borderStyle="round"
      borderColor="border"
      padding={1}
      flexDirection="column"
      width={width}
      minWidth={width}
      maxWidth={width}
      height="100%"
    >
      {items.map((item) => {
        const isActive = item.id === activeId;
        return (
          <Box
            key={item.id}
            variant={isActive ? 'selected' : 'default'}
            borderStyle="round"
            borderColor={isActive ? 'primary' : 'default'}
            padding={1}
            marginBottom={1}
            flexDirection="row"
            alignItems="center"
            onClick={() => onSelect(item.id)}
          >
            {item.icon && (
              <Box width={4} flexDirection="row" alignItems="center" justifyContent="center" marginRight={1}>
                {item.icon}
              </Box>
            )}
            <Box flex={1} flexDirection="column">
              <Text weight={isActive ? 'bold' : 'normal'} variant={isActive ? 'bright' : 'default'}>
                {item.label}
              </Text>
              {item.badge && (
                <Text variant="subtle" fontSize="small" marginTop={0}>
                  {item.badge}
                </Text>
              )}
            </Box>
          </Box>
        );
      })}
    </Box>
  );
};
