import React, { useMemo, useCallback } from 'react';
import { Box, Text } from './index.js';

export interface Column<T = any> {
  key: string;
  header: string;
  width?: number | string;
  align?: 'left' | 'center' | 'right';
  render?: (value: any, row: T, rowIndex: number) => React.ReactNode;
}

export interface TableProps<T = any> {
  columns: Column<T>[];
  rows: T[];
  selectedIndex?: number | null;
  onSelect?: (index: number, row: T) => void;
  onActivate?: (index: number, row: T) => void;
  showHeader?: boolean;
  showBorders?: boolean;
  emptyMessage?: string;
  isLoading?: boolean;
  rowClassName?: (row: T, index: number) => string;
}

export function Table<T = any>({
  columns,
  rows,
  selectedIndex = null,
  onSelect,
  onActivate,
  showHeader = true,
  showBorders = true,
  emptyMessage = 'No data available',
  isLoading = false,
  rowClassName,
}: TableProps<T>) {
  const handleRowClick = useCallback(
    (index: number, row: T) => {
      onSelect?.(index, row);
    },
    [onSelect]
  );

  const handleRowDoubleClick = useCallback(
    (index: number, row: T) => {
      onActivate?.(index, row);
    },
    [onActivate]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, index: number, row: T) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onActivate?.(index, row);
      }
    },
    [onActivate]
  );

  const columnStyles = useMemo(() => {
    return columns.map((col) => ({
      width: col.width,
      textAlign: col.align || 'left',
    }));
  }, [columns]);

  if (isLoading) {
    return (
      <Box flexDirection="column" alignItems="center" justifyContent="center" padding={4} role="status" aria-live="polite">
        <Box flexDirection="row">
          <Text variant="muted">Loading...</Text>
        </Box>
      </Box>
    );
  }

  if (rows.length === 0) {
    return (
      <Box flexDirection="column" alignItems="center" justifyContent="center" padding={4} role="table" aria-label="Empty table">
        <Text variant="muted">{emptyMessage}</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" role="table" aria-label="Data table">
      {showHeader && (
        <Box
          flexDirection="row"
          borderBottomWidth={showBorders ? 1 : 0}
          borderBottomStyle="single"
          borderBottomColor="border"
          paddingBottom={1}
          marginBottom={1}
          role="rowgroup"
        >
          {columns.map((col, colIndex) => (
            <Box
              key={col.key}
              width={columnStyles[colIndex].width}
              textAlign={columnStyles[colIndex].textAlign}
              paddingHorizontal={1}
              flex={typeof col.width === 'number' ? 0 : 1}
              role="columnheader"
              aria-sort="none"
            >
              <Text weight="bold" variant="muted" fontSize="small" textTransform="uppercase">
                {col.header}
              </Text>
            </Box>
          ))}
        </Box>
      )}

      <Box role="rowgroup">
        {rows.map((row, rowIndex) => {
          const isSelected = selectedIndex === rowIndex;
          const customRowClass = rowClassName?.(row, rowIndex) || '';

          return (
            <Box
              key={rowIndex}
              variant={isSelected ? 'selected' : 'default'}
              flexDirection="row"
              paddingVertical={1}
              borderBottomWidth={showBorders && rowIndex < rows.length - 1 ? 1 : 0}
              borderBottomStyle="single"
              borderBottomColor="border"
              role="row"
              tabIndex={onActivate ? 0 : -1}
              aria-selected={isSelected}
              onClick={() => handleRowClick(rowIndex, row)}
              onDoubleClick={() => handleRowDoubleClick(rowIndex, row)}
              onKeyDown={(e) => handleKeyDown(e, rowIndex, row)}
            >
              {columns.map((col, colIndex) => {
                const value = (row as Record<string, unknown>)[col.key];
                const rendered = col.render
                  ? col.render(value, row, rowIndex)
                  : String(value ?? '');

                return (
                  <Box
                    key={col.key}
                    width={columnStyles[colIndex].width}
                    textAlign={columnStyles[colIndex].textAlign}
                    paddingHorizontal={1}
                    flex={typeof col.width === 'number' ? 0 : 1}
                    overflow="hidden"
                    textOverflow="ellipsis"
                    whiteSpace="nowrap"
                    role="gridcell"
                  >
                    <Text>{rendered}</Text>
                  </Box>
                );
              })}
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}

export default Table;
