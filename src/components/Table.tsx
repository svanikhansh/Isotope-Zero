import React, { useMemo, useCallback } from 'react';

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
  className?: string;
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
  className = '',
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
      <div className={`table-container ${className}`} role="status" aria-live="polite">
        <div className="table-loading" aria-label="Loading table data">
          <div className="table-spinner" />
          <span>Loading...</span>
        </div>
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className={`table-container ${className}`} role="table" aria-label="Empty table">
        <div className="table-empty" aria-live="polite">{emptyMessage}</div>
      </div>
    );
  }

  return (
    <div className={`table-container ${className}`} role="table" aria-label="Data table">
      {showHeader && (
        <div
          className={`table-header ${showBorders ? 'with-borders' : ''}`}
          role="rowgroup"
        >
          {columns.map((col, colIndex) => (
            <div
              key={col.key}
              className="table-header-cell"
              role="columnheader"
              style={{
                width: columnStyles[colIndex].width,
                textAlign: columnStyles[colIndex].textAlign,
              } as React.CSSProperties}
              aria-sort="none"
            >
              {col.header}
            </div>
          ))}
        </div>
      )}

      <div
        className={`table-body ${showBorders ? 'with-borders' : ''}`}
        role="rowgroup"
      >
        {rows.map((row, rowIndex) => {
          const isSelected = selectedIndex === rowIndex;
          const customRowClass = rowClassName?.(row, rowIndex) || '';

          return (
            <div
              key={rowIndex}
              className={`table-row ${isSelected ? 'selected' : ''} ${customRowClass}`}
              role="row"
              tabIndex={onActivate ? 0 : -1}
              aria-selected={isSelected}
              onClick={() => handleRowClick(rowIndex, row)}
              onDoubleClick={() => handleRowDoubleClick(rowIndex, row)}
              onKeyDown={(e) => handleKeyDown(e, rowIndex, row)}
            >
              {columns.map((col, colIndex) => {
                const value = row[col.key];
                const rendered = col.render
                  ? col.render(value, row, rowIndex)
                  : value;

                return (
                  <div
                    key={col.key}
                    className="table-cell"
                    role="gridcell"
                    style={{
                      width: columnStyles[colIndex].width,
                      textAlign: columnStyles[colIndex].textAlign,
                    } as React.CSSProperties}
                  >
                    {rendered}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default Table;