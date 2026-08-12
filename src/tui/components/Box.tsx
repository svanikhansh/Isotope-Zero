import React from 'react';
import { Box as InkBox, Text as InkText, BoxProps as InkBoxProps } from 'ink';
import { getTheme, type ThemeName } from '../theme';

export interface BoxProps extends InkBoxProps {
  variant?: 'default' | 'surface' | 'bordered' | 'selected' | 'subtle';
  borderStyle?: 'none' | 'single' | 'double' | 'round' | 'thick';
  borderColor?: 'default' | 'primary' | 'muted' | 'success' | 'warning' | 'error' | 'info' | string;
  theme?: ThemeName;
}

export const Box = ({
  variant = 'default',
  borderStyle = 'none',
  borderColor = 'default',
  theme: themeName = 'ocean-blue',
  children,
  ...props
}: BoxProps) => {
  const theme = getTheme(themeName);

  const resolveBorderColor = () => {
    if (borderColor.startsWith('#')) return borderColor;
    const colorMap: Record<string, string> = {
      default: theme.border,
      primary: theme.primary,
      muted: theme.muted,
      success: theme.success,
      warning: theme.warning,
      error: theme.error,
      info: theme.info,
    };
    return colorMap[borderColor] || theme.border;
  };

  const resolveBackgroundColor = () => {
    const bgMap: Record<string, string> = {
      default: theme.background,
      surface: theme.surface,
      bordered: theme.surface,
      selected: theme.selection,
      subtle: theme.subtle,
    };
    return bgMap[variant] || theme.background;
  };

  const finalBorderStyle = borderStyle === 'none' ? undefined : borderStyle;
  const finalBorderColor = borderStyle === 'none' ? undefined : resolveBorderColor();

  return (
    <InkBox
      backgroundColor={resolveBackgroundColor()}
      borderColor={finalBorderColor}
      borderStyle={finalBorderStyle}
      {...props}
    >
      {children}
    </InkBox>
  );
};
