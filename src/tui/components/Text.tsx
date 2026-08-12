import React from 'react';
import { Text as InkText, TextProps as InkTextProps } from 'ink';
import { getTheme, type ThemeName } from '../theme';

export interface TextProps extends InkTextProps {
  variant?: 'default' | 'muted' | 'subtle' | 'primary' | 'success' | 'warning' | 'error' | 'info' | 'bright' | 'dim';
  weight?: 'normal' | 'bold' | 'light';
  theme?: ThemeName;
}

export const Text = ({
  variant = 'default',
  weight = 'normal',
  theme: themeName = 'ocean-blue',
  children,
  ...props
}: TextProps) => {
  const theme = getTheme(themeName);

  const getColor = () => {
    const colorMap: Record<string, string> = {
      default: theme.foreground,
      muted: theme.muted,
      subtle: theme.subtle,
      primary: theme.primary,
      success: theme.success,
      warning: theme.warning,
      error: theme.error,
      info: theme.info,
      bright: theme.foreground,
      dim: theme.subtle,
    };
    return colorMap[variant] || theme.foreground;
  };

  return (
    <InkText
      color={getColor()}
      bold={weight === 'bold'}
      {...props}
    >
      {children}
    </InkText>
  );
};
