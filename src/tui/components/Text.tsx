import React from 'react';
import { Text as InkText, TextProps as InkTextProps } from 'ink';
import { getTheme, type ThemeName } from '../theme/index.js';

export interface TextProps extends InkTextProps {
  variant?: 'default' | 'muted' | 'subtle' | 'primary' | 'success' | 'warning' | 'error' | 'info' | 'bright' | 'dim';
  weight?: 'normal' | 'bold' | 'light';
  theme?: ThemeName;
  // Non-Ink styling props — accepted for the app's surface API but destructured
  // out so they never leak onto <InkText> (terminals have a fixed font size, so
  // these are cosmetic no-ops here; Ink only supports color + chalk effects):
  fontSize?: string;
  letterSpacing?: number;
  textTransform?: string;
  lineHeight?: number;
  textAlign?: string;
  marginTop?: number;
  marginRight?: number;
  marginBottom?: number;
  marginLeft?: number;
}

export const Text = ({
  variant = 'default',
  weight = 'normal',
  theme: themeName = 'ocean-blue',
  fontSize,
  letterSpacing,
  textTransform,
  lineHeight,
  textAlign,
  marginTop,
  marginRight,
  marginBottom,
  marginLeft,
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

  // Destructure-but-drop the non-Ink props so they can never reach <InkText>.
  void fontSize;
  void letterSpacing;
  void textTransform;
  void lineHeight;
  void textAlign;
  void marginTop;
  void marginRight;
  void marginBottom;
  void marginLeft;

  return (
    <InkText color={getColor()} bold={weight === 'bold'} {...props}>
      {children}
    </InkText>
  );
};