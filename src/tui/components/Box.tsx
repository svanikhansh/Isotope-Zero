import React from 'react';
import { Box as InkBox, BoxProps as InkBoxProps } from 'ink';
import { getTheme, type ThemeName } from '../theme/index.js';

// The app's surface API uses a handful of styling/a11y/event props that Ink
// does not expose on <Box> (there is no `flex`, `maxWidth`, `backgroundColor`,
// `opacity`, absolute offsets, or DOM event/aria props in Ink's layout type).
// We accept them on the wrapper so callers typecheck, translate the ones Ink
// can honor (`flex`→flexGrow, paddingHorizontal/Vertical→paddingX/Y), and
// destructure the rest out so they never leak onto the underlying Ink element —
// forwarding unknown props to Ink is a TS error and a runtime warn.
export interface BoxProps
  extends Omit<InkBoxProps, 'borderStyle' | 'borderColor'> {
  variant?: 'default' | 'surface' | 'bordered' | 'selected' | 'subtle';
  theme?: ThemeName;
  // 'none' is our sentinel for "no border" — Ink's type has no such key.
  borderStyle?: 'none' | InkBoxProps['borderStyle'];
  borderColor?:
    | InkBoxProps['borderColor']
    | 'default'
    | 'primary'
    | 'muted'
    | 'success'
    | 'warning'
    | 'error'
    | 'info';
  children?: React.ReactNode;

  // Props the app uses that Ink does not support natively:
  flex?: number; // → flexGrow
  maxWidth?: number | string;
  paddingHorizontal?: number; // → paddingLeft/Right
  paddingVertical?: number; // → paddingTop/Bottom
  backgroundColor?: string;
  opacity?: number;
  zIndex?: number;
  borderRadius?: number;
  top?: number;
  left?: number;
  right?: number;
  bottom?: number;
  borderBottomWidth?: number;
  borderBottomStyle?: string;
  textOverflow?: string;
  whiteSpace?: string;
  textAlign?: string;
  role?: string;
  tabIndex?: number;
  ref?: React.Ref<unknown>;
  onClick?: (...args: any[]) => void;
  onKeyDown?: (...args: any[]) => void;
  onDoubleClick?: (...args: any[]) => void;
}

export const Box = ({
  variant = 'default',
  theme: themeName = 'ocean-blue',
  borderStyle = 'none',
  borderColor = 'default',
  flex,
  maxWidth,
  paddingHorizontal,
  paddingVertical,
  backgroundColor,
  opacity,
  zIndex,
  borderRadius,
  top,
  left,
  right,
  bottom,
  borderBottomWidth,
  borderBottomStyle,
  textOverflow,
  whiteSpace,
  textAlign,
  role,
  tabIndex,
  ref,
  onClick,
  onKeyDown,
  onDoubleClick,
  children,
  ...props
}: BoxProps) => {
  const theme = getTheme(themeName);

  const resolveBorderColor = () => {
    if (borderColor.startsWith('#')) return borderColor;
    const colorMap: Record<string, string | undefined> = {
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

  // Destructure-but-drop the props Ink cannot render so they never leak.
  void maxWidth;
  void backgroundColor;
  void opacity;
  void zIndex;
  void borderRadius;
  void top;
  void left;
  void right;
  void bottom;
  void borderBottomWidth;
  void borderBottomStyle;
  void textOverflow;
  void whiteSpace;
  void textAlign;
  void role;
  void tabIndex;
  void ref;
  void onClick;
  void onKeyDown;
  void onDoubleClick;

  const finalBorderStyle = borderStyle === 'none' ? undefined : borderStyle;
  const finalBorderColor = finalBorderStyle === undefined ? undefined : resolveBorderColor();

  return (
    <InkBox
      ref={ref as React.Ref<React.ElementRef<typeof InkBox>>}
      borderColor={finalBorderColor}
      borderStyle={finalBorderStyle}
      flexGrow={flex}
      paddingLeft={paddingHorizontal}
      paddingRight={paddingHorizontal}
      paddingTop={paddingVertical}
      paddingBottom={paddingVertical}
      {...props}
    >
      {children}
    </InkBox>
  );
};