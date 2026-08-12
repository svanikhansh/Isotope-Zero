import React, { useEffect, useRef, useState } from 'react';
import { Box, Text, Badge } from '.';

export interface ModalProps {
  title: string;
  message: string;
  isOpen: boolean;
  confirmText?: string;
  cancelText?: string;
  variant?: 'default' | 'danger';
  onConfirm: () => void;
  onCancel: () => void;
}

export const Modal = ({
  title,
  message,
  isOpen,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'default',
  onConfirm,
  onCancel,
}: ModalProps) => {
  if (!isOpen) return null;

  const confirmVariant = variant === 'danger' ? 'error' : 'primary';

  return (
    <Box
      position="absolute"
      top={0}
      left={0}
      right={0}
      bottom={0}
      flexDirection="row"
      alignItems="center"
      justifyContent="center"
      backgroundColor="subtle"
      zIndex={1000}
    >
      <Box
        variant="surface"
        borderStyle="round"
        borderColor={variant === 'danger' ? 'error' : 'primary'}
        padding={3}
        flexDirection="column"
        width={60}
        maxWidth="90%"
        minWidth={40}
        gap={2}
        role="alertdialog"
        aria-modal="true"
      >
        <Box alignItems="center" justifyContent="center">
          <Text variant={variant === 'danger' ? 'error' : 'warning'} weight="bold" fontSize="xlarge">
            {variant === 'danger' ? '⚠' : '?'}
          </Text>
        </Box>

        <Box flexDirection="column" alignItems="center" justifyContent="center">
          <Text weight="bold" variant="bright" textAlign="center" marginBottom={1}>
            {title}
          </Text>
          <Text variant="muted" textAlign="center" lineHeight={1.5}>
            {message}
          </Text>
        </Box>

        <Box flexDirection="row" justifyContent="space-between" marginTop={1} gap={2}>
          <Box
            variant="subtle"
            borderStyle="round"
            borderColor="muted"
            padding={1}
            flexDirection="row"
            alignItems="center"
            justifyContent="center"
            flex={1}
            onClick={onCancel}
            tabIndex={0}
            role="button"
          >
            <Text variant="muted" weight="bold">{cancelText}</Text>
          </Box>
          <Box
            variant="selected"
            borderStyle="round"
            borderColor={confirmVariant}
            padding={1}
            flexDirection="row"
            alignItems="center"
            justifyContent="center"
            flex={1}
            onClick={onConfirm}
            tabIndex={0}
            role="button"
          >
            <Text variant="bright" weight="bold">{confirmText}</Text>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export interface AlertProps {
  message: string;
  variant?: 'info' | 'success' | 'warning' | 'error';
  title?: string;
  dismissible?: boolean;
  onDismiss?: () => void;
}

export const Alert = ({
  message,
  variant = 'info',
  title,
  dismissible = false,
  onDismiss,
}: AlertProps) => {
  const variantConfig = {
    info: { icon: 'ℹ', color: 'info', bg: 'info' },
    success: { icon: '✓', color: 'success', bg: 'success' },
    warning: { icon: '⚠', color: 'warning', bg: 'warning' },
    error: { icon: '✕', color: 'error', bg: 'error' },
  };

  const config = variantConfig[variant];

  return (
    <Box
      variant="surface"
      borderStyle="round"
      borderColor={config.color}
      padding={2}
      marginBottom={1}
      flexDirection="row"
      gap={2}
    >
      <Box width={4} alignItems="center" justifyContent="center">
        <Text variant="bright" weight="bold" color={config.color}>
          {config.icon}
        </Text>
      </Box>
      <Box flex={1} flexDirection="column">
        {title && <Text weight="bold" variant="bright" marginBottom={0}>{title}</Text>}
        <Text variant="muted">{message}</Text>
      </Box>
      {dismissible && (
        <Box alignItems="center" justifyContent="center" opacity={0.6} onClick={onDismiss} tabIndex={0} role="button">
          <Text variant="subtle" weight="bold">✕</Text>
        </Box>
      )}
    </Box>
  );
};
