import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';

export interface ConfirmDialogProps {
  /** Dialog title */
  title: string;
  /** Dialog message/content */
  message: string;
  /** Text for confirm button */
  confirmText?: string;
  /** Text for cancel button */
  cancelText?: string;
  /** Visual variant - 'default' or 'danger' */
  variant?: 'default' | 'danger';
  /** Whether dialog is open */
  isOpen: boolean;
  /** Called when user confirms */
  onConfirm: () => void;
  /** Called when user cancels */
  onCancel: () => void;
  /** Optional additional CSS class */
  className?: string;
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'default',
  isOpen,
  onConfirm,
  onCancel,
  className = '',
}) => {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);
  const focusableElementsRef = useRef<HTMLElement[]>([]);

  // Handle keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      // Close on Escape
      if (event.key === 'Escape') {
        event.preventDefault();
        onCancel();
        return;
      }

      // Confirm on Enter or 'y'
      if (event.key === 'Enter' || event.key === 'y' || event.key === 'Y') {
        // Don't trigger if typing in an input
        const target = event.target as HTMLElement;
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA' && !target.isContentEditable) {
          event.preventDefault();
          onConfirm();
          return;
        }
      }

      // Cancel on 'n'
      if (event.key === 'n' || event.key === 'N') {
        const target = event.target as HTMLElement;
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA' && !target.isContentEditable) {
          event.preventDefault();
          onCancel();
          return;
        }
      }

      // Trap focus with Tab
      if (event.key === 'Tab') {
        const focusableElements = focusableElementsRef.current;
        if (focusableElements.length === 0) return;

        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];
        const activeElement = document.activeElement as HTMLElement;

        if (event.shiftKey) {
          if (activeElement === firstElement) {
            event.preventDefault();
            lastElement.focus();
          }
        } else {
          if (activeElement === lastElement) {
            event.preventDefault();
            firstElement.focus();
          }
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onConfirm, onCancel]);

  // Manage focus and portal
  useEffect(() => {
    if (isOpen) {
      previousActiveElement.current = document.activeElement as HTMLElement;
      // Focus will be set after render via ref callback
    } else if (previousActiveElement.current) {
      previousActiveElement.current.focus();
      previousActiveElement.current = null;
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const dialogContent = (
    <dialog
      ref={(el) => {
        dialogRef.current = el;
        if (el) {
          // Collect focusable elements after render
          const focusable = el.querySelectorAll<HTMLElement>(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
          );
          focusableElementsRef.current = Array.from(focusable);

          // Focus first button (confirm) by default
          const confirmButton = el.querySelector<HTMLButtonElement>('[data-action="confirm"]');
          confirmButton?.focus();

          el.showModal();
        }
      }}
      className={`confirm-dialog ${variant} ${className}`}
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="confirm-dialog-title"
      aria-describedby="confirm-dialog-message"
      onClose={(e) => {
        // Prevent closing via backdrop click - only allow explicit actions
        if (e.target === e.currentTarget) {
          e.preventDefault();
        }
      }}
    >
      <div className="confirm-dialog__backdrop" />

      <div className="confirm-dialog__container">
        <header className="confirm-dialog__header">
          <h2 id="confirm-dialog-title" className="confirm-dialog__title">
            {title}
          </h2>
        </header>

        <div id="confirm-dialog-message" className="confirm-dialog__message">
          {message}
        </div>

        <footer className="confirm-dialog__footer">
          <button
            type="button"
            className="confirm-dialog__btn confirm-dialog__btn--cancel"
            data-action="cancel"
            onClick={onCancel}
          >
            {cancelText}
          </button>
          <button
            type="button"
            className={`confirm-dialog__btn confirm-dialog__btn--confirm ${variant === 'danger' ? 'confirm-dialog__btn--danger' : ''}`}
            data-action="confirm"
            onClick={onConfirm}
          >
            {confirmText}
          </button>
        </footer>
      </div>
    </dialog>
  );

  // Render via portal to body for proper stacking context
  return createPortal(dialogContent, document.body);
};

export default ConfirmDialog;