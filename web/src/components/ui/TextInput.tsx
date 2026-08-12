"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Eye, EyeOff, Search, X, AlertCircle, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export interface TextInputProps
  extends Omit<
    React.InputHTMLAttributes<HTMLInputElement>,
    "onAnimationStart" | "onAnimationEnd" | "onDrag" | "onDragStart" | "onDragEnd"
  > {
  label?: string;
  help?: string;
  error?: string;
  type?: "text" | "password" | "search";
  maxLength?: number;
  showCharacterCount?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  clearable?: boolean;
  fullWidth?: boolean;
  autoFocus?: boolean;
  showValidationIcon?: boolean;
}

export function TextInput({
  label,
  help,
  error,
  type = "text",
  maxLength,
  showCharacterCount = false,
  leftIcon,
  rightIcon,
  clearable = false,
  fullWidth = false,
  autoFocus = false,
  showValidationIcon = false,
  value,
  onChange,
  onSubmit,
  onFocus,
  onBlur,
  onKeyDown,
  disabled,
  placeholder,
  className,
  id,
  "aria-describedby": ariaDescribedBy,
  ...props
}: TextInputProps) {
  const inputId = id || `text-input-${React.useId()}`;
  const [isFocused, setIsFocused] = React.useState(false);
  const [showPassword, setShowPassword] = React.useState(false);
  const [localValue, setLocalValue] = React.useState(value || "");
  const inputRef = React.useRef<HTMLInputElement>(null);
  const helpId = `${inputId}-help`;
  const errorId = `${inputId}-error`;

  const controlled = value !== undefined;
  const currentValue = controlled ? value : localValue;
  const effectiveType = type === "password" && showPassword ? "text" : type;
  const hasError = Boolean(error);
  const isEmpty = currentValue === "";

  React.useEffect(() => {
    if (controlled) {
      setLocalValue(value || "");
    }
  }, [value, controlled]);

  React.useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus();
    }
  }, [autoFocus]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    if (!controlled) setLocalValue(newValue);
    onChange?.(e);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      onSubmit?.(currentValue);
    }
    onKeyDown?.(e);
  };

  const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
    setIsFocused(true);
    onFocus?.(e);
  };

  const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    setIsFocused(false);
    onBlur?.(e);
  };

  const handleClear = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (controlled) onChange?.("" as any); else setLocalValue("");
    inputRef.current?.focus();
  };

  const handlePasswordToggle = () => {
    setShowPassword((prev) => !prev);
    inputRef.current?.focus();
  };

  const characterCount = currentValue.length;
  const isAtMaxLength = maxLength && characterCount >= maxLength;

  const describedBy = [
    help && helpId,
    error && errorId,
    ariaDescribedBy,
  ].filter(Boolean).join(" ") || undefined;

  return (
    <div
      className={cn(
        "relative w-full",
        fullWidth && "w-full",
        className
      )}
    >
      {label && (
        <label
          htmlFor={inputId}
          className={cn(
            "block text-sm font-medium text-[var(--foreground)] mb-1.5",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        >
          {label}
        </label>
      )}

      <div
        className={cn(
          "relative flex items-center gap-2 rounded-[12px] transition-all duration-200 ease-out",
          "bg-[var(--glass-bg)] backdrop-blur-xl border",
          isFocused
            ? "border-[var(--sage)] shadow-[0_0_0_1px_var(--sage),0_4px_16px_rgba(144,202,249,0.1)]"
            : hasError
            ? "border-[var(--error)]/50 hover:border-[var(--error)]"
            : "border-[var(--glass-border)] hover:border-[var(--border-active)]",
          disabled && "opacity-50 cursor-not-allowed bg-[var(--muted)]/50",
          "px-4 py-3"
        )}
      >
        <AnimatePresence mode="wait">
          {leftIcon && (
            <motion.div
              initial={{ opacity: 0, x: -4, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: -4, scale: 0.9 }}
              className={cn(
                "flex items-center justify-center text-[var(--moss)] flex-shrink-0",
                isFocused && "text-[var(--sage)]"
              )}
              aria-hidden="true"
            >
              {leftIcon}
            </motion.div>
          )}
        </AnimatePresence>

        <input
          ref={inputRef}
          id={inputId}
          type={effectiveType}
          value={currentValue}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          disabled={disabled}
          maxLength={maxLength}
          autoComplete={type === "password" ? "current-password" : "off"}
          spellCheck={false}
          className={cn(
            "flex-1 bg-transparent border-none outline-none text-[var(--foreground)]",
            "placeholder:text-[var(--moss)] font-mono text-sm",
            "w-full min-w-0",
            disabled && "cursor-not-allowed",
            hasError && "pr-10",
            showValidationIcon && !hasError && !isEmpty && "pr-10"
          )}
          aria-invalid={hasError}
          aria-describedby={describedBy}
          aria-label={label}
          {...props}
        />

        <AnimatePresence mode="wait">
          {type === "password" && (
            <motion.button
              type="button"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              onClick={handlePasswordToggle}
              className={cn(
                "flex items-center justify-center p-1 rounded-lg",
                "text-[var(--moss)] hover:text-[var(--foreground)]",
                "hover:bg-[var(--muted)] transition-colors",
                "flex-shrink-0",
                disabled && "cursor-not-allowed opacity-50"
              )}
              disabled={disabled}
              aria-label={showPassword ? "Hide password" : "Show password"}
              aria-pressed={showPassword}
            >
              {showPassword ? (
                <EyeOff className="h-4 w-4" aria-hidden="true" />
              ) : (
                <Eye className="h-4 w-4" aria-hidden="true" />
              )}
            </motion.button>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {clearable && !isEmpty && !disabled && (
            <motion.button
              type="button"
              initial={{ opacity: 0, scale: 0, rotate: -90 }}
              animate={{ opacity: 1, scale: 1, rotate: 0 }}
              exit={{ opacity: 0, scale: 0, rotate: 90 }}
              onClick={handleClear}
              className={cn(
                "flex items-center justify-center p-1 rounded-lg",
                "text-[var(--moss)] hover:text-[var(--foreground)]",
                "hover:bg-[var(--muted)] transition-colors",
                "flex-shrink-0"
              )}
              aria-label="Clear input"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </motion.button>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {(hasError || (showValidationIcon && !isEmpty && !hasError)) && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className={cn(
                "flex items-center justify-center p-1 rounded-lg flex-shrink-0",
                hasError
                  ? "text-[var(--error)]"
                  : "text-[var(--success)]"
              )}
              aria-hidden="true"
            >
              {hasError ? (
                <AlertCircle className="h-4 w-4" />
              ) : (
                <CheckCircle className="h-4 w-4" />
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {type === "search" && !leftIcon && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8, rotate: -12 }}
              animate={{ opacity: 1, scale: 1, rotate: 0 }}
              exit={{ opacity: 0, scale: 0.8, rotate: 12 }}
              className="text-[var(--moss)] flex-shrink-0"
              aria-hidden="true"
            >
              <Search className="h-5 w-5" />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence mode="wait">
        {(help || (maxLength && showCharacterCount) || error) && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className="flex items-center justify-between mt-2 gap-2"
            id={helpId}
            role={error ? "alert" : undefined}
            aria-live={error ? "polite" : undefined}
            aria-atomic={error}
          >
            {help && !error && (
              <span className="text-sm text-[var(--muted-foreground)]">
                {help}
              </span>
            )}

            {error && (
              <motion.span
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center gap-1.5 text-sm text-[var(--error)]"
              >
                <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" aria-hidden="true" />
                <span>{error}</span>
              </motion.span>
            )}

            {maxLength && showCharacterCount && !error && (
              <span
                className={cn(
                  "text-sm font-mono tabular-nums",
                  isAtMaxLength ? "text-[var(--error)]" : "text-[var(--muted-foreground)]"
                )}
                aria-live="polite"
              >
                {characterCount} / {maxLength}
              </span>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function TextInputGroup({
  children,
  className,
  orientation = "vertical",
  gap = 4,
}: {
  children: React.ReactNode;
  className?: string;
  orientation?: "vertical" | "horizontal";
  gap?: number;
}) {
  return (
    <div
      className={cn(
        "flex items-start",
        orientation === "vertical" ? "flex-col" : "flex-row",
        `gap-${gap}`,
        className
      )}
      role="group"
    >
      {children}
    </div>
  );
}