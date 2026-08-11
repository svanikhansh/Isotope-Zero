"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

export interface LiquidGlassButtonProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "onAnimationStart" | "onAnimationEnd" | "onDrag" | "onDragStart" | "onDragEnd"> {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
}

export function LiquidGlassButton({
  children,
  variant = "primary",
  size = "md",
  isLoading = false,
  leftIcon,
  rightIcon,
  fullWidth = false,
  className,
  disabled,
  onClick,
  ...props
}: LiquidGlassButtonProps) {
  const baseStyles = cn(
    "relative inline-flex items-center justify-center gap-2 font-medium",
    "rounded-[10px] border transition-all duration-200 ease-out",
    "backdrop-blur-xl",
    "focus:outline-none focus:ring-2 focus:ring-[var(--sage)]/50 focus:border-[var(--sage)]",
    "disabled:opacity-50 disabled:cursor-not-allowed",
    fullWidth && "w-full"
  );

  const sizeStyles = {
    sm: "px-3 py-1.5 text-xs gap-1.5",
    md: "px-5 py-2.5 text-sm gap-2",
    lg: "px-7 py-3.5 text-base gap-2.5",
  };

  const variantStyles = {
    primary: cn(
      "bg-[var(--sage)]/20 border-[var(--sage)]/30 text-[var(--sage)]",
      "hover:bg-[var(--sage)]/30 hover:border-[var(--sage)]/50",
      "active:bg-[var(--sage)]/40 active:border-[var(--sage)]",
      "shadow-[0_0_0_1px_var(--sage)/20]"
    ),
    secondary: cn(
      "bg-[var(--glass-bg)] border-[var(--glass-border)] text-[var(--foreground)]",
      "hover:bg-[var(--muted)] hover:border-[var(--border-active)]",
      "active:bg-[var(--muted)] active:border-[var(--border)]",
    ),
    ghost: cn(
      "bg-transparent border-transparent text-[var(--muted-foreground)]",
      "hover:bg-[var(--muted)] hover:text-[var(--foreground)]",
      "active:bg-[var(--muted)] active:text-[var(--foreground)]",
    ),
    danger: cn(
      "bg-[var(--error)]/20 border-[var(--error)]/30 text-[var(--error)]",
      "hover:bg-[var(--error)]/30 hover:border-[var(--error)]/50",
      "active:bg-[var(--error)]/40 active:border-[var(--error)]",
      "shadow-[0_0_0_1px_var(--error)/20]"
    ),
  };

  return (
    <motion.button
      {...props}
      onClick={onClick}
      disabled={disabled || isLoading}
      className={cn(baseStyles, sizeStyles[size], variantStyles[variant], className)}
      whileHover={!disabled && !isLoading ? { scale: 1.02 } : {}}
      whileTap={!disabled && !isLoading ? { scale: 0.98 } : {}}
      style={{
        transformOrigin: "center",
      }}
    >
      <AnimatePresence mode="wait">
        {!isLoading && leftIcon && (
          <motion.span
            initial={{ opacity: 0, x: -4 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 4 }}
            className="flex-shrink-0"
          >
            {leftIcon}
          </motion.span>
        )}
      </AnimatePresence>

      {isLoading ? (
        <motion.span
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          className="flex items-center gap-2"
        >
          <svg
            className="animate-spin h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="3"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <span>Loading…</span>
        </motion.span>
      ) : (
        <span className="relative z-10">{children}</span>
      )}

      <AnimatePresence mode="wait">
        {!isLoading && rightIcon && (
          <motion.span
            initial={{ opacity: 0, x: 4 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -4 }}
            className="flex-shrink-0"
          >
            {rightIcon}
          </motion.span>
        )}
      </AnimatePresence>

      <div
        className={cn(
          "absolute inset-0 rounded-[10px] pointer-events-none",
          "bg-gradient-to-br from-white/10 via-transparent to-white/5",
          "opacity-0 hover:opacity-100 transition-opacity duration-300"
        )}
        aria-hidden="true"
      />
    </motion.button>
  );
}

export function LiquidGlassButtonGroup({
  children,
  className,
  orientation = "horizontal",
}: {
  children: React.ReactNode;
  className?: string;
  orientation?: "horizontal" | "vertical";
}) {
  return (
    <div
      className={cn(
        "flex items-center",
        orientation === "horizontal" ? "gap-2" : "flex-col gap-2",
        className
      )}
      role="group"
    >
      {children}
    </div>
  );
}