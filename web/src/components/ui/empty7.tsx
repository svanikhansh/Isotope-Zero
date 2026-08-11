"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  FileText,
  Database,
  Search,
} from "lucide-react";

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
  variant?: "default" | "search" | "error" | "loading";
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
  variant = "default",
}: EmptyStateProps) {
  const defaultIcons = {
    default: <Database className="h-12 w-12 text-[var(--moss)]/50" aria-hidden="true" />,
    search: <Search className="h-12 w-12 text-[var(--moss)]/50" aria-hidden="true" />,
    error: <FileText className="h-12 w-12 text-[var(--error)]/50" aria-hidden="true" />,
    loading: (
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        className="h-12 w-12 border-3 border-[var(--moss)]/20 border-t-[var(--sage)] rounded-full"
        aria-hidden="true"
      />
    ),
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "flex flex-col items-center justify-center text-center py-12 px-6",
        className
      )}
      role={variant === "loading" ? "status" : undefined}
      aria-live={variant === "loading" ? "polite" : undefined}
    >
      <div className="mb-6">{icon ?? defaultIcons[variant]}</div>

      <h3 className={cn(
        "text-lg font-semibold",
        variant === "error" ? "text-[var(--error)]" : "text-[var(--foreground)]"
      )}>
        {title}
      </h3>

      {description && (
        <p className="mt-2 text-[var(--moss)] max-w-sm">
          {description}
        </p>
      )}

      {action && (
        <div className="mt-6">
          {action}
        </div>
      )}
    </motion.div>
  );
}

export function EmptyStateStack({
  children,
  className,
  spacing = "space-y-8",
}: {
  children: React.ReactNode;
  className?: string;
  spacing?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center", spacing, className)}>
      {children}
    </div>
  );
}