"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

export interface Phase {
  at: number;
  label: string;
}

export interface ProgressiveFluxLoaderProps {
  value: number;
  phases: Phase[];
  className?: string;
  size?: "sm" | "md" | "lg";
  showValue?: boolean;
}

const sizeClasses = {
  sm: "h-2",
  md: "h-3",
  lg: "h-4",
};

const sizeValueClasses = {
  sm: "text-xs",
  md: "text-sm",
  lg: "text-base",
};

export function ProgressiveFluxLoader({
  value,
  phases = [],
  className,
  size = "md",
  showValue = true,
}: ProgressiveFluxLoaderProps) {
  const clampedValue = Math.max(0, Math.min(100, value));
  const sortedPhases = React.useMemo(
    () => [...phases].sort((a, b) => a.at - b.at),
    [phases]
  );

  const currentPhase = React.useMemo(() => {
    let phase = sortedPhases[0] || { at: 0, label: "" };
    for (const p of sortedPhases) {
      if (clampedValue >= p.at) phase = p;
      else break;
    }
    return phase;
  }, [clampedValue, sortedPhases]);

  const nextPhase = React.useMemo(() => {
    for (const p of sortedPhases) {
      if (clampedValue < p.at) return p;
    }
    return null;
  }, [clampedValue, sortedPhases]);

  const progressToNext = React.useMemo(() => {
    if (!nextPhase) return 0;
    const prevAt = currentPhase.at;
    const nextAt = nextPhase.at;
    const span = nextAt - prevAt;
    if (span <= 0) return 1;
    return Math.max(0, Math.min(1, (clampedValue - prevAt) / span));
  }, [clampedValue, currentPhase, nextPhase]);

  return (
    <div className={cn("w-full", className)}>
      <div className="flex items-center justify-between gap-4 mb-2">
        <motion.span
          key={currentPhase.label}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          className={cn("font-medium", sizeValueClasses[size])}
        >
          {currentPhase.label}
        </motion.span>
        {showValue && (
          <motion.span
            key={clampedValue}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className={cn("font-mono tabular-nums text-[var(--sage)]", sizeValueClasses[size])}
          >
            {Math.round(clampedValue)}%
          </motion.span>
        )}
      </div>

      <div
        className={cn(
          "relative overflow-hidden rounded-full bg-[var(--muted)]",
          sizeClasses[size]
        )}
        role="progressbar"
        aria-valuenow={Math.round(clampedValue)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Progress"
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={clampedValue}
            initial={{ width: 0 }}
            animate={{ width: `${clampedValue}%` }}
            transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
            className={cn(
              "h-full rounded-full relative overflow-hidden",
              "bg-gradient-to-r from-[var(--sage)] via-[var(--moss)] to-[var(--forest)]"
            )}
          >
            <div
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-[izero-shimmer_2s_infinite]"
            />
          </motion.div>
        </AnimatePresence>

        {sortedPhases.map((phase) => (
          <motion.div
            key={phase.at}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: clampedValue >= phase.at ? 1 : 0.4, scale: 1 }}
            className={cn(
              "absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full",
              "bg-[var(--card)] border-2 border-[var(--border)]",
              "transition-all duration-300",
              clampedValue >= phase.at && "bg-[var(--sage)] border-[var(--sage)]"
            )}
            style={{ left: `${phase.at}%` }}
          />
        ))}

        {nextPhase && (
          <motion.div
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 + progressToNext * 0.5 }}
            className={cn(
              "absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full",
              "bg-[var(--muted)] border-2 border-[var(--sage)]",
              "transition-all duration-300"
            )}
            style={{ left: `${nextPhase.at}%` }}
          />
        )}
      </div>

      <div className="flex items-center justify-between gap-2 mt-2 text-xs text-[var(--moss)]">
        {sortedPhases.map((phase) => (
          <motion.span
            key={phase.at}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: clampedValue >= phase.at ? 1 : 0.5 }}
            style={{ left: `${phase.at}%` }}
            className="absolute"
          >
            {phase.at === 0 ? phase.label : phase.at === 100 ? phase.label : `${phase.at}%`}
          </motion.span>
        ))}
      </div>
    </div>
  );
}