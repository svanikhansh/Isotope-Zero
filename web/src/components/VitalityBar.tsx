"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface VitalityBarProps {
  fresh: number;
  aging: number;
  decayed: number;
  total: number;
  className?: string;
  showLabels?: boolean;
  compact?: boolean;
}

export function VitalityBar({
  fresh,
  aging,
  decayed,
  total,
  className,
  showLabels = true,
  compact = false,
}: VitalityBarProps) {
  const freshPct = total > 0 ? (fresh / total) * 100 : 0;
  const agingPct = total > 0 ? (aging / total) * 100 : 0;
  const decayedPct = total > 0 ? (decayed / total) * 100 : 0;

  if (total === 0) {
    return (
      <div className={cn("w-full", className)}>
        <div className="flex h-3 rounded-full overflow-hidden bg-[var(--muted)] border border-[var(--border)]">
          <div className="w-full bg-[var(--muted)]" />
        </div>
        {showLabels && (
          <p className="mt-2 text-xs text-[var(--moss)] text-center">No memories yet</p>
        )}
      </div>
    );
  }

  return (
    <div className={cn("w-full", className)}>
      <div
        className={cn(
          "relative flex overflow-hidden rounded-full border border-[var(--glass-border)]",
          compact ? "h-2" : "h-3"
        )}
        role="img"
        aria-label={`Vitality: ${fresh} fresh, ${aging} aging, ${decayed} decayed out of ${total} total`}
      >
        {freshPct > 0 && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: `${freshPct}%`, opacity: 1 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="bg-[var(--success)] relative"
            style={{ flexGrow: fresh }}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-[var(--sage)] to-[#BBDEFB]" />
          </motion.div>
        )}
        {agingPct > 0 && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: `${agingPct}%`, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.1, ease: "easeOut" }}
            className="bg-[var(--warning)]"
            style={{ flexGrow: aging }}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-[var(--moss)] to-[var(--warning)]" />
          </motion.div>
        )}
        {decayedPct > 0 && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: `${decayedPct}%`, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2, ease: "easeOut" }}
            className="bg-[var(--error)]"
            style={{ flexGrow: decayed }}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-[#6B0F0D] to-[var(--error)]" />
          </motion.div>
        )}
      </div>

      {showLabels && !compact && (
        <div className="mt-3 flex items-center justify-between gap-4 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[var(--success)]" aria-hidden="true" />
            <span className="text-[var(--foreground)] font-mono font-medium">{fresh}</span>
            <span className="text-[var(--moss)]">fresh</span>
            <span className="text-[var(--moss)]/60">({freshPct.toFixed(0)}%)</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[var(--warning)]" aria-hidden="true" />
            <span className="text-[var(--foreground)] font-mono font-medium">{aging}</span>
            <span className="text-[var(--moss)]">aging</span>
            <span className="text-[var(--moss)]/60">({agingPct.toFixed(0)}%)</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[var(--error)]" aria-hidden="true" />
            <span className="text-[var(--foreground)] font-mono font-medium">{decayed}</span>
<span className="text-[var(--moss)]">decayed</span>
            <span className="text-[var(--moss)]/60">({decayedPct.toFixed(0)}%)</span>
          </span>
        </div>
      )}
    </div>
  );
}