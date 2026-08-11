"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Radio, Wifi, WifiOff } from "lucide-react";

interface ConnectionBadgeProps {
  connected: boolean;
  usingSSE: boolean;
  lastUpdate?: string;
  className?: string;
}

export function ConnectionBadge({ connected, usingSSE, lastUpdate, className }: ConnectionBadgeProps) {
  const status = connected ? (usingSSE ? "stream" : "poll") : "offline";
  const color = connected ? "var(--success)" : "var(--error)";
  const Icon = connected ? (usingSSE ? Radio : Wifi) : WifiOff;

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 px-2.5 py-1 rounded-full",
        "bg-[var(--glass-bg)] backdrop-blur-xl border border-[var(--glass-border)]",
        className
      )}
      role="status"
      aria-live="polite"
    >
      <motion.span
        animate={{ scale: connected ? [1, 1.3, 1] : 1, opacity: connected ? [0.6, 1, 0.6] : 0.5 }}
        transition={{ duration: 2, repeat: connected ? Infinity : 0 }}
        className="w-2 h-2 rounded-full"
        style={{ backgroundColor: color }}
      />
      <Icon className="h-3 w-3" style={{ color }} aria-hidden="true" />
      <span className="text-[10px] font-mono uppercase tracking-wider" style={{ color }}>
        {status}
      </span>
      {lastUpdate && (
        <span className="text-[10px] font-mono text-[var(--moss)] border-l border-[var(--border)] pl-2 ml-0.5">
          {lastUpdate}
        </span>
      )}
    </div>
  );
}