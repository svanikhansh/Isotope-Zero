"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface KpiCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon?: LucideIcon;
  tone?: "default" | "success" | "warning" | "error" | "info";
  index?: number;
}

const toneStyles = {
  default: { text: "text-[var(--foreground)]", icon: "text-[var(--moss)]", glow: "" },
  success: { text: "text-[var(--success)]", icon: "text-[var(--success)]", glow: "shadow-[0_0_24px_-8px_var(--success)]" },
  warning: { text: "text-[var(--warning)]", icon: "text-[var(--warning)]", glow: "shadow-[0_0_24px_-8px_var(--warning)]" },
  error: { text: "text-[var(--error)]", icon: "text-[var(--error)]", glow: "shadow-[0_0_24px_-8px_var(--error)]" },
  info: { text: "text-[var(--info)]", icon: "text-[var(--info)]", glow: "shadow-[0_0_24px_-8px_var(--info)]" },
};

export function KpiCard({ label, value, sub, icon: Icon, tone = "default", index = 0 }: KpiCardProps) {
  const styles = toneStyles[tone];
  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay: index * 0.06, duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
      className={cn(
        "relative overflow-hidden rounded-[14px] p-5",
        "bg-[var(--glass-bg)] backdrop-blur-xl border border-[var(--glass-border)]",
        "hover:border-[var(--border-active)] transition-all duration-300",
        "group",
        styles.glow
      )}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.04] to-transparent pointer-events-none" aria-hidden="true" />
      <div className="relative flex items-start justify-between">
        <div className="min-w-0 flex-1">
          <p className="text-[11px] uppercase tracking-[0.18em] text-[var(--moss)] font-medium">
            {label}
          </p>
          <p className={cn("mt-2 text-3xl font-bold font-mono tabular-nums", styles.text)}>
            {value}
          </p>
          {sub && (
            <p className="mt-1 text-xs text-[var(--moss)]/80">{sub}</p>
          )}
        </div>
        {Icon && (
          <div className={cn(
            "flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center",
            "bg-[var(--muted)] group-hover:scale-110 transition-transform"
          )}>
            <Icon className={cn("h-5 w-5", styles.icon)} aria-hidden="true" />
          </div>
        )}
      </div>
    </motion.div>
  );
}