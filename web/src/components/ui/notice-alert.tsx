"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Info,
  CheckCircle,
  AlertTriangle,
  AlertCircle,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

export type NoticeAlertTone = "info" | "success" | "warning" | "destructive" | "primary";

export interface NoticeAlertProps {
  tone: NoticeAlertTone;
  title: string;
  description?: string;
  className?: string;
  dismissible?: boolean;
  onDismiss?: () => void;
  action?: React.ReactNode;
}

const toneStyles: Record<NoticeAlertTone, { icon: React.ReactNode; bg: string; border: string; iconColor: string }> = {
  info: {
    icon: <Info className="h-5 w-5" aria-hidden="true" />,
    bg: "bg-[var(--info)]/10",
    border: "border-[var(--info)]/30",
    iconColor: "text-[var(--info)]",
  },
  success: {
    icon: <CheckCircle className="h-5 w-5" aria-hidden="true" />,
    bg: "bg-[var(--success)]/10",
    border: "border-[var(--success)]/30",
    iconColor: "text-[var(--success)]",
  },
  warning: {
    icon: <AlertTriangle className="h-5 w-5" aria-hidden="true" />,
    bg: "bg-[var(--warning)]/10",
    border: "border-[var(--warning)]/30",
    iconColor: "text-[var(--warning)]",
  },
  destructive: {
    icon: <AlertCircle className="h-5 w-5" aria-hidden="true" />,
    bg: "bg-[var(--error)]/10",
    border: "border-[var(--error)]/30",
    iconColor: "text-[var(--error)]",
  },
  primary: {
    icon: <Info className="h-5 w-5" aria-hidden="true" />,
    bg: "bg-[var(--sage)]/10",
    border: "border-[var(--sage)]/30",
    iconColor: "text-[var(--sage)]",
  },
};

export function NoticeAlert({
  tone = "info",
  title,
  description,
  className,
  dismissible = false,
  onDismiss,
  action,
}: NoticeAlertProps) {
  const styles = toneStyles[tone];
  const [isOpen, setIsOpen] = React.useState(true);

  if (!isOpen) return null;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        initial={{ opacity: 0, y: -12, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -12, scale: 0.96 }}
        className={cn(
          "relative flex items-start gap-3 p-4 rounded-[12px]",
          styles.bg,
          styles.border,
          "border backdrop-blur-sm",
          className
        )}
        role="alert"
        aria-live={tone === "destructive" ? "assertive" : "polite"}
      >
        <div className={cn("flex-shrink-0 mt-0.5", styles.iconColor)}>
          {styles.icon}
        </div>

        <div className="flex-1 min-w-0">
          <h4 className={cn(
            "font-semibold text-sm",
            tone === "destructive" ? "text-[var(--error)]" :
            tone === "success" ? "text-[var(--success)]" :
            tone === "warning" ? "text-[var(--warning)]" :
            "text-[var(--foreground)]"
          )}>
            {title}
          </h4>
          {description && (
            <p className={cn(
              "mt-1 text-sm leading-relaxed",
              "text-[var(--muted-foreground)]"
            )}>
              {description}
            </p>
          )}
          {action && (
            <div className="mt-3">
              {action}
            </div>
          )}
        </div>

        {dismissible && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => {
              setIsOpen(false);
              onDismiss?.();
            }}
            className={cn(
              "flex-shrink-0 p-1 rounded-lg",
              "text-[var(--muted-foreground)] hover:text-[var(--foreground)]",
              "hover:bg-[var(--muted)] transition-colors"
            )}
            aria-label="Dismiss"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </motion.button>
        )}
      </motion.div>
    </AnimatePresence>
  );
}

export function NoticeAlertGroup({ alerts }: { alerts: NoticeAlertProps[] }) {
  return (
    <div className="flex flex-col gap-3" role="region" aria-label="Notifications">
      {alerts.map((alert, index) => (
        <NoticeAlert key={index} {...alert} />
      ))}
    </div>
  );
}