"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface TagCloudProps {
  tags: Array<{ tag: string; count: number }>;
  className?: string;
  activeTag?: string | null;
  onTagClick?: (tag: string) => void;
}

function colorForCount(count: number, max: number) {
  const ratio = max > 0 ? count / max : 0;
  if (ratio > 0.66) return { text: "text-[var(--success)]", bg: "bg-[var(--success)]/15", border: "border-[var(--success)]/30" };
  if (ratio > 0.33) return { text: "text-[var(--info)]", bg: "bg-[var(--info)]/15", border: "border-[var(--info)]/30" };
  return { text: "text-[var(--moss)]", bg: "bg-[var(--muted)]", border: "border-[var(--border)]" };
}

export function TagCloud({ tags, className, activeTag, onTagClick }: TagCloudProps) {
  if (tags.length === 0) {
    return (
      <p className={cn("text-[var(--moss)] text-sm", className)}>No tags yet</p>
    );
  }

  const max = Math.max(...tags.map((t) => t.count));

  return (
    <div className={cn("flex flex-wrap gap-2", className)} role="list" aria-label="Tags">
      {tags.map((tag, index) => {
        const colors = colorForCount(tag.count, max);
        const isActive = activeTag === tag.tag;
        return (
          <motion.button
            key={tag.tag}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.03, duration: 0.2 }}
            whileHover={{ scale: 1.06, y: -2 }}
            whileTap={{ scale: 0.96 }}
            onClick={() => onTagClick?.(tag.tag)}
            className={cn(
              "relative inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full",
              "border text-xs font-mono font-medium transition-colors",
              "focus:outline-none focus:ring-2 focus:ring-[var(--sage)]/50",
              colors.text,
              colors.bg,
              colors.border,
              isActive && "ring-2 ring-[var(--sage)]"
            )}
            role="listitem"
            aria-pressed={isActive}
            aria-label={`Tag ${tag.tag} with ${tag.count} memories`}
          >
            <span className="opacity-70">#</span>
            <span>{tag.tag}</span>
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-[var(--background)]/50 text-[10px] font-mono">
              {tag.count}
            </span>
          </motion.button>
        );
      })}
    </div>
  );
}