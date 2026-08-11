"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export interface MemoryItem {
  id: string;
  fact: string;
  tags?: string[];
  vitality: number;
  bucket: "fresh" | "aging" | "decayed";
  age_days: number;
  updated_at?: number;
}

interface MemoryListProps {
  memories: MemoryItem[];
  className?: string;
  emptyMessage?: string;
  highlightTag?: string | null;
  searchQuery?: string;
  showVitality?: boolean;
  variant?: "recent" | "decay";
}

const bucketColors = {
  fresh: "var(--success)",
  aging: "var(--warning)",
  decayed: "var(--error)",
};

function vitalityBar(vitality: number, bucket: MemoryItem["bucket"]) {
  const segments = 8;
  const filled = Math.round(vitality * segments);
  const color = bucketColors[bucket];
  return (
    <span
      className="inline-flex items-center gap-px font-mono"
      aria-hidden="true"
      title={`vitality ${vitality.toFixed(3)}`}
    >
      {Array.from({ length: segments }).map((_, i) => (
        <span
          key={i}
          className="inline-block w-1 h-3 rounded-sm"
          style={{ backgroundColor: i < filled ? color : "var(--muted)" }}
        />
      ))}
    </span>
  );
}

function highlightText(text: string, query?: string) {
  if (!query) return text;
  const lower = text.toLowerCase();
  const q = query.toLowerCase();
  let idx = lower.indexOf(q);
  if (idx === -1) return text;
  const parts: React.ReactNode[] = [];
  let last = 0;
  let key = 0;
  while (idx !== -1) {
    parts.push(text.slice(last, idx));
    parts.push(
      <mark
        key={key++}
        className="bg-[var(--sage)]/30 text-[var(--foreground)] rounded px-0.5"
      >
        {text.slice(idx, idx + q.length)}
      </mark>
    );
    last = idx + q.length;
    idx = lower.indexOf(q, last);
  }
  parts.push(text.slice(last));
  return <>{parts}</>;
}

export function MemoryList({
  memories,
  className,
  emptyMessage = "No memories",
  highlightTag,
  searchQuery,
  showVitality = true,
  variant = "recent",
}: MemoryListProps) {
  if (memories.length === 0) {
    return (
      <p className={cn("text-[var(--moss)] text-sm text-center py-8", className)}>
        {emptyMessage}
      </p>
    );
  }

  return (
    <div className={cn("flex flex-col gap-2", className)} role="list" aria-label="Memory list">
      {memories.map((mem, index) => {
        const tagMatch = highlightTag && mem.tags?.includes(highlightTag);
        return (
          <motion.div
            key={mem.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(index * 0.03, 0.3) }}
            layout
            className={cn(
              "group relative flex items-start gap-3 p-3 rounded-[10px]",
              "bg-[var(--card)] border border-[var(--border)]",
              "hover:border-[var(--border-active)] hover:bg-[var(--muted)]/40",
              "transition-all duration-200",
              tagMatch && "ring-1 ring-[var(--sage)]/40"
            )}
            role="listitem"
          >
            <div
              className="flex-shrink-0 mt-1 w-2 h-2 rounded-full"
              style={{ backgroundColor: bucketColors[mem.bucket] }}
              aria-label={`${mem.bucket} vitality`}
            />

            <div className="flex-1 min-w-0">
              <p className="text-sm text-[var(--foreground)] leading-snug">
                {highlightText(mem.fact, searchQuery)}
              </p>
              <div className="mt-2 flex items-center gap-3 flex-wrap">
                {mem.tags && mem.tags.length > 0 && (
                  <div className="flex items-center gap-1 flex-wrap">
                    {mem.tags.slice(0, 4).map((tag) => (
                      <span
                        key={tag}
                        className={cn(
                          "px-1.5 py-0.5 rounded text-[10px] font-mono",
                          tag === highlightTag
                            ? "bg-[var(--sage)]/20 text-[var(--sage)]"
                            : "bg-[var(--muted)] text-[var(--moss)]"
                        )}
                      >
                        {tag}
                      </span>
                    ))}
                    {mem.tags.length > 4 && (
                      <span className="text-[10px] text-[var(--moss)]/60">
                        +{mem.tags.length - 4}
                      </span>
                    )}
                  </div>
                )}
                <span className="text-[10px] text-[var(--moss)]/70 font-mono">
                  {mem.age_days}d
                </span>
              </div>
            </div>

            {showVitality && (
              <div className="flex flex-col items-end gap-1 flex-shrink-0">
                {variant === "recent" ? (
                  vitalityBar(mem.vitality, mem.bucket)
                ) : (
                  <span
                    className="font-mono text-sm tabular-nums"
                    style={{ color: bucketColors[mem.bucket] }}
                  >
                    {mem.vitality.toFixed(3)}
                  </span>
                )}
              </div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}