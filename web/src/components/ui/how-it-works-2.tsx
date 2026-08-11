"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Brain,
  Search,
  Zap,
  CheckCircle,
  BarChart2,
} from "lucide-react";

export interface HowItWorksStep {
  number: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  details?: string[];
}

export interface HowItWorksProps {
  steps?: HowItWorksStep[];
  className?: string;
  animated?: boolean;
}

const defaultSteps: HowItWorksStep[] = [
  {
    number: "01",
    icon: <Brain className="h-6 w-6" aria-hidden="true" />,
    title: "Capture memories",
    description: "Add facts, observations, and insights with natural language. Tags are optional — isotope_zero infers them.",
    details: [
      "Local-first: nothing leaves your machine",
      "Supports markdown, code blocks, and structured data",
      "Auto-tags via semantic analysis",
    ],
  },
  {
    number: "02",
    icon: <Search className="h-6 w-6" aria-hidden="true" />,
    title: "Hybrid retrieval",
    description: "Vector + keyword + graph search fused into one ranked list. No embedding API keys required.",
    details: [
      "NumPy/BLAS cosine similarity (no Rust, no network)",
      "BM25 full-text for exact matches",
      "Graph walk for related memories",
    ],
  },
  {
    number: "03",
    icon: <Zap className="h-6 w-6" aria-hidden="true" />,
    title: "Ebbinghaus decay",
    description: "Memories fade like human recall — vitality drops over time unless reinforced. Consolidation merges duplicates.",
    details: [
      "Fresh → Aging → Decayed buckets",
      "Auto-merge similar cards (dry-run first)",
      "Reclaim tokens from decayed memories",
    ],
  },
  {
    number: "04",
    icon: <BarChart2 className="h-6 w-6" aria-hidden="true" />,
    title: "Live dashboard",
    description: "Terminal TUI + browser dashboard showing vitality, tags, recent, and decay candidates in real time.",
    details: [
      "SSE live updates from Python backend",
      "Rich TUI with sparklines + block bars",
      "React + shadcn web UI at localhost",
    ],
  },
];

export function HowItWorks({
  steps = defaultSteps,
  className,
  animated = true,
}: HowItWorksProps) {
  return (
    <section className={cn("py-16 px-6", className)} aria-labelledby="how-it-works-heading">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h2 id="how-it-works-heading" className="text-3xl md:text-4xl font-bold tracking-tight">
            <span className="text-[var(--foreground)]">How </span>
            <span className="text-[var(--sage)]">isotope_zero</span>
            <span className="text-[var(--foreground)]"> works</span>
          </h2>
          <p className="mt-4 text-[var(--moss)] text-lg max-w-2xl mx-auto">
            Four steps to a living, self-organizing memory layer for your AI agents.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {steps.map((step, index) => (
            <motion.article
              key={step.number}
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: animated ? index * 0.1 : 0 }}
              className={cn(
                "relative rounded-[16px] p-6",
                "bg-[var(--card)] border border-[var(--border)]",
                "hover:border-[var(--sage)]/50 hover:shadow-[0_0_0_1px_var(--sage)]",
                "transition-all duration-300"
              )}
            >
              <div className="flex items-start gap-4">
                <div className={cn(
                  "flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center",
                  "bg-[var(--sage)]/10 text-[var(--sage)]"
                )}>
                  {step.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline gap-2 mb-2">
                    <span className={cn(
                      "font-mono text-[var(--sage)] font-bold",
                      "text-sm md:text-base"
                    )}>
                      {step.number}
                    </span>
                    <h3 className="text-lg font-semibold text-[var(--foreground)]">
                      {step.title}
                    </h3>
                  </div>
                  <p className="text-[var(--moss)] text-sm leading-relaxed mb-4">
                    {step.description}
                  </p>
                  {step.details && step.details.length > 0 && (
                    <ul className="space-y-1.5 text-xs text-[var(--moss)]/80">
                      {step.details.map((detail, di) => (
                        <li key={di} className="flex items-center gap-2">
                          <CheckCircle className="h-3 w-3 text-[var(--sage)]/60 flex-shrink-0" aria-hidden="true" />
                          {detail}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>

              {index < steps.length - 1 && (
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ delay: animated ? index * 0.1 + 0.3 : 0 }}
                  className="absolute bottom-0 right-0 w-1/2 h-px bg-gradient-to-r from-[var(--sage)] to-transparent"
                  style={{ transformOrigin: "left center" }}
                  aria-hidden="true"
                />
              )}
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}

export function HowItWorksCompact({ className }: { className?: string }) {
  return (
    <div className={cn("grid sm:grid-cols-2 lg:grid-cols-4 gap-4", className)}>
      {defaultSteps.map((step, index) => (
        <motion.div
          key={step.number}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.08 }}
          className={cn(
            "p-4 rounded-[12px]",
            "bg-[var(--card)] border border-[var(--border)]",
            "hover:border-[var(--sage)]/50 transition-colors"
          )}
        >
          <div className="flex items-center gap-3 mb-3">
            <span className={cn(
              "font-mono text-[var(--sage)] font-bold text-lg",
              "w-8 text-center"
            )}>
              {step.number}
            </span>
            <div className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center",
              "bg-[var(--sage)]/10 text-[var(--sage)]"
            )}>
              {step.icon}
            </div>
          </div>
          <h4 className="font-medium text-sm text-[var(--foreground)] mb-1">
            {step.title}
          </h4>
          <p className="text-[var(--moss)] text-xs leading-snug">
            {step.description}
          </p>
        </motion.div>
      ))}
    </div>
  );
}