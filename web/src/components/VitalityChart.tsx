"use client";

import { motion } from "framer-motion";
import { ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis } from "recharts";
import { cn } from "@/lib/utils";

export interface VitalityChartProps {
  fresh: number;
  aging: number;
  decayed: number;
  total: number;
  className?: string;
}

export function VitalityChart({ fresh, aging, decayed, total, className }: VitalityChartProps) {
  if (total === 0) {
    return (
      <div className={cn("flex items-center justify-center h-full min-h-[160px]", className)}>
        <p className="text-[var(--moss)] text-sm">No memories to chart</p>
      </div>
    );
  }

  const healthyPct = total > 0 ? ((fresh + aging * 0.5) / total) * 100 : 0;
  const decayedPct = total > 0 ? (decayed / total) * 100 : 0;

  const data = [
    { name: "vitality", value: healthyPct, fill: "url(#vitalGrad)" },
  ];

  return (
    <div className={cn("relative w-full h-full min-h-[160px] flex flex-col items-center justify-center", className)}>
      <ResponsiveContainer width="100%" height={170}>
        <RadialBarChart
          innerRadius="74%"
          outerRadius="100%"
          data={data}
          startAngle={90}
          endAngle={-270}
        >
          <defs>
            <linearGradient id="vitalGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#90CAF9" />
              <stop offset="100%" stopColor="#2196F3" />
            </linearGradient>
          </defs>
          <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
          <RadialBar
            dataKey="value"
            cornerRadius={16}
            background={{ fill: "var(--muted)", opacity: 0.5 }}
          />
        </RadialBarChart>
      </ResponsiveContainer>

      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <motion.span
          key={Math.round(healthyPct)}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-3xl font-bold font-mono tabular-nums text-[var(--foreground)]"
        >
          {healthyPct.toFixed(0)}
          <span className="text-lg text-[var(--moss)]">%</span>
        </motion.span>
        <span className="text-[10px] uppercase tracking-[0.18em] text-[var(--moss)] mt-0.5">
          healthy
        </span>
        {decayedPct > 0 && (
          <span className="text-[10px] text-[var(--error)]/80 mt-2 font-mono">
            {decayedPct.toFixed(0)}% decayed
          </span>
        )}
      </div>
    </div>
  );
}