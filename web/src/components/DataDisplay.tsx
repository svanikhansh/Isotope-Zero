"use client";

/**
 * DataDisplay - Barrel export for all data display components
 *
 * This module provides a unified export surface for data visualization
 * and display components used throughout the application.
 */

// KPI Card - Key performance indicator display
export { KpiCard } from "./KpiCard";
export type { KpiCardProps } from "./KpiCard";

// Vitality Bar - Segmented vitality progress bar
export { VitalityBar } from "./VitalityBar";
export type { VitalityBarProps } from "./VitalityBar";

// Vitality Chart - Radial vitality visualization
export { VitalityChart } from "./VitalityChart";
export type { VitalityChartProps } from "./VitalityChart";

// Tag Cloud - Interactive tag visualization
export { TagCloud } from "./TagCloud";
export type { TagCloudProps } from "./TagCloud";

// Memory List - Memory item list display
export { MemoryList } from "./MemoryList";
export type { MemoryListProps, MemoryItem } from "./MemoryList";

// Connection Badge - Connection status indicator
export { ConnectionBadge } from "./ConnectionBadge";
export type { ConnectionBadgeProps } from "./ConnectionBadge";

// Re-export all for convenience
export * from "./KpiCard";
export * from "./VitalityBar";
export * from "./VitalityChart";
export * from "./TagCloud";
export * from "./MemoryList";
export * from "./ConnectionBadge";

/**
 * Component categories for discovery:
 *
 * KPI & Metrics:
 *   - KpiCard - Single metric display with icon, trend, tone variants
 *
 * Vitality & Health:
 *   - VitalityBar - Segmented bar (fresh/aging/decayed)
 *   - VitalityChart - Radial gauge visualization
 *
 * Tags & Categories:
 *   - TagCloud - Interactive tag cloud with counts
 *
 * Lists & Collections:
 *   - MemoryList - Memory items with vitality, tags, highlighting
 *
 * Status & Indicators:
 *   - ConnectionBadge - Real-time connection status (stream/poll/offline)
 */