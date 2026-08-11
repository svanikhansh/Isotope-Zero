"use client";

import { useState, useMemo, useCallback, type JSX } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Database,
  HardDrive,
  Cpu,
  Recycle,
  RefreshCw,
  Activity,
  Gauge,
  Layers,
  Sparkles,
  ChevronDown,
  Terminal,
  Clock,
  TrendingDown,
} from "lucide-react";

import { useDashboard } from "@/hooks/use-dashboard";
import { KpiCard } from "@/components/KpiCard";
import { VitalityBar } from "@/components/VitalityBar";
import { VitalityChart } from "@/components/VitalityChart";
import { TagCloud } from "@/components/TagCloud";
import { MemoryList } from "@/components/MemoryList";
import { ConnectionBadge } from "@/components/ConnectionBadge";
import { Dock, type DockItem } from "@/components/ui/dock";
import { GooeySearchBar } from "@/components/ui/animated-search-bar";
import { NoticeAlert } from "@/components/ui/notice-alert";
import { EmptyState } from "@/components/ui/empty7";
import { HowItWorksCompact } from "@/components/ui/how-it-works-2";
import { LiquidGlassButton } from "@/components/ui/liquid-glass-button";
import { ProfileDropdown } from "@/components/ui/profile-dropdown";

const SECTIONS = ["overview", "vitality", "memories", "tags", "decay"] as const;

export default function App() {
  const { state, loading, error, connected, usingSSE, refresh } = useDashboard({
    pollInterval: 4000,
  });

  const [search, setSearch] = useState("");
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(false);

  // Filter recent memories by search + tag
  const filteredRecent = useMemo(() => {
    if (!state) return [];
    let items = state.recent;
    if (activeTag) items = items.filter((m) => m.tags.includes(activeTag));
    if (search.trim()) {
      const q = search.toLowerCase();
      items = items.filter(
        (m) =>
          m.fact.toLowerCase().includes(q) ||
          m.tags.some((t) => t.toLowerCase().includes(q))
      );
    }
    return items;
  }, [state, search, activeTag]);

  const filteredDecay = useMemo(() => {
    if (!state) return [];
    let items = state.decay;
    if (activeTag) items = items.filter((m) => m.tags?.includes(activeTag));
    if (search.trim()) {
      const q = search.toLowerCase();
      items = items.filter((m) => m.fact.toLowerCase().includes(q));
    }
    return items;
  }, [state, search, activeTag]);

  const suggestions = useMemo(
    () => Array.from(new Set((state?.top_tags || []).map((t) => t.tag))).slice(0, 6),
    [state]
  );

  const handleSearch = useCallback((value: string) => {
    setSearch(value);
    if (value.trim()) setActiveTag(null);
  }, []);

  const handleTagClick = useCallback((tag: string) => {
    setActiveTag((prev) => (prev === tag ? null : tag));
  }, []);

  const lastUpdate = state
    ? new Date(state.rendered_at).toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : undefined;

  const isEmpty = state && state.count === 0;
  const hasError = !!error;
  const isReady = !!state && !hasError;

  const dockItems: DockItem[] = SECTIONS.map((section) => {
    const icons: Record<string, JSX.Element> = {
      overview: <Gauge className="h-5 w-5" />,
      vitality: <Activity className="h-5 w-5" />,
      memories: <Layers className="h-5 w-5" />,
      tags: <Sparkles className="h-5 w-5" />,
      decay: <TrendingDown className="h-5 w-5" />,
    };
    return {
      title: section,
      icon: icons[section],
      onClick: () => {
        const el = document.getElementById(`section-${section}`);
        el?.scrollIntoView({ behavior: "smooth", block: "start" });
      },
    };
  });

  return (
    <div className="min-h-full bg-[var(--background)] text-[var(--foreground)]">
      {/* Ambient gradient glow */}
      <div
        className="pointer-events-none fixed inset-0 opacity-40"
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(144,202,249,0.12), transparent), radial-gradient(ellipse 60% 40% at 100% 100%, rgba(33,150,243,0.15), transparent)",
        }}
        aria-hidden="true"
      />

      {/* Header */}
      <header className="relative sticky top-0 z-40 backdrop-blur-xl bg-[var(--background)]/70 border-b border-[var(--border)]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="w-9 h-9 rounded-xl bg-gradient-to-br from-[var(--sage)] to-[var(--forest)] flex items-center justify-center"
            >
              <Database className="h-5 w-5 text-[var(--background)]" aria-hidden="true" />
            </motion.div>
            <div>
              <h1 className="text-lg font-bold tracking-tight leading-none">
                isotope<span className="text-[var(--sage)]">_zero</span>
              </h1>
              <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--moss)] mt-0.5">
                local-first cognitive memory
              </p>
            </div>
          </div>

          <div className="flex-1 max-w-xl hidden md:block">
            <GooeySearchBar
              value={search}
              onChange={handleSearch}
              onSubmit={setSearch}
              suggestions={suggestions}
              onSuggestionClick={setSearch}
              placeholder="Search memories, tags, facts…"
            />
          </div>

          <div className="flex items-center gap-3">
            <ConnectionBadge
              connected={connected}
              usingSSE={usingSSE}
              lastUpdate={lastUpdate}
            />
            <LiquidGlassButton
              variant="secondary"
              size="sm"
              onClick={refresh}
              leftIcon={<RefreshCw className="h-3.5 w-3.5" />}
              aria-label="Refresh"
            >
              <span className="hidden sm:inline">Refresh</span>
            </LiquidGlassButton>
            <ProfileDropdown
              user={{
                name: "local",
                email: "localhost",
                initials: "iz",
              }}
            />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="relative max-w-7xl mx-auto px-6 py-8 pb-32">
        {/* Mobile search */}
        <div className="md:hidden mb-6">
          <GooeySearchBar
            value={search}
            onChange={handleSearch}
            onSubmit={setSearch}
            suggestions={suggestions}
            onSuggestionClick={setSearch}
            placeholder="Search memories, tags, facts…"
          />
        </div>

        {/* Loading state */}
        {loading && !state && (
          <div className="flex items-center justify-center py-24">
            <EmptyState variant="loading" title="Connecting…" description="Fetching dashboard state from the local isotope_zero server." />
          </div>
        )}

        {/* Error */}
        <AnimatePresence mode="wait">
          {hasError && !isReady && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <NoticeAlert
                tone="destructive"
                title="Connection failed"
                description={`The dashboard can't reach the isotope_zero server. Make sure \`izero serve\` is running (default port 8930). Error: ${error}`}
                dismissible
                action={
                  <LiquidGlassButton variant="primary" size="sm" onClick={refresh} leftIcon={<RefreshCw className="h-3.5 w-3.5" />}>
                    Retry connection
                  </LiquidGlassButton>
                }
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error banner but with prior data */}
        <AnimatePresence mode="wait">
          {hasError && isReady && (
            <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} className="mb-6">
              <NoticeAlert
                tone="warning"
                title="Stream interrupted"
                description={`Live updates paused — showing last known state. ${error}`}
                dismissible
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Onboarding prompt for empty store */}
        <AnimatePresence mode="wait">
          {showOnboarding && isEmpty && isReady && (
            <motion.section
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-8 overflow-hidden"
            >
              <div className="rounded-[16px] bg-[var(--card)] border border-[var(--border)] p-6">
                <h2 className="text-lg font-semibold mb-1">How isotope_zero works</h2>
                <p className="text-sm text-[var(--moss)] mb-6">Four steps to a living memory layer.</p>
                <HowItWorksCompact />
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Empty store */}
        {isEmpty && isReady && (
          <div className="flex flex-col items-center justify-center py-16">
            <EmptyState
              variant="default"
              title="No memories yet"
              description="Your memory store is empty. Add your first memory with the CLI to see it appear here in real time."
              action={
                <LiquidGlassButton
                  variant="primary"
                  size="md"
                  onClick={() => setShowOnboarding((prev) => !prev)}
                  rightIcon={<ChevronDown className="h-4 w-4" />}
                >
                  {showOnboarding ? "Hide guide" : "Learn how it works"}
                </LiquidGlassButton>
              }
            />
            <div className="mt-6 rounded-[10px] bg-[var(--card)] border border-[var(--border)] p-4 font-mono text-sm text-[var(--moss)] max-w-lg">
              <span className="text-[var(--sage)]">$</span> izero add "isotope_zero stores facts locally"
            </div>
          </div>
        )}

        {/* Main dashboard */}
        {isReady && !isEmpty && (
          <AnimatePresence mode="wait">
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.3 }}
            >
              {/* KPI row */}
              <div id="section-overview" className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8 scroll-mt-20">
                <KpiCard
                  index={0}
                  label="cards"
                  value={state!.count.toLocaleString()}
                  icon={Database}
                  tone="default"
                />
                <KpiCard
                  index={1}
                  label="storage"
                  value={state!.size_human}
                  icon={HardDrive}
                  tone="info"
                  sub={state!.db_path.split("/").pop() || state!.db_path}
                />
                <KpiCard
                  index={2}
                  label="tokens"
                  value={`~${state!.tokens_total.toLocaleString()}`}
                  icon={Cpu}
                  tone="success"
                />
                <KpiCard
                  index={3}
                  label="reclaimable"
                  value={`~${state!.reclaimable_tokens.toLocaleString()}`}
                  icon={Recycle}
                  tone={state!.reclaimable_tokens > 0 ? "warning" : "default"}
                  sub={state!.reclaimable_tokens > 0 ? "from decayed" : "all fresh"}
                />
              </div>

              {/* Vitality + Decay alert */}
              {state!.decay_total > 0 && (
                <div className="mb-6">
                  <NoticeAlert
                    tone="warning"
                    title={`${state!.decay_total} memory${state!.decay_total === 1 ? "" : "ies"} decaying`}
                    description={`~${state!.reclaimable_tokens.toLocaleString()} tokens could be reclaimed. Consider running \`izero inspect --dry-run-consolidation\`.`}
                  />
                </div>
              )}

              {/* Vitality section — chart + linear bar */}
              <section id="section-vitality" className="grid lg:grid-cols-3 gap-6 mb-8 scroll-mt-20">
                <div className="lg:col-span-1 rounded-[16px] bg-[var(--card)] border border-[var(--border)] p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-[var(--foreground)] flex items-center gap-2">
                      <Activity className="h-4 w-4 text-[var(--sage)]" />
                      Vitality
                    </h3>
                    <span className="text-[10px] uppercase tracking-wider text-[var(--moss)]">
                      Ebbinghaus
                    </span>
                  </div>
                  <VitalityChart
                    fresh={state!.histogram.fresh}
                    aging={state!.histogram.aging}
                    decayed={state!.histogram.decayed}
                    total={state!.count}
                  />
                </div>

                <div className="lg:col-span-2 rounded-[16px] bg-[var(--card)] border border-[var(--border)] p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-[var(--foreground)] flex items-center gap-2">
                      <Gauge className="h-4 w-4 text-[var(--sage)]" />
                      Distribution
                    </h3>
                    <span className="text-[10px] uppercase tracking-wider text-[var(--moss)] font-mono">
                      {state!.count} total
                    </span>
                  </div>
                  <VitalityBar
                    fresh={state!.histogram.fresh}
                    aging={state!.histogram.aging}
                    decayed={state!.histogram.decayed}
                    total={state!.count}
                    className="mt-6"
                  />
                  <div className="mt-6 pt-6 border-t border-[var(--border)]">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-xs font-semibold text-[var(--moss)] uppercase tracking-wider">
                        Modes
                      </h4>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <span className="px-2 py-1 rounded-md bg-[var(--muted)] text-[var(--foreground)] font-mono">
                        {state!.mode}
                      </span>
                      <span className="px-2 py-1 rounded-md bg-[var(--muted)] text-[var(--moss)] font-mono flex items-center gap-1.5">
                        <Database className="h-3 w-3" />
                        sqlite
                      </span>
                      <span className="px-2 py-1 rounded-md bg-[var(--muted)] text-[var(--moss)] font-mono flex items-center gap-1.5">
                        <Terminal className="h-3 w-3" />
                        offline
                      </span>
                    </div>
                  </div>
                </div>
              </section>

              {/* Tags */}
              {state!.top_tags.length > 0 && (
                <section id="section-tags" className="mb-8 rounded-[16px] bg-[var(--card)] border border-[var(--border)] p-6 scroll-mt-20">
                  <div className="flex items-center justify-between mb-5">
                    <h3 className="text-sm font-semibold text-[var(--foreground)] flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-[var(--sage)]" />
                      Tags
                      {activeTag && (
                        <button
                          onClick={() => setActiveTag(null)}
                          className="ml-2 text-[10px] uppercase tracking-wider text-[var(--moss)] hover:text-[var(--foreground)]"
                        >
                          clear filter ×
                        </button>
                      )}
                    </h3>
                    <span className="text-[10px] uppercase tracking-wider text-[var(--moss)]">
                      {state!.top_tags.length} unique
                    </span>
                  </div>
                  <TagCloud
                    tags={state!.top_tags}
                    activeTag={activeTag}
                    onTagClick={handleTagClick}
                  />
                </section>
              )}

              {/* Recent + Decay two-column */}
              <section id="section-memories" className="grid lg:grid-cols-2 gap-6 scroll-mt-20">
                <div className="rounded-[16px] bg-[var(--card)] border border-[var(--border)] p-6 max-h-[640px] flex flex-col">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-[var(--foreground)] flex items-center gap-2">
                      <Layers className="h-4 w-4 text-[var(--sage)]" />
                      Recent
                    </h3>
                    <span className="text-[10px] uppercase tracking-wider text-[var(--moss)]">
                      {filteredRecent.length}
                      {(search || activeTag) && ` of ${state!.recent.length}`}
                    </span>
                  </div>
                  <div className="overflow-y-auto pr-2 -mr-2 flex-1">
                    <MemoryList
                      memories={filteredRecent}
                      variant="recent"
                      searchQuery={search}
                      highlightTag={activeTag}
                      emptyMessage={
                        search || activeTag
                          ? "No memories match this filter"
                          : "No recent memories"
                      }
                    />
                  </div>
                </div>

                <div className="rounded-[16px] bg-[var(--card)] border border-[var(--border)] p-6 max-h-[640px] flex flex-col">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-[var(--foreground)] flex items-center gap-2">
                      <TrendingDown className="h-4 w-4 text-[var(--warning)]" />
                      Decay candidates
                    </h3>
                    <span className="text-[10px] uppercase tracking-wider text-[var(--moss)]">
                      {state!.decay_total} total
                    </span>
                  </div>
                  <div className="overflow-y-auto pr-2 -mr-2 flex-1">
                    <MemoryList
                      memories={filteredDecay}
                      variant="decay"
                      searchQuery={search}
                      highlightTag={activeTag}
                      emptyMessage={
                        state!.decay_total === 0
                          ? "No memories decaying — healthy store"
                          : "No decay candidates match this filter"
                      }
                    />
                  </div>
                </div>
              </section>

              {/* Footer info */}
              <div className="mt-8 flex items-center justify-center gap-4 text-[10px] uppercase tracking-wider text-[var(--moss)]/60">
                <span className="flex items-center gap-1.5">
                  <Clock className="h-3 w-3" />
                  rendered {lastUpdate}
                </span>
                <span>·</span>
                <span>mempool: {state!.merged_cards} merged</span>
                <span>·</span>
                <span>read-only</span>
              </div>
            </motion.div>
          </AnimatePresence>
        )}
      </main>

      {/* Dock */}
      <Dock
        items={dockItems}
        position="bottom"
      />
    </div>
  );
}
