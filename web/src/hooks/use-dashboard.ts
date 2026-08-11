"use client";

import { useState, useEffect, useCallback, useRef } from "react";

export interface DashboardState {
  db_path: string;
  size_bytes: number;
  size_human: string;
  mode: string;
  count: number;
  histogram: {
    fresh: number;
    aging: number;
    decayed: number;
  };
  pct: {
    fresh: number;
    aging: number;
    decayed: number;
  };
  recent: Array<{
    id: string;
    fact: string;
    tags: string[];
    vitality: number;
    bucket: "fresh" | "aging" | "decayed";
    age_days: number;
    updated_at: number;
  }>;
  decay: Array<{
    id: string;
    fact: string;
    tags?: string[];
    vitality: number;
    age_days: number;
    bucket: "fresh" | "aging" | "decayed";
  }>;
  decay_total: number;
  tags: Record<string, number>;
  top_tags: Array<{ tag: string; count: number }>;
  tokens_total: number;
  reclaimable_tokens: number;
  merged_cards: number;
  rendered_at: string;
}

interface UseDashboardOptions {
  pollInterval?: number;
  sseEnabled?: boolean;
  onError?: (error: Error) => void;
}

export function useDashboard({
  pollInterval = 5000,
  sseEnabled = true,
  onError,
}: UseDashboardOptions = {}) {
  const [state, setState] = useState<DashboardState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [usingSSE, setUsingSSE] = useState(false);

  const eventSourceRef = useRef<EventSource | null>(null);
  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimerRef = useRef<NodeJS.Timeout | null>(null);
  const sseDeadRef = useRef<number>(0);
  const mountedRef = useRef(true);

  const fetchState = useCallback(async () => {
    try {
      const response = await fetch("/api/state", {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      if (mountedRef.current) {
        setState(data);
        setError(null);
        setLoading(false);
      }
    } catch (err) {
      if (mountedRef.current) {
        const msg = err instanceof Error ? err.message : String(err);
        setError(msg);
        setLoading(false);
        onError?.(err instanceof Error ? err : new Error(msg));
      }
    }
  }, [onError]);

  const connectSSE = useCallback(() => {
    if (!sseEnabled || eventSourceRef.current) return;

    try {
      const es = new EventSource("/api/events");
      eventSourceRef.current = es;

      es.onopen = () => {
        if (mountedRef.current) {
          setConnected(true);
          setUsingSSE(true);
          sseDeadRef.current = 0;
        }
      };

      es.onmessage = (event) => {
        if (mountedRef.current) {
          try {
            const data = JSON.parse(event.data);
            if (data && typeof data === "object" && "count" in data) {
              setState(data);
              setError(null);
              setLoading(false);
            }
          } catch {
            // Ignore parse errors (heartbeats, etc.)
          }
        }
      };

      es.onerror = () => {
        if (mountedRef.current) {
          setConnected(false);
          sseDeadRef.current = Date.now();
          // Don't immediately close - let the poll fallback handle it
        }
      };
    } catch {
      if (mountedRef.current) {
        setUsingSSE(false);
      }
    }
  }, [sseEnabled]);

  const disconnectSSE = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (mountedRef.current) {
      setConnected(false);
      setUsingSSE(false);
    }
  }, []);

  const startPolling = useCallback(() => {
    if (pollTimerRef.current) return;
    pollTimerRef.current = setInterval(() => {
      if (mountedRef.current) {
        // Only poll if SSE has been dead for >10s
        const sseDeadMs = sseDeadRef.current ? Date.now() - sseDeadRef.current : Infinity;
        if (sseDeadMs > 10000) {
          fetchState();
        }
      }
    }, pollInterval);
  }, [fetchState, pollInterval]);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimerRef.current) return;
    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null;
      if (mountedRef.current && sseEnabled) {
        disconnectSSE();
        connectSSE();
      }
    }, 3000);
  }, [sseEnabled, connectSSE, disconnectSSE]);

  // Initial load
  useEffect(() => {
    fetchState();
    if (sseEnabled) {
      connectSSE();
      startPolling();
    }
    return () => {
      mountedRef.current = false;
      disconnectSSE();
      stopPolling();
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
    };
  }, [fetchState, connectSSE, disconnectSSE, startPolling, stopPolling, sseEnabled]);

  // Monitor SSE health and reconnect
  useEffect(() => {
    if (!sseEnabled) return;
    const checkInterval = setInterval(() => {
      if (!mountedRef.current) {
        clearInterval(checkInterval);
        return;
      }
      const sseDeadMs = sseDeadRef.current ? Date.now() - sseDeadRef.current : 0;
      if (sseDeadMs > 15000 && usingSSE) {
        scheduleReconnect();
      } else if (sseDeadMs < 5000 && !usingSSE && sseEnabled) {
        // SSE recovered
        connectSSE();
      }
    }, 5000);
    return () => clearInterval(checkInterval);
  }, [sseEnabled, usingSSE, connectSSE, scheduleReconnect]);

  const refresh = useCallback(() => {
    fetchState();
  }, [fetchState]);

  return {
    state,
    loading,
    error,
    connected,
    usingSSE,
    refresh,
  };
}