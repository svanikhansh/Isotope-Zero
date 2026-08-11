"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, X, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

export interface GooeySearchBarProps {
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  onSubmit?: (value: string) => void;
  className?: string;
  suggestions?: string[];
  onSuggestionClick?: (suggestion: string) => void;
  maxSuggestions?: number;
}

export function GooeySearchBar({
  placeholder = "Search memories…",
  value,
  onChange,
  onSubmit,
  className,
  suggestions = [],
  onSuggestionClick,
  maxSuggestions = 5,
}: GooeySearchBarProps) {
  const [isFocused, setIsFocused] = React.useState(false);
  const [showSuggestions, setShowSuggestions] = React.useState(false);
  const [localValue, setLocalValue] = React.useState(value || "");
  const inputRef = React.useRef<HTMLInputElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  const controlled = value !== undefined;
  const currentValue = controlled ? value : localValue;

  React.useEffect(() => {
    if (controlled) {
      setLocalValue(value || "");
    }
  }, [value, controlled]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    if (!controlled) setLocalValue(newValue);
    onChange?.(newValue);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      onSubmit?.(currentValue);
    }
    if (e.key === "Escape") {
      setShowSuggestions(false);
      inputRef.current?.blur();
    }
  };

  const handleFocus = () => {
    setIsFocused(true);
    setShowSuggestions(true);
  };

  const handleBlur = () => {
    setTimeout(() => {
      setIsFocused(false);
      setShowSuggestions(false);
    }, 150);
  };

  const filteredSuggestions = suggestions
    .filter((s) => s.toLowerCase().includes(currentValue.toLowerCase()))
    .slice(0, maxSuggestions);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative w-full max-w-xl",
        className
      )}
    >
      <div
        className={cn(
          "relative flex items-center gap-3 rounded-[16px] transition-all duration-300 ease-out",
          "bg-[var(--glass-bg)] backdrop-blur-xl border",
          isFocused
            ? "border-[var(--sage)] shadow-[0_0_0_1px_var(--sage),0_8px_32px_rgba(144,202,249,0.15)]"
            : "border-[var(--glass-border)] hover:border-[var(--border-active)]",
          "px-4 py-3"
        )}
      >
        <AnimatePresence mode="wait">
          <motion.div
            initial={{ opacity: 0, scale: 0.8, rotate: -12 }}
            animate={{ opacity: 1, scale: 1, rotate: 0 }}
            exit={{ opacity: 0, scale: 0.8, rotate: 12 }}
            className="text-[var(--moss)] flex-shrink-0"
          >
            <Search className="h-5 w-5" aria-hidden="true" />
          </motion.div>
        </AnimatePresence>

        <input
          ref={inputRef}
          type="text"
          value={currentValue}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          className={cn(
            "flex-1 bg-transparent border-none outline-none text-[var(--foreground)]",
            "placeholder:text-[var(--moss)] font-mono text-sm",
            "w-full"
          )}
          autoComplete="off"
          spellCheck={false}
          aria-label="Search memories"
          aria-autocomplete="list"
          aria-controls="search-suggestions"
          aria-expanded={showSuggestions && filteredSuggestions.length > 0}
        />

        <AnimatePresence mode="wait">
          {currentValue && (
            <motion.button
              initial={{ opacity: 0, scale: 0, rotate: -90 }}
              animate={{ opacity: 1, scale: 1, rotate: 0 }}
              exit={{ opacity: 0, scale: 0, rotate: 90 }}
              onClick={() => {
                if (controlled) onChange?.(""); else setLocalValue("");
                inputRef.current?.focus();
              }}
              className={cn(
                "flex items-center justify-center p-1 rounded-lg",
                "text-[var(--moss)] hover:text-[var(--foreground)]",
                "hover:bg-[var(--muted)] transition-colors",
                "flex-shrink-0"
              )}
              aria-label="Clear search"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </motion.button>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {!currentValue && isFocused && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="flex items-center gap-1 text-[var(--sage)] text-xs font-medium px-2 py-1 rounded-full bg-[var(--sage)]/10"
            >
              <Sparkles className="h-3 w-3" aria-hidden="true" />
              <span>Search</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence mode="wait">
        {showSuggestions && filteredSuggestions.length > 0 && (
          <motion.ul
            id="search-suggestions"
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            className={cn(
              "absolute top-full left-0 right-0 mt-2 z-50",
              "bg-[var(--card)] border border-[var(--border)] rounded-[12px]",
              "shadow-xl overflow-hidden",
              "backdrop-blur-xl"
            )}
            role="listbox"
          >
            {filteredSuggestions.map((suggestion, index) => (
              <motion.li
                key={suggestion}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ delay: index * 0.03 }}
              >
                <button
                  onClick={() => {
                    onSuggestionClick?.(suggestion);
                    if (controlled) onChange?.(suggestion); else setLocalValue(suggestion);
                    onSubmit?.(suggestion);
                    setShowSuggestions(false);
                  }}
                  className={cn(
                    "w-full px-4 py-3 text-left text-sm",
                    "text-[var(--foreground)] hover:bg-[var(--muted)]",
                    "flex items-center gap-3 transition-colors",
                    "focus:outline-none focus:bg-[var(--muted)]"
                  )}
                  role="option"
                >
                  <Search className="h-4 w-4 text-[var(--moss)] flex-shrink-0" aria-hidden="true" />
                  <span>{suggestion}</span>
                </button>
              </motion.li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}