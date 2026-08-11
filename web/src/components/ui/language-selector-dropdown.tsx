"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  ChevronDown,
  Globe,
  Check,
  Search,
} from "lucide-react";

export interface LanguageOption {
  code: string;
  name: string;
  nativeName: string;
  flag?: string;
}

export interface LanguageSelectorProps {
  value: string;
  onChange: (code: string) => void;
  options: LanguageOption[];
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  searchable?: boolean;
}

const DEFAULT_LANGUAGES: LanguageOption[] = [
  { code: "en", name: "English", nativeName: "English", flag: "🇺🇸" },
  { code: "es", name: "Spanish", nativeName: "Español", flag: "🇪🇸" },
  { code: "fr", name: "French", nativeName: "Français", flag: "🇫🇷" },
  { code: "de", name: "German", nativeName: "Deutsch", flag: "🇩🇪" },
  { code: "zh", name: "Chinese", nativeName: "中文", flag: "🇨🇳" },
  { code: "ja", name: "Japanese", nativeName: "日本語", flag: "🇯🇵" },
  { code: "ko", name: "Korean", nativeName: "한국어", flag: "🇰🇷" },
  { code: "pt", name: "Portuguese", nativeName: "Português", flag: "🇵🇹" },
  { code: "it", name: "Italian", nativeName: "Italiano", flag: "🇮🇹" },
  { code: "ru", name: "Russian", nativeName: "Русский", flag: "🇷🇺" },
  { code: "ar", name: "Arabic", nativeName: "العربية", flag: "🇸🇦" },
  { code: "hi", name: "Hindi", nativeName: "हिन्दी", flag: "🇮🇳" },
];

export function LanguageSelector({
  value,
  onChange,
  options = DEFAULT_LANGUAGES,
  placeholder = "Select language",
  className,
  disabled = false,
  searchable = true,
}: LanguageSelectorProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [searchQuery, setSearchQuery] = React.useState("");
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const listRef = React.useRef<HTMLDivElement>(null);

  const selectedOption = options.find((o) => o.code === value);
  const filteredOptions = searchable
    ? options.filter(
        (o) =>
          o.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          o.nativeName.toLowerCase().includes(searchQuery.toLowerCase()) ||
          o.code.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : options;

  React.useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (
        triggerRef.current &&
        !triggerRef.current.contains(e.target as Node) &&
        listRef.current &&
        !listRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      setIsOpen(false);
      triggerRef.current?.focus();
    }
    if (e.key === "ArrowDown" && !isOpen) {
      e.preventDefault();
      setIsOpen(true);
    }
  };

  const handleSelect = (code: string) => {
    onChange(code);
    setIsOpen(false);
    setSearchQuery("");
  };

  return (
    <div className={cn("relative w-full max-w-xs", className)}>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => !disabled && setIsOpen((prev) => !prev)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className={cn(
          "w-full flex items-center justify-between gap-3 px-4 py-2.5",
          "rounded-[10px] border transition-all duration-200",
          "bg-[var(--glass-bg)] backdrop-blur-xl",
          disabled
            ? "border-[var(--glass-border)] opacity-50 cursor-not-allowed"
            : "border-[var(--glass-border)] hover:border-[var(--border-active)]",
          "focus:outline-none focus:ring-2 focus:ring-[var(--sage)]/50 focus:border-[var(--sage)]",
          isOpen && "border-[var(--sage)] shadow-[0_0_0_1px_var(--sage)]"
        )}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-label="Select language"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <Globe className="h-4 w-4 text-[var(--moss)] flex-shrink-0" aria-hidden="true" />
          <div className="flex-1 min-w-0 text-left">
            {selectedOption ? (
              <>
                <span className="font-medium text-[var(--foreground)] truncate block">
                  {selectedOption.name}
                </span>
                <span className="text-xs text-[var(--moss)] truncate block">
                  {selectedOption.nativeName}
                </span>
              </>
            ) : (
              <span className="text-[var(--moss)] truncate block">
                {placeholder}
              </span>
            )}
          </div>
        </div>
        <motion.span
          initial={false}
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="flex-shrink-0 text-[var(--moss)]"
        >
          <ChevronDown className="h-4 w-4" aria-hidden="true" />
        </motion.span>
      </button>

      <AnimatePresence mode="wait">
        {isOpen && !disabled && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            className={cn(
              "absolute top-full left-0 right-0 z-50 mt-1.5",
              "bg-[var(--card)] border border-[var(--border)] rounded-[12px]",
              "shadow-xl overflow-hidden backdrop-blur-xl",
              "max-h-64"
            )}
            role="listbox"
            ref={listRef}
          >
            {searchable && (
              <div className="p-3 border-b border-[var(--border)]">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--moss)]" aria-hidden="true" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search languages…"
                    className={cn(
                      "w-full pl-10 pr-4 py-2 rounded-lg",
                      "bg-[var(--muted)] border border-[var(--border)]",
                      "text-[var(--foreground)] placeholder:text-[var(--moss)]",
                      "focus:outline-none focus:border-[var(--sage)] focus:ring-1 focus:ring-[var(--sage)]",
                      "text-sm"
                    )}
                    autoFocus
                    aria-label="Search languages"
                  />
                </div>
              </div>
            )}
            <ul className="py-1 max-h-[320px] overflow-y-auto" role="listbox">
              {filteredOptions.length === 0 ? (
                <li className="px-4 py-3 text-center text-[var(--moss)] text-sm">
                  No languages found
                </li>
              ) : (
                filteredOptions.map((option) => (
                  <motion.li
                    key={option.code}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                  >
                    <button
                      onClick={() => handleSelect(option.code)}
                      role="option"
                      aria-selected={option.code === value}
                      className={cn(
                        "w-full px-4 py-2.5 text-left flex items-center gap-3",
                        "hover:bg-[var(--muted)] transition-colors",
                        option.code === value
                          ? "bg-[var(--sage)]/10 text-[var(--sage)]"
                          : "text-[var(--foreground)]"
                      )}
                    >
                      {option.flag && (
                        <span className="text-lg" aria-hidden="true">
                          {option.flag}
                        </span>
                      )}
                      <div className="flex-1 min-w-0 text-left">
                        <span className="font-medium truncate block">
                          {option.name}
                        </span>
                        <span className="text-xs text-[var(--moss)] truncate block">
                          {option.nativeName}
                        </span>
                      </div>
                      {option.code === value && (
                        <Check className="h-4 w-4 text-[var(--sage)] flex-shrink-0" aria-hidden="true" />
                      )}
                    </button>
                  </motion.li>
                ))
              )}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}