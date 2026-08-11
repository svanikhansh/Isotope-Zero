"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  User,
  Settings,
  LogOut,
  Bell,
  HelpCircle,
  ChevronRight,
} from "lucide-react";

export interface ProfileDropdownProps {
  user: {
    name: string;
    email: string;
    avatar?: string;
    initials?: string;
  };
  items?: ProfileDropdownItem[];
  className?: string;
  trigger?: React.ReactNode;
  onSignOut?: () => void;
  onSettings?: () => void;
}

export interface ProfileDropdownItem {
  label: string;
  icon?: React.ReactNode;
  onClick: () => void;
  variant?: "default" | "destructive";
  shortcut?: string;
}

const DEFAULT_ITEMS: ProfileDropdownItem[] = [
  {
    label: "Profile",
    icon: <User className="h-4 w-4" aria-hidden="true" />,
    onClick: () => {},
    shortcut: "⌘P",
  },
  {
    label: "Settings",
    icon: <Settings className="h-4 w-4" aria-hidden="true" />,
    onClick: () => {},
    shortcut: "⌘,",
  },
  {
    label: "Notifications",
    icon: <Bell className="h-4 w-4" aria-hidden="true" />,
    onClick: () => {},
  },
  {
    label: "Help & Docs",
    icon: <HelpCircle className="h-4 w-4" aria-hidden="true" />,
    onClick: () => {},
    shortcut: "⌘?",
  },
  {
    label: "Sign out",
    icon: <LogOut className="h-4 w-4" aria-hidden="true" />,
    onClick: () => {},
    variant: "destructive",
  },
];

export function ProfileDropdown({
  user,
  items = DEFAULT_ITEMS,
  className,
  trigger,
  onSignOut,
  onSettings,
}: ProfileDropdownProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const listRef = React.useRef<HTMLDivElement>(null);

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
  };

  const defaultTrigger = (
    <button
      ref={triggerRef}
      type="button"
      onClick={() => setIsOpen((prev) => !prev)}
      onKeyDown={handleKeyDown}
      className={cn(
        "flex items-center gap-3 px-3 py-2 rounded-[10px]",
        "bg-[var(--glass-bg)] backdrop-blur-xl border",
        "border-[var(--glass-border)] hover:border-[var(--border-active)]",
        "transition-all duration-200",
        "focus:outline-none focus:ring-2 focus:ring-[var(--sage)]/50 focus:border-[var(--sage)]"
      )}
      aria-haspopup="menu"
      aria-expanded={isOpen}
      aria-label="User menu"
    >
      <div
        className={cn(
          "rounded-full border-2 border-[var(--background)] overflow-hidden",
          "bg-[var(--muted)] h-8 w-8 flex items-center justify-center font-medium"
        )}
        style={{
          color: user.avatar ? undefined : "#90CAF9",
          backgroundColor: user.avatar ? "transparent" : "rgba(144, 202, 249, 0.15)",
        }}
      >
        {user.avatar ? (
          <img src={user.avatar} alt={user.name} className="h-full w-full object-cover" />
        ) : (
          user.initials || user.name.slice(0, 2).toUpperCase()
        )}
      </div>
      <div className="hidden sm:block text-left min-w-0">
        <span className="font-medium text-sm text-[var(--foreground)] truncate block">
          {user.name}
        </span>
        <span className="text-xs text-[var(--moss)] truncate block">
          {user.email}
        </span>
      </div>
      <motion.span
        initial={false}
        animate={{ rotate: isOpen ? 180 : 0 }}
        transition={{ duration: 0.2 }}
        className="text-[var(--moss)] flex-shrink-0"
      >
        <ChevronRight className="h-4 w-4" aria-hidden="true" />
      </motion.span>
    </button>
  );

  return (
    <div className={cn("relative", className)}>
      {trigger ?? defaultTrigger}

      <AnimatePresence mode="wait">
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            className={cn(
              "absolute right-0 top-full z-50 mt-1.5 w-56",
              "bg-[var(--card)] border border-[var(--border)] rounded-[12px]",
              "shadow-xl overflow-hidden backdrop-blur-xl"
            )}
            role="menu"
            ref={listRef}
          >
            <div className="px-3 py-3 border-b border-[var(--border)]">
              <div className="flex items-center gap-3">
                <div
                  className={cn(
                    "rounded-full border-2 border-[var(--background)] overflow-hidden",
                    "bg-[var(--muted)] h-10 w-10 flex items-center justify-center font-medium"
                  )}
                  style={{
                    color: user.avatar ? undefined : "#90CAF9",
                    backgroundColor: user.avatar ? "transparent" : "rgba(144, 202, 249, 0.15)",
                  }}
                >
                  {user.avatar ? (
                    <img src={user.avatar} alt={user.name} className="h-full w-full object-cover" />
                  ) : (
                    user.initials || user.name.slice(0, 2).toUpperCase()
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-sm text-[var(--foreground)] truncate">
                    {user.name}
                  </p>
                  <p className="text-xs text-[var(--moss)] truncate">{user.email}</p>
                </div>
              </div>
            </div>

            <ul className="py-1" role="menu">
              {items.map((item, index) => (
                <motion.li
                  key={item.label}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  transition={{ delay: index * 0.03 }}
                >
                  <button
                    onClick={() => {
                      item.onClick();
                      if (item.variant === "destructive") onSignOut?.();
                      if (item.label === "Settings") onSettings?.();
                      setIsOpen(false);
                    }}
                    role="menuitem"
                    className={cn(
                      "w-full px-3 py-2.5 flex items-center gap-3 text-left",
                      "hover:bg-[var(--muted)] transition-colors",
                      item.variant === "destructive"
                        ? "text-[var(--error)]"
                        : "text-[var(--foreground)]"
                    )}
                  >
                    {item.icon && (
                      <span className={cn("flex-shrink-0", item.variant === "destructive" ? "text-[var(--error)]" : "text-[var(--moss)]")}>
                        {item.icon}
                      </span>
                    )}
                    <span className="flex-1 font-medium text-sm">{item.label}</span>
                    {item.shortcut && (
                      <span className="text-xs text-[var(--moss)] font-mono flex-shrink-0">
                        {item.shortcut}
                      </span>
                    )}
                  </button>
                </motion.li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}