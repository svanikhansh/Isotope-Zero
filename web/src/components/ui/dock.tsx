"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

export interface DockItem {
  title: string;
  icon: React.ReactNode;
  href?: string;
  onClick?: () => void;
}

export interface DockProps {
  items: DockItem[];
  className?: string;
  position?: "bottom" | "top" | "left" | "right";
}

interface DockItemProps {
  item: DockItem;
  isActive: boolean;
  onClick: (item: DockItem) => void;
  position: DockProps["position"];
}

const DOCK_ITEM_SIZE = 56;
const MAX_SCALE = 1.6;
const SCALE_RANGE = 80;

function DockIcon({ children, className, ...props }: React.ComponentPropsWithoutRef<"div"> & { children: React.ReactNode }) {
  return (
    <div
      className={cn(
        "flex h-10 w-10 items-center justify-center rounded-lg transition-colors",
        "bg-[var(--glass-bg)] border border-[var(--glass-border)]",
        "hover:bg-[var(--muted)] hover:border-[var(--border-active)]",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

function DockItemComponent({ item, isActive, onClick, position, ref }: DockItemProps & { ref?: React.Ref<HTMLDivElement> }) {
  const [hovered, setHovered] = React.useState(false);
  const isHorizontal = position === "bottom" || position === "top";

  return (
    <motion.div
      ref={ref}
      whileHover={{ scale: isActive ? 1 : 1.1 }}
      whileTap={{ scale: 0.95 }}
      onClick={() => onClick(item)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={cn(
        "relative flex flex-col items-center gap-1.5 cursor-pointer",
        "transition-all duration-200 ease-out",
        isActive && "opacity-100",
        !isActive && "opacity-60 hover:opacity-100"
      )}
      style={{
        transformOrigin: isHorizontal ? "center bottom" : "center right",
      }}
    >
      <AnimatePresence mode="wait">
        {hovered && (
          <motion.span
            initial={{ opacity: 0, y: 4, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.9 }}
            className={cn(
              "absolute bottom-full left-1/2 mb-1.5 px-2 py-1 text-xs font-medium rounded",
              "bg-[var(--card)] text-[var(--card-foreground)] border border-[var(--border)]",
              "whitespace-nowrap shadow-lg",
              "transform -translate-x-1/2",
              isHorizontal ? "" : "left-auto right-full bottom-1/2 mb-0 mr-2 -translate-y-1/2"
            )}
          >
            {item.title}
          </motion.span>
        )}
      </AnimatePresence>

      <DockIcon className={cn(isActive && "ring-2 ring-[var(--ocean-light)]")}>
        {item.icon}
      </DockIcon>

      <AnimatePresence mode="wait">
        {isActive && (
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            className={cn(
              "absolute bottom-full left-1/2 mb-1 w-1.5 h-1.5 rounded-full",
              "bg-[var(--ocean-light)]",
              "transform -translate-x-1/2",
              isHorizontal ? "" : "left-auto right-full bottom-1/2 mb-0 mr-2 -translate-y-1/2"
            )}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export function Dock({ items, className, position = "bottom" }: DockProps) {
  const [activeItem, setActiveItem] = React.useState<DockItem | null>(items[0] || null);
  const containerRef = React.useRef<HTMLDivElement>(null);
  const itemRefs = React.useRef<Array<HTMLDivElement | null>>([]);

  const handleItemClick = (item: DockItem) => {
    setActiveItem(item);
    item.onClick?.();
    if (item.href) window.location.assign(item.href);
  };

  const setMagnification = (mousePos: number | null) => {
    if (!containerRef.current || items.length === 0) return;
    const isHorizontal = position === "bottom" || position === "top";

    items.forEach((_, index) => {
      const node = itemRefs.current[index];
      if (!node) return;
      if (mousePos === null) {
        node.style.transform = "";
        node.style.margin = "";
        return;
      }
      const rect = node.getBoundingClientRect();
      const itemCenter = isHorizontal
        ? rect.left + rect.width / 2
        : rect.top + rect.height / 2;
      const distance = Math.abs(mousePos - itemCenter);
      const scale = distance < SCALE_RANGE ? 1 + (MAX_SCALE - 1) * (1 - distance / SCALE_RANGE) : 1;
      const targetSize = DOCK_ITEM_SIZE * (scale - 1);
      const margin = isHorizontal ? `0 ${targetSize / 2}px` : `${targetSize / 2}px 0`;
      node.style.transform = `scale(${scale})`;
      node.style.margin = margin;
      node.style.transformOrigin = isHorizontal ? "center bottom" : "center right";
    });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const isHorizontal = position === "bottom" || position === "top";
    setMagnification(isHorizontal ? e.clientX : e.clientY);
  };

  const handleMouseLeave = () => setMagnification(null);

  const isHorizontal = position === "bottom" || position === "top";

  return (
    <motion.div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      initial={{ opacity: 0, y: isHorizontal ? 20 : -20, x: isHorizontal ? 0 : 20 }}
      animate={{ opacity: 1, y: 0, x: 0 }}
      exit={{ opacity: 0, y: isHorizontal ? 20 : -20, x: isHorizontal ? 0 : 20 }}
      className={cn(
        "fixed z-50 flex items-center justify-center gap-2 px-4 py-3",
        "bg-[var(--glass-bg)] backdrop-blur-xl border border-[var(--glass-border)]",
        "rounded-[24px] shadow-2xl",
        isHorizontal
          ? "bottom-6 left-1/2 -translate-x-1/2 w-auto max-w-[calc(100%-2rem)]"
          : "right-6 top-1/2 -translate-y-1/2 h-auto max-h-[calc(100%-2rem)]",
        className
      )}
      role="navigation"
      aria-label="Dock"
    >
      {items.map((item, index) => (
        <DockItemComponent
          key={item.title}
          ref={(el) => { itemRefs.current[index] = el; }}
          item={item}
          isActive={activeItem === item}
          onClick={handleItemClick}
          position={position}
        />
      ))}
    </motion.div>
  );
}

export function DockLabel({ children, className, ...props }: React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        "text-xs font-medium text-[var(--muted-foreground)]",
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}