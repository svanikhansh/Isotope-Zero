"use client";

import { useState, useCallback, useMemo } from "react";

export interface Tag {
  id: string;
  label: string;
  color?: string;
}

export interface UseTagsOptions {
  maxTags?: number;
  onChange?: (tags: Tag[]) => void;
  initialTags?: Tag[];
}

export interface UseTagsReturn {
  tags: Tag[];
  addTag: (tag: Omit<Tag, "id">) => void;
  removeTag: (id: string) => void;
  removeLastTag: () => void;
  hasReachedMax: boolean;
  setTags: React.Dispatch<React.SetStateAction<Tag[]>>;
}

const TAG_COLORS = [
  "#90CAF9", // ocean light
  "#4FC3F7", // sky blue
  "#29B6F6", // cyan-blue
  "#26C6DA", // teal-blue
  "#5C6BC0", // indigo
  "#7E57C2", // violet
  "#42A5F5", // material blue
  "#1565C0", // ocean dark
];

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function getNextColor(existingTags: Tag[]): string {
  const usedColors = new Set(existingTags.map((t) => t.color).filter(Boolean));
  for (const color of TAG_COLORS) {
    if (!usedColors.has(color)) return color;
  }
  return TAG_COLORS[existingTags.length % TAG_COLORS.length];
}

export function useTags({
  maxTags = 10,
  onChange,
  initialTags = [],
}: UseTagsOptions = {}): UseTagsReturn {
  const [tags, setTags] = useState<Tag[]>(initialTags);

  const hasReachedMax = useMemo(() => tags.length >= maxTags, [tags, maxTags]);

  const notifyChange = useCallback(
    (newTags: Tag[]) => {
      onChange?.(newTags);
    },
    [onChange]
  );

  const addTag = useCallback(
    (tag: Omit<Tag, "id">) => {
      setTags((prev) => {
        if (prev.length >= maxTags) return prev;
        if (prev.some((t) => t.label.toLowerCase() === tag.label.toLowerCase())) {
          return prev;
        }
        const newTag: Tag = {
          ...tag,
          id: generateId(),
          color: tag.color || getNextColor(prev),
        };
        const next = [...prev, newTag];
        notifyChange(next);
        return next;
      });
    },
    [maxTags, notifyChange]
  );

  const removeTag = useCallback(
    (id: string) => {
      setTags((prev) => {
        const next = prev.filter((t) => t.id !== id);
        notifyChange(next);
        return next;
      });
    },
    [notifyChange]
  );

  const removeLastTag = useCallback(() => {
    setTags((prev) => {
      if (prev.length === 0) return prev;
      const next = prev.slice(0, -1);
      notifyChange(next);
      return next;
    });
  }, [notifyChange]);

  return {
    tags,
    addTag,
    removeTag,
    removeLastTag,
    hasReachedMax,
    setTags,
  };
}