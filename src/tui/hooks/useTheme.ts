import { useState, useCallback, useEffect, useRef } from 'react';
import { useApp } from 'ink';

export function useTheme() {
  const [themeName, setThemeName] = useState<'ocean-blue' | 'forest-green' | 'sunset-orange' | 'monochrome'>('ocean-blue');

  const toggleTheme = useCallback(() => {
    const themes: Array<'ocean-blue' | 'forest-green' | 'sunset-orange' | 'monochrome'> = ['ocean-blue', 'forest-green', 'sunset-orange', 'monochrome'];
    const currentIndex = themes.indexOf(themeName);
    const nextIndex = (currentIndex + 1) % themes.length;
    setThemeName(themes[nextIndex]);
  }, [themeName]);

  return { themeName, setThemeName, toggleTheme };
}
