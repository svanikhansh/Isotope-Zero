import { useState, useCallback, useEffect, useRef } from 'react';

export interface UseKeyboardShortcutsOptions {
  global?: Record<string, () => void>;
  views?: Record<string, Record<string, () => void>>;
  currentView?: string;
}

export function useKeyboardShortcuts({
  global = {},
  views = {},
  currentView = 'default',
}: UseKeyboardShortcutsOptions) {
  const shortcutsRef = useRef({ global, views, currentView });
  shortcutsRef.current = { global, views, currentView };

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const parts: string[] = [];
      if (event.ctrlKey) parts.push('Ctrl');
      if (event.metaKey) parts.push('Meta');
      if (event.altKey) parts.push('Alt');
      if (event.shiftKey) parts.push('Shift');

      const key = event.key;
      const keyMap: Record<string, string> = {
        Escape: 'Escape',
        Enter: 'Enter',
        Tab: 'Tab',
        Backspace: 'Backspace',
        Delete: 'Delete',
        ArrowUp: 'Up',
        ArrowDown: 'Down',
        ArrowLeft: 'Left',
        ArrowRight: 'Right',
        Home: 'Home',
        End: 'End',
        PageUp: 'PageUp',
        PageDown: 'PageDown',
        ' ': 'Space',
      };
      parts.push(keyMap[key] || key);
      const combo = parts.join('+');

      const viewShortcuts = shortcutsRef.current.views[shortcutsRef.current.currentView] || {};
      if (viewShortcuts[combo]) {
        event.preventDefault();
        viewShortcuts[combo]();
        return;
      }

      if (shortcutsRef.current.global[combo]) {
        event.preventDefault();
        shortcutsRef.current.global[combo]();
        return;
      }
    },
    []
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return { handleKeyDown };
}

export function useListNavigation<T>(
  items: T[],
  options: {
    selectedIndex: number;
    onSelect: (index: number) => void;
    onActivate?: (index: number) => void;
    loop?: boolean;
    disabledIndices?: Set<number>;
  }
) {
  const { selectedIndex, onSelect, onActivate, loop = true, disabledIndices = new Set() } = options;

  const getNextEnabled = useCallback(
    (fromIndex: number, direction: 1 | -1) => {
      let nextIndex = fromIndex;
      const maxIterations = items.length;
      let iterations = 0;

      do {
        nextIndex += direction;
        if (loop) {
          if (nextIndex >= items.length) nextIndex = 0;
          if (nextIndex < 0) nextIndex = items.length - 1;
        } else {
          if (nextIndex >= items.length) nextIndex = items.length - 1;
          if (nextIndex < 0) nextIndex = 0;
        }
        iterations++;
      } while (disabledIndices.has(nextIndex) && iterations < maxIterations);

      return disabledIndices.has(nextIndex) ? fromIndex : nextIndex;
    },
    [items.length, loop, disabledIndices]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      switch (event.key) {
        case 'ArrowDown':
        case 'j':
          event.preventDefault();
          onSelect(getNextEnabled(selectedIndex, 1));
          break;
        case 'ArrowUp':
        case 'k':
          event.preventDefault();
          onSelect(getNextEnabled(selectedIndex, -1));
          break;
        case 'Home':
        case 'g':
          event.preventDefault();
          onSelect(getNextEnabled(-1, 1));
          break;
        case 'End':
        case 'G':
          event.preventDefault();
          onSelect(getNextEnabled(items.length, -1));
          break;
        case 'Enter':
        case ' ':
          if (onActivate) {
            event.preventDefault();
            onActivate(selectedIndex);
          }
          break;
        case 'PageDown':
        case 'd':
          if (event.ctrlKey) {
            event.preventDefault();
            onSelect(getNextEnabled(selectedIndex + 10, 1));
          }
          break;
        case 'PageUp':
        case 'u':
          if (event.ctrlKey) {
            event.preventDefault();
            onSelect(getNextEnabled(selectedIndex - 10, -1));
          }
          break;
      }
    },
    [selectedIndex, onSelect, onActivate, getNextEnabled, items.length]
  );

  return { handleKeyDown };
}

export interface FormField {
  name: string;
  label: string;
  type?: 'text' | 'password' | 'select' | 'confirm';
  required?: boolean;
  placeholder?: string;
  options?: Array<{ value: string; label: string }>;
  validate?: (value: string) => string | null;
}

export interface FormState {
  values: Record<string, string>;
  errors: Record<string, string>;
  touched: Record<string, boolean>;
  isSubmitting: boolean;
  isValid: boolean;
}

export function useForm(
  fields: FormField[],
  onSubmit: (values: Record<string, string>) => void | Promise<void>
) {
  const [state, setState] = useState<FormState>({
    values: {},
    errors: {},
    touched: {},
    isSubmitting: false,
    isValid: false,
  });

  const validateField = useCallback(
    (name: string, value: string) => {
      const field = fields.find((f) => f.name === name);
      if (!field) return null;
      if (field.required && !value.trim()) return `${field.label} is required`;
      if (field.validate) return field.validate(value);
      return null;
    },
    [fields]
  );

  const validateAll = useCallback(() => {
    const errors: Record<string, string> = {};
    let isValid = true;
    for (const field of fields) {
      const value = state.values[field.name] || '';
      const error = validateField(field.name, value);
      if (error) {
        errors[field.name] = error;
        isValid = false;
      }
    }
    return { errors, isValid };
  }, [fields, state.values, validateField]);

  const setValue = useCallback((name: string, value: string) => {
    setState((prev) => {
      const newValues = { ...prev.values, [name]: value };
      const error = validateField(name, value);
      const newErrors = { ...prev.errors, [name]: error || '' };
      const { errors, isValid } = validateAll();
      return {
        ...prev,
        values: newValues,
        errors: newErrors,
        touched: { ...prev.touched, [name]: true },
        isValid,
      };
    });
  }, [validateField, validateAll]);

  const setTouched = useCallback((name: string) => {
    setState((prev) => ({ ...prev, touched: { ...prev.touched, [name]: true } }));
  }, []);

  const handleSubmit = useCallback(async () => {
    const { errors, isValid } = validateAll();
    setState((prev) => ({
      ...prev,
      errors,
      touched: fields.reduce((acc, f) => ({ ...acc, [f.name]: true }), {}),
      isValid,
    }));
    if (isValid) {
      setState((prev) => ({ ...prev, isSubmitting: true }));
      try {
        await onSubmit(state.values);
      } finally {
        setState((prev) => ({ ...prev, isSubmitting: false }));
      }
    }
  }, [validateAll, fields, onSubmit, state.values]);

  const reset = useCallback(() => {
    setState({
      values: {},
      errors: {},
      touched: {},
      isSubmitting: false,
      isValid: false,
    });
  }, []);

  return { state, setValue, setTouched, handleSubmit, reset, validateAll };
}
