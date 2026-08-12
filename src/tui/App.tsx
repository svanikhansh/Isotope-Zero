import React, { useState, useMemo, useEffect } from 'react';
import { render, useApp } from 'ink';
import { TextInput } from '@inkjs/ui';
import {
  Box,
  Text,
  Card,
  Header,
  Footer,
  Sidebar,
  StatTile,
  Table,
  ProgressBar,
  Modal,
} from './components';
import { useKeyboardShortcuts, useListNavigation } from './hooks/useKeyboard';
import { useTheme } from './hooks/useTheme';
import { getTheme } from './theme';

// --- Model Registry Data ---
const MODEL_CATALOG = [
  { id: 'navy/schizogpt', name: 'SchizoGPT', type: 'Experimental', tags: ['fast', 'unfiltered'], free: true },
  { id: 'navy/gpt-5.6-sol', name: 'GPT-5.6 Sol', type: 'Production', tags: ['reasoning', 'large'], free: false },
  { id: 'navy/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', type: 'Production', tags: ['coding', 'nuanced'], free: false },
  { id: 'navy/llama-3-70b', name: 'Llama 3 70B', type: 'Open', tags: ['general', 'local'], free: true },
  { id: 'navy/mistral-large', name: 'Mistral Large', type: 'Production', tags: ['multilingual'], free: false },
  { id: 'navy/phi-3-mini', name: 'Phi-3 Mini', type: 'Small', tags: ['edge', 'efficient'], free: true },
  { id: 'navy/deepseek-coder', name: 'DeepSeek Coder', type: 'Coding', tags: ['python', 'rust'], free: true },
];

interface AppState {
  view: 'dashboard' | 'memories' | 'model_selector';
  selectedModelIndex: number;
  modelFilter: 'All' | 'Visible' | 'Hidden' | 'Free';
  modelSearchQuery: string;
  themeName: string;
}

const ModelSelector = ({
  onSelect,
  currentTheme
}: {
  onSelect: (id: string) => void,
  currentTheme: string
}) => {
  const { exit } = useApp();
  const [searchQuery, setSearchQuery] = useState('');
  const [filterIndex, setFilterIndex] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const filters: AppState['modelFilter'][] = ['All', 'Visible', 'Hidden', 'Free'];

  const filteredModels = useMemo(() => {
    return MODEL_CATALOG.filter(m => {
      const matchesSearch = m.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
                            m.name.toLowerCase().includes(searchQuery.toLowerCase());
      const filter = filters[filterIndex];
      if (filter === 'Free') return matchesSearch && m.free;
      if (filter === 'Visible') return matchesSearch && m.type === 'Production';
      if (filter === 'Hidden') return matchesSearch && m.type !== 'Production';
      return matchesSearch;
    });
  }, [searchQuery, filterIndex]);

  const nav = useListNavigation(filteredModels, {
    selectedIndex,
    onSelect: setSelectedIndex,
    onActivate: (index) => onSelect(filteredModels[index].id),
  });

  const shortcuts = useMemo(() => ({
    global: {
      'Escape': () => exit(),
      'Tab': () => setFilterIndex(prev => (prev + 1) % filters.length),
    }
  }), [exit]);

  useKeyboardShortcuts({ global: shortcuts.global });

  return (
    <Box
      flexDirection="column"
      width="100%"
      height="100%"
      backgroundColor="#0f172a"
      padding={2}
    >
      {/* Metadata Row */}
      <Box
        flexDirection="row"
        justifyContent="space-between"
        padding={1}
        backgroundColor="#1e293b"
        borderStyle="round"
        borderColor="muted"
      >
        <Text variant="muted" fontSize="small">
          Filter models... {filteredModels.length}/{MODEL_CATALOG.length} active
        </Text>
        <Box flexDirection="row" gap={2}>
          {filters.map((f, i) => (
            <Text
              key={f}
              variant={i === filterIndex ? 'primary' : 'subtle'}
              weight={i === filterIndex ? 'bold' : 'normal'}
            >
              {i === filterIndex ? `[ ${f} ]` : f}
            </Text>
          ))}
        </Box>
      </Box>

      {/* Prompt Area */}
      <Box flexDirection="row" alignItems="center" marginTop={2} marginBottom={2}>
        <Text variant="primary" weight="bold" marginRight={1}>/model </Text>
        <TextInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search registry..."
        />
        <Text variant="primary" weight="bold">█</Text>
      </Box>

      {/* Model List */}
      <Box
        flexDirection="column"
        flex={1}
        borderStyle="round"
        borderColor="border"
        padding={1}
        backgroundColor="#1e293b"
      >
        {filteredModels.map((model, i) => (
          <Box
            key={model.id}
            flexDirection="row"
            justifyContent="space-between"
            padding={1}
            backgroundColor={i === selectedIndex ? '#1e3a8a' : 'transparent'}
          >
            <Box flexDirection="row">
              <Text variant={i === selectedIndex ? 'bright' : 'muted'} weight={i === selectedIndex ? 'bold' : 'normal'}>
                {i === selectedIndex ? '› ' : '  '} {model.id}
              </Text>
            </Box>
            <Text variant="subtle" italic fontSize="small">
              (Click to set alias)
            </Text>
          </Box>
        ))}
      </Box>
    </Box>
  );
};

const App = () => {
  const { themeName, toggleTheme } = useTheme();
  const [state, setState] = useState<AppState>({
    view: 'dashboard',
    selectedModelIndex: 0,
    modelFilter: 'All',
    modelSearchQuery: '',
    themeName,
  });

  const shortcuts = useMemo(() => ({
    global: {
      '/': () => setState(s => ({ ...s, view: 'model_selector' })),
      'Escape': () => setState(s => ({ ...s, view: 'dashboard' })),
      't': toggleTheme,
    }
  }), [toggleTheme]);

  useKeyboardShortcuts({ global: shortcuts.global });

  if (state.view === 'model_selector') {
    return <ModelSelector onSelect={(id) => console.log('Selected model:', id)} currentTheme={themeName} />;
  }

  return (
    <Box flexDirection="column" width="100%" height="100%">
      <Header title="isotope-zero" subtitle="Cognitive Memory Layer" version="1.5.0" connected={true} />
      <Box flexDirection="row" flex={1} overflow="hidden">
        <Sidebar
          items={[
            { id: 'dashboard', label: 'Dashboard', icon: <Text>📊</Text> },
            { id: 'memories', label: 'Memories', icon: <Text>🧠</Text> },
          ]}
          activeId={state.view}
          onSelect={(id) => setState(s => ({ ...s, view: id as any }))}
        />
        <Box flex={1} padding={2}>
          {state.view === 'dashboard' ? (
            <Box flexDirection="column" gap={2}>
              <Text variant="bright" weight="bold" fontSize="large">System Dashboard</Text>
              <Box flexDirection="row" gap={2}>
                <StatTile label="Memories" value="124" icon={<Text>🧠</Text>} />
                <StatTile label="Storage" value="42MB" icon={<Text>💾</Text>} />
              </Box>
            </Box>
          ) : (
            <Text variant="muted">Select a view from the sidebar.</Text>
          )}
        </Box>
      </Box>
      <Footer hints={[{ key: '/', action: 'Model Selector' }, { key: 'T', action: 'Theme' }]} />
    </Box>
  );
};

export async function runTUI() {
  render(<App />);
}

if (import.meta.main) {
  await runTUI();
}

export default App;
