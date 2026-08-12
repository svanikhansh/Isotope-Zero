# isotope-zero TUI

A premium Terminal User Interface for isotope-zero, built with Ink, Yoga, and React. Inspired by the aesthetic of Claude Code.

## Features

- **Yoga Flexbox Layout** - Clean, responsive layouts using Yoga's flexbox engine
- **Premium Styling** - Ocean-blue theme (default) with forest-green, sunset-orange, and monochrome variants
- **Keyboard Navigation** - Full vim-like keybindings with global and view-specific shortcuts
- **Component Library** - Reusable components: Box, Text, Card, Table, StatTile, ProgressBar, Modal, Alert, Sidebar, Header, Footer
- **React Hooks** - Custom hooks for keyboard shortcuts, list navigation, forms, and theming
- **TypeScript** - Full type safety throughout

## Quick Start

```bash
# Install dependencies (npm — the package ships pure Node, no Bun needed)
npm install

# Typecheck + build (compile to dist/)
npm run build

# Run the TUI (compiled — needs `npm run build` first)
node bin/izero.js tui

# ...or run straight from source with tsx
npm run dev
```

## Keyboard Shortcuts

### Global
| Key | Action |
|-----|--------|
| `Ctrl+Q` / `Ctrl+C` | Quit |
| `Tab` / `Shift+Tab` | Next/Previous view |
| `1-5` | Jump to view (Dashboard, Memories, Tags, Decay, Settings) |
| `R` | Refresh data |
| `T` | Toggle theme |
| `?` / `H` | Help |

### Navigation
| Key | Action |
|-----|--------|
| `↑` / `↓` / `j` / `k` | Navigate lists |
| `Enter` / `Space` | Select/Activate |
| `Home` / `g` | Jump to top |
| `End` / `G` | Jump to bottom |
| `PgUp` / `Ctrl+U` | Page up |
| `PgDn` / `Ctrl+D` | Page down |

### Search
| Key | Action |
|-----|--------|
| `/` | Focus search |
| `Esc` | Clear search |
| Click tag | Filter by tag |

## Views

1. **Dashboard** - Overview with stats, vitality distribution, quick actions, recent memories
2. **Memories** - Full memory browser with search and tag filtering
3. **Tags** - Tag cloud with filtering
4. **Decay** - Decay candidates and consolidation actions
5. **Settings** - Theme selection, keyboard shortcuts reference
6. **Help** - Complete help documentation

## Themes

- **ocean-blue** (default) - Professional blue theme inspired by Claude Code
- **forest-green** - Nature-inspired green theme
- **sunset-orange** - Warm sunset theme
- **monochrome** - Clean black & white

## Architecture

```
src/
├── index.ts              # Library entry (no side effects)
├── cli.ts                # Node argument dispatcher (dist/cli.js)
├── pybridge.ts           # Hybrid bridge to the Python `izero` CLI
├── tui/
│   ├── App.tsx           # Main TUI application
│   ├── theme/
│   │   ├── colors.ts     # Theme color definitions
│   │   ├── spacing.ts    # Spacing, layout constants, flex styles
│   │   └── index.ts      # Theme exports
│   ├── components/
│   │   ├── Box.tsx       # Flexbox container with variants
│   │   ├── Text.tsx      # Styled text with variants
│   │   ├── Card.tsx      # Card component with badge support
│   │   ├── Header.tsx    # Application header
│   │   ├── Footer.tsx    # Application footer with hints
│   │   ├── Sidebar.tsx   # Navigation sidebar
│   │   ├── StatTile.tsx  # KPI/metric display
│   │   ├── Table.tsx     # Data table with selection
│   │   ├── Progress.tsx  # Progress bars and spinners
│   │   ├── Modal.tsx     # Modal dialogs and alerts
│   │   └── index.ts      # Component exports
│   └── hooks/
│       ├── useKeyboard.ts # Keyboard shortcuts & list navigation
│       └── useTheme.ts    # Theme management
└── bin/
    └── izero.js          # CLI entry point
```

## Development

```bash
# Type checking
npm run typecheck

# Build (compile to dist/)
npm run build

# Smoke (version + help)
npm test
```

## Design Principles

1. **Yoga Flexbox Grid** - All visual segments wrapped in clean `<Box>` elements using dedicated margins, padding, and `borderStyle="round"` for breathing room
2. **High-End Styling** - `chalk.dim` for past step histories and muted gray instructions. Bright neon greens, cyans, or pastels exclusively for active menu pointers (`›`) and selection cues
3. **Clean State Management** - React hooks manage step transitions gracefully, ensuring elements don't flicker or clip text when re-rendering layout rows