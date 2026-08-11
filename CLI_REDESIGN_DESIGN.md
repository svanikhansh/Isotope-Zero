# isotope_zero CLI Redesign — Design Document

## Vision

Create a CLI that feels like **antigravity** (minimalist, delightful, zero-friction) but has the **power of Claude Code** (rich TUI, live dashboard, keyboard navigation, contextual help). Color palette: **black (#000), beige (#F5F0E1), white (#FFF)**.

## Current State Audit

### Existing Commands (must preserve 100%):
| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `add` | Remember a fact | `--evidence`, `--tags`, `--scope`, `--id`, `--json` |
| `recall` | Semantic retrieval | `--k`, `--alpha`, `--json`, `--verbose` |
| `search` | Hybrid search (vector+BM25+graph) | `--k`, `--alpha`, `--json`, `--verbose` |
| `list` | All cards, newest-first | `--tags`, `--limit`, `--json`, `--verbose` |
| `get` | Full card detail | `--json`, `--verbose` |
| `forget` | Delete card | `--yes` |
| `touch` | Bump access tracking | - |
| `tags` | Tag distribution | `--json`, `--verbose` |
| `stats` | Store overview | `--json`, `--verbose` |
| `inspect` | Diagnostic store report | `--top`, `--json` |
| `dry-run-consolidation` | Preview consolidation | `--limit` |
| `dashboard` | Live TUI overview | `--interval`, `--once` |
| `hook` | Claude Code lifecycle hook | `event`, `--db` |
| `plugin install` | Install editor plugin | `--target`, `--editor`, `--dry-run` |
| *bare `izero`* | Interactive menu | - |

### Current Architecture:
- **Entry point**: `src/isotope_zero/cli/debug.py` — single 1277-line file with argparse
- **Render**: `src/isotope_zero/cli/render.py` — pure formatters (no I/O)
- **Dashboard**: `src/isotope_zero/cli/dashboard.py` — live TUI with rich/plain transports
- **Menu**: `src/isotope_zero/cli/menu.py` — interactive onboarding menu
- **Format utils**: `src/isotope_zero/cli/_fmtutil.py` — stdlib helpers
- **NPM launcher**: `npm/bin/izero.js` — delegates to Python CLI

### Libraries in Use:
- **argparse** — CLI parsing (stdlib)
- **rich** — optional extra for TUI (dashboard, menu)
- **textual** — NOT used currently
- **click/typer** — NOT used

## Design Principles

### 1. antigravity Minimalism
- **Zero-friction first run**: `izero` → beautiful welcome, not usage error
- **Hero content first**: Facts are the star, not IDs or metadata
- **Delightful details**: Easter eggs, subtle animations, thoughtful defaults
- **Graceful degradation**: Works beautifully without optional deps

### 2. Claude Code Power
- **Rich TUI when available**: Arrow navigation, live refresh, alternate screen
- **Keyboard-first**: Vim-style keys, command palette, contextual help
- **Live data**: Dashboard auto-refreshes, vitality bars animate
- **Composable**: Every command works as pipeable `--json` or human-readable

### 3. Black/Beige/White Aesthetic
```
Background:  #000000 (black)
Surface:     #1A1A1A (elevated black)
Primary:     #F5F0E1 (warm beige)
Secondary:   #FFFFFF (white)
Muted:       #8B7D6B (muted beige)
Accent:      #D4C4A8 (light beige)
Border:      #333333 (subtle border)
Success:     #A8D4A8 (soft green-beige)
Warning:     #E8D4A8 (warm amber)
Error:       #E8A8A8 (soft red-beige)
```

## New Architecture

### File Structure
```
src/isotope_zero/cli/
├── __init__.py              # Public exports
├── main.py                  # New entry point (typer-based)
├── commands/
│   ├── __init__.py
│   ├── memory.py           # add, recall, search, list, get, forget, touch
│   ├── inspect.py          # inspect, dry-run-consolidation, stats, tags
│   ├── dashboard.py        # Live dashboard (enhanced)
│   ├── hook.py             # hook command
│   ├── plugin.py           # plugin install
│   └── menu.py             # Interactive menu (enhanced)
├── ui/
│   ├── __init__.py
│   ├── theme.py            # Black/beige/white theme definition
│   ├── components.py       # Reusable UI components
│   ├── layout.py           # Layout helpers (boxes, panels, bars)
│   ├── animations.py       # Subtle animations
│   └── glyphs.py           # Unicode/ASCII glyph system
├── render/
│   ├── __init__.py
│   ├── base.py             # Base renderer protocol
│   ├── json_renderer.py    # --json output
│   ├── plain_renderer.py   # Human-readable plain text
│   └── rich_renderer.py    # Rich TUI renderer
├── core/
│   ├── __init__.py
│   ├── context.py          # Shared CLI context (db, client, config)
│   ├── completers.py       # Tab completion
│   └── hooks.py            # Pre/post command hooks
└── _fmtutil.py             # Keep existing (stdlib only)
```

### Library Choices
| Purpose | Library | Rationale |
|---------|---------|-----------|
| CLI Framework | **typer** | Modern, type-safe, rich integration, auto-completion |
| TUI/Rich Output | **rich** | Already used, excellent for tables, panels, live |
| Advanced TUI | **textual** | For future interactive modes (command palette, etc.) |
| Colors | **rich.color** | Built-in, supports our custom palette |
| Prompts | **rich.prompt** | Beautiful prompts with validation |

### Color Theme Implementation (rich)

```python
# ui/theme.py
from rich.theme import Theme

IZERO_THEME = Theme({
    # Base
    "izero.black": "#000000",
    "izero.beige": "#F5F0E1",
    "izero.white": "#FFFFFF",
    "izero.surface": "#1A1A1A",
    "izero.muted": "#8B7D6B",
    "izero.accent": "#D4C4A8",
    "izero.border": "#333333",
    
    # Semantic
    "izero.success": "#A8D4A8",
    "izero.warning": "#E8D4A8",
    "izero.error": "#E8A8A8",
    "izero.info": "#A8C8E8",
    
    # UI Elements
    "izero.panel": "on #1A1A1A #F5F0E1",
    "izero.panel.border": "#333333",
    "izero.title": "bold #F5F0E1 on #000000",
    "izero.subtitle": "#D4C4A8 on #000000",
    "izero.hero": "bold #FFFFFF on #000000",
    "izero.dim": "#8B7D6B on #000000",
    "izero.bar.fresh": "#A8D4A8",
    "izero.bar.aging": "#E8D4A8",
    "izero.bar.decay": "#E8A8A8",
    "izero.glyph.ok": "bold #A8D4A8",
    "izero.glyph.bullet": "#D4C4A8",
    "izero.glyph.arrow": "#A8C8E8",
    "izero.number": "bold #F5F0E1",
    "izero.id": "#8B7D6B",
    "izero.tag": "#A8C8E8",
    "izero.timestamp": "#8B7D6B",
})
```

## Command UX Redesign

### Global Flags (all commands)
```
--db PATH           Database path (default: ~/.isotope_zero/isotope_zero.db or :memory:)
--json              Machine-readable JSON output
--verbose / -v      Technical detail (ids, scores, timestamps)
--no-color          Disable colors (for pipes/CI)
--help / -h         Show help
--version / -V      Show version
```

### `izero add` — Remember a Fact
```
$ izero add "The user prefers dark mode for coding"
✓ remembered  ·  a1b2c3

$ izero add "The user prefers dark mode" --evidence "settings: dark theme" --tags ui,preference --json
{
  "id": "a1b2c3d4e5f6",
  "fact": "The user prefers dark mode",
  "evidence": "settings: dark theme",
  "tags": ["ui", "preference"],
  "scope": "default",
  "created": true
}
```

### `izero recall` / `izero search` — Find Memories
```
$ izero recall "dark mode"
1.  The user prefers dark mode for coding  ·  a1b2c3
2.  Dark theme reduces eye strain at night  ·  d4e5f6

$ izero search "dark mode" --k 10 --verbose
  #  id          fact                                        score   age_d
  1  a1b2c3      The user prefers dark mode for coding      0.9234   2.1
  2  d4e5f6      Dark theme reduces eye strain at night     0.8761   5.3
```

### `izero list` — Browse All
```
$ izero list
1.  The user prefers dark mode for coding  ·  ui, preference
2.  Dark theme reduces eye strain at night  ·  ui
3.  Python is the favorite language         ·  a1b2c3
```

### `izero get` — Deep Dive
```
$ izero get a1b2c3
The user prefers dark mode for coding
evidence: settings: dark theme enabled
tags:     ui, preference
id:       a1b2c3d4e5f67890
```

### `izero stats` — Overview
```
$ izero stats
12 memories  ·  4.2 KB  ·  fallback embeddings  ·  ~280 tokens

$ izero stats --verbose
12 memories  ·  4.2 KB  ·  fallback embeddings  ·  ~280 tokens

tag distribution:
  ui              5
  preference      3
  debug           2

vitality histogram:
  fresh (>=0.66):     7
  aging (0.33-0.66):  3
  decayed (<0.33):    2
```

### `izero dashboard` — Live TUI
```
┌ isotope_zero · ~/.isotope_zero/isotope_zero.db ────────────────────────┐
│ cards: 12        size: 4.2 KB      mode: FALLBACK                     │
│ vitality: ████████▒▒▒▒░░  fresh 7 / aging 3 / decay 2                │
│ tokens: 280 total  ·  ~45 reclaimable                                │
│ recent:                                                               │
│   · The user prefers dark mode for coding        (0.1d)              │
│   · Dark theme reduces eye strain at night       (0.5d)              │
│ decay candidates (2):                                                 │
│   · Ancient stale fact never recalled      v=0.12 age=400d           │
│   · Old debug note                         v=0.08 age=350d           │
│ refresh 2s  ·  [Ctrl-C to exit]  ·  [?] help                          │
└──────────────────────────────────────────────────────────────────────┘
```

### `izero` (bare) — Interactive Menu
```
┌ isotope_zero · welcome back ────────────────────────────────────────┐
│ Your memories live at: ~/.isotope_zero/isotope_zero.db               │
│ 12 cards remembered so far.                                          │
│ What would you like to do?                                           │
└──────────────────────────────────────────────────────────────────────┘

  ❯ add a memory
    recall something
    search the store
    list memories
    get a memory
    forget a memory
    touch (refresh) a memory
    see stats
    see tags
    inspect the store
    dry-run consolidation
    open the live dashboard
    exit

↑↓ navigate · enter to run · q to quit · / to search commands
```

### Easter Eggs (antigravity spirit)
- `izero --antigravity` → opens XKCD 353 in browser
- `izero --zen` → prints a random programming koan
- `izero --version` with `--verbose` → shows haiku about memory

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create `ui/theme.py` with black/beige/white palette
- [ ] Create `ui/components.py` with reusable components (Panel, Bar, Glyph, Table)
- [ ] Create `render/` protocol with base, json, plain, rich renderers
- [ ] Create `core/context.py` for shared CLI state
- [ ] Migrate `_fmtutil.py` helpers to new structure

### Phase 2: Core Commands (Week 2)
- [ ] Implement `commands/memory.py` (add, recall, search, list, get, forget, touch)
- [ ] Implement `commands/inspect.py` (inspect, dry-run, stats, tags)
- [ ] Wire up typer app in `main.py` with global flags
- [ ] Preserve all existing `--json` contracts exactly

### Phase 3: Advanced UX (Week 3)
- [ ] Enhance `dashboard.py` with new theme, vitality bar animations
- [ ] Enhance `menu.py` with rich arrow navigation, command search (/)
- [ ] Add `hook.py` and `plugin.py` with new renderers
- [ ] Add tab completion for tags, scopes, card IDs

### Phase 4: Polish (Week 4)
- [ ] Easter eggs (--antigravity, --zen)
- [ ] Comprehensive test coverage (all existing tests must pass)
- [ ] NPM launcher update (ensure version lockstep)
- [ ] Documentation update (README, CLI docs)
- [ ] Performance verification (cold start < 50ms)

## Backward Compatibility Checklist

- [ ] All existing subcommands work with identical argv
- [ ] `--json` output is byte-for-byte identical for scripting
- [ ] Exit codes unchanged (0=success, 1=error, 2=hook block)
- [ ] Default DB path resolution unchanged
- [ ] `izero` bare → interactive menu (not error)
- [ ] NPM launcher `izero` delegates correctly
- [ ] All 378 existing tests pass

## Success Metrics

| Metric | Target |
|--------|--------|
| Cold start (no cache) | < 50ms |
| Cold start (with cache) | < 20ms |
| `izero add` latency | < 10ms |
| `izero recall` p99 @ 10k | < 1ms |
| Test suite | 378 passed / 0 failed |
| Wheel size | < 2MB (pure Python) |
| Optional deps | rich only (textual optional) |