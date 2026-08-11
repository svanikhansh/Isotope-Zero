# isotope_zero CLI Architecture Design
## Antigravity Minimalism + Claude Code Power

---

## 1. Design Philosophy

### Core Aesthetic: "Antigravity Minimalism"
- **Invisible until needed** — no chrome, no banners, no noise
- **Weightless interactions** — every keystroke has purpose
- **Deep power on demand** — Claude Code-level sophistication beneath the surface

### Visual Language: Black / Beige / White
| Role | Hex | Usage |
|------|-----|-------|
| **Void** | `#000000` | Background, deep space |
| **Parchment** | `#F5F0E1` | Primary text, content, warmth |
| **Lumina** | `#FFFFFF` | Hero text, highlights, focus |
| **Muted** | `#8B7D6B` | Secondary, timestamps, IDs |
| **Accent** | `#D4C4A8` | Interactive, borders, glyphs |
| **Success** | `#A8D4A8` | Fresh vitality, confirmations |
| **Warning** | `#E8D4A8` | Aging vitality, cautions |
| **Error** | `#E8A8A8` | Decayed vitality, failures |
| **Info** | `#A8C8E8` | Links, tags, semantic hints |

> **Note**: These are semantic aliases. The actual theme uses `izero.black`, `izero.beige`, `izero.white`, etc. — enabling instant palette swaps.

---

## 2. File Structure

```
src/isotope_zero/cli/
├── main.py                    # Typer entrypoint, global callbacks
├── core/
│   ├── __init__.py
│   ├── context.py             # CLIContext, option factories, injection
│   ├── config.py              # Config loading (toml/env/flags)
│   ├── lifecycle.py           # Startup/shutdown, DB connection
│   └── completers.py          # Tab completion for tags, scopes, IDs
├── render/
│   ├── __init__.py            # Renderer protocol + factory
│   ├── base.py                # BaseRenderer abstract class
│   ├── components.py          # MemoryRow, StatRow, TagCount, VitalityHistogram
│   ├── json_renderer.py       # Machine contract (unchanged)
│   ├── plain_renderer.py      # Stdlib-only, ANSI colors, ASCII glyphs
│   └── rich_renderer.py       # Full TUI: panels, tables, live, animations
├── ui/
│   ├── __init__.py
│   ├── theme.py               # IZERO_THEME, palette, Console factory
│   ├── glyphs.py              # Unicode/ASCII glyph system
│   └── components.py          # High-level: boxes, bars, menus, dashboards
├── commands/
│   ├── __init__.py            # register_commands()
│   ├── memory/
│   │   ├── __init__.py        # add, get, list, forget, touch
│   │   ├── add.py
│   │   ├── get.py
│   │   ├── list.py
│   │   ├── forget.py
│   │   └── touch.py
│   ├── query/
│   │   ├── __init__.py        # recall, search
│   │   ├── recall.py
│   │   └── search.py
│   ├── insight/
│   │   ├── __init__.py        # stats, tags, inspect, active, dry-run
│   │   ├── stats.py
│   │   ├── tags.py
│   │   ├── inspect.py
│   │   ├── active.py
│   │   └── dry_run.py
│   ├── ui/
│   │   ├── __init__.py        # dashboard, menu
│   │   ├── dashboard.py
│   │   └── menu.py
│   ├── integrations/
│   │   ├── __init__.py        # plugin, mcp, hook
│   │   ├── plugin.py
│   │   ├── mcp.py
│   │   └── hook.py
│   └── dev/
│       ├── __init__.py        # benchmark, debug
│       ├── benchmark.py
│       └── debug.py
└── integrations/              # Editor plugin core (testable, not CLI)
    ├── __init__.py
    ├── install.py
    ├── uninstall.py
    └── status.py
```

---

## 3. Component Design

### 3.1 Renderer Protocol (Unchanged Contract)

```python
# render/base.py
@runtime_checkable
class Renderer(Protocol):
    # Memory
    def render_memory_rows(self, rows: Sequence[MemoryRow], show_score=False, show_vitality=False, show_tags=True) -> str: ...
    def render_memory_card(self, card: MemoryRow, verbose=False) -> str: ...
    
    # Insight
    def render_stats(self, count, size_bytes, embedding_mode, tokens, tag_dist=(), histogram=None, verbose=False) -> str: ...
    def render_tags(self, tag_counts: Sequence[TagCount], verbose=False) -> str: ...
    def render_inspect(self, db_path, total, size_bytes, embedding_mode, avg_dim, cards_with_emb, decay_rows, tokens) -> str: ...
    def render_dry_run(self, plan: dict, limit=0) -> str: ...
    
    # UI
    def render_dashboard(self, state: dict, interval: float) -> str: ...
    def render_menu(self, banner: str, entries: list[tuple[str, Any]], selected_idx: int) -> str: ...
    
    # Actions
    def render_add_result(self, card_id: str, created: bool) -> str: ...
    def render_forget_result(self, card_id: str) -> str: ...
    def render_touch_result(self, card_id: str) -> str: ...
    
    # JSON passthrough
    def render_json(self, obj: Any) -> str: ...
```

**Three implementations, one contract:**
- `JsonRenderer` → `json.dumps(obj, indent=2)` — stable machine interface
- `PlainRenderer` → ANSI-colored text, ASCII glyphs, zero deps
- `RichRenderer` → Full TUI with panels, tables, live updates, animations

### 3.2 Theme System (`ui/theme.py`)

```python
# Semantic color aliases (not hardcoded hex)
IZERO_THEME = Theme({
    # Base
    "izero.void":       "#000000",
    "izero.parchment":  "#F5F0E1",
    "izero.lumina":     "#FFFFFF",
    "izero.muted":      "#8B7D6B",
    "izero.accent":     "#D4C4A8",
    "izero.border":     "#333333",
    
    # Semantic
    "izero.success":    "#A8D4A8",
    "izero.warning":    "#E8D4A8",
    "izero.error":      "#E8A8A8",
    "izero.info":       "#A8C8E8",
    
    # Composed styles
    "izero.hero":       "bold izero.lumina on izero.void",
    "izero.title":      "bold izero.parchment on izero.void",
    "izero.body":       "izero.parchment on izero.void",
    "izero.dim":        "izero.muted on izero.void",
    "izero.number":     "bold izero.parchment on izero.void",
    "izero.id":         "izero.muted on izero.void",
    "izero.tag":        "izero.info on izero.void",
    "izero.timestamp":  "izero.muted on izero.void",
    "izero.evidence":   "izero.muted on izero.void",
    
    # Glyphs
    "izero.glyph.ok":       "bold izero.success on izero.void",
    "izero.glyph.bullet":   "izero.accent on izero.void",
    "izero.glyph.arrow":    "izero.info on izero.void",
    
    # Vitality bars
    "izero.bar.fresh":  "izero.success on izero.void",
    "izero.bar.aging":  "izero.warning on izero.void",
    "izero.bar.decay":  "izero.error on izero.void",
    "izero.bar.empty":  "izero.border on izero.void",
    
    # Interactive
    "izero.selected":   "bold izero.void on izero.parchment",
    "izero.focused":    "bold izero.lumina on izero.surface",
    "izero.hint":       "izero.muted on izero.void",
    
    # Panels
    "izero.panel":           "on izero.surface izero.parchment",
    "izero.panel.border":    "izero.border",
    "izero.panel.title":     "bold izero.parchment on izero.void",
    "izero.panel.subtitle":  "izero.accent on izero.void",
})
```

### 3.3 Glyph System (`ui/glyphs.py`)

```python
_GLYPHS = {
    # Status
    "ok":         {"unicode": "✓", "ascii": "ok"},
    "cross":      {"unicode": "✗", "ascii": "x"},
    "warning":    {"unicode": "⚠", "ascii": "!"},
    "info":       {"unicode": "ℹ", "ascii": "i"},
    
    # Navigation
    "bullet":     {"unicode": "·", "ascii": "-"},
    "arrow":      {"unicode": "→", "ascii": "->"},
    "chevron":    {"unicode": "›", "ascii": ">"},
    
    # Selection
    "selected":   {"unicode": "❯", "ascii": ">"},
    "unselected": {"unicode": " ", "ascii": " "},
    "radio_on":   {"unicode": "◉", "ascii": "(*)"},
    "radio_off":  {"unicode": "○", "ascii": "( )"},
    
    # Box drawing (light)
    "box_tl": "┌", "box_tr": "┐", "box_bl": "└", "box_br": "┘",
    "box_h":  "─", "box_v":  "│", "box_t":  "┬", "box_b":  "┴",
    "box_l":  "├", "box_r":  "┤", "box_x":  "┼",
    
    # Box drawing (heavy)
    "box_heavy_tl": "┏", "box_heavy_tr": "┓",
    "box_heavy_bl": "┗", "box_heavy_br": "┛",
    "box_heavy_h":  "━", "box_heavy_v":  "┃",
    
    # Vitality bars
    "bar_full":   {"unicode": "█", "ascii": "#"},
    "bar_medium": {"unicode": "▒", "ascii": "="},
    "bar_low":    {"unicode": "░", "ascii": "-"},
    "bar_empty":  {"unicode": " ", "ascii": " "},
}
```

**Auto-detection**: `can_unicode()` checks `stdout.encoding` at import time. No runtime overhead.

### 3.4 Vitality Bar (`ui/glyphs.py::build_vitality_bar`)

```python
def build_vitality_bar(fresh, aging, decayed, width=20, use_rich=False):
    """Fixed-width proportional bar. Returns str (plain) or [(text, style)] (rich)."""
    total = max(1, fresh + aging + decayed)
    f = round(fresh / total * width)
    d = round(decayed / total * width)
    a = width - f - d
    
    if use_rich:
        return [
            (glyph_unicode("bar_full") * f,   "izero.bar.fresh"),
            (glyph_unicode("bar_medium") * a, "izero.bar.aging"),
            (glyph_unicode("bar_low") * d,    "izero.bar.decay"),
        ]
    return glyph("bar_full") * f + glyph("bar_medium") * a + glyph("bar_low") * d
```

### 3.5 Box Builder (`ui/glyphs.py::build_box`)

```python
def build_box(title, lines, footer="", heavy=False) -> str:
    """Complete bordered box with centered title, body lines, left-aligned footer."""
    # Calculates inner width from content
    # Uses heavy or light box chars
    # Returns single string with newlines
```

---

## 4. Command Architecture

### 4.1 Global Options (Injected into Every Command)

```python
# core/context.py
@dataclass
class CLIContext:
    db_path: str | None = None
    json: bool = False
    verbose: bool = False
    no_color: bool = False
    use_rich: bool = True
    dry_run: bool = False
    force: bool = False
    
    # Resolved at startup
    db: MemoryStore = None  # lazy
    config: dict = field(default_factory=dict)
```

### 4.2 Command Registration Pattern

```python
# commands/memory/add.py
def add_command(ctx: typer.Context, fact: str, evidence: str = "", tags: str = "", scope: str = "default", card_id: str | None = None):
    cli_ctx = get_context()
    renderer = cli_ctx.get_renderer()
    client = cli_ctx.get_client()
    
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    card_id = client.remember(fact, evidence, tag_list, scope, card_id)
    
    if cli_ctx.json:
        print(renderer.render_json({"id": card_id, "created": True, ...}))
    else:
        print(renderer.render_add_result(card_id, created=True))
```

**Lazy imports** inside command functions for fast startup (~50ms cold).

### 4.3 Main Entrypoint (`main.py`)

```python
app = typer.Typer(
    name="izero",
    help="isotope_zero — local-first cognitive memory layer for AI agents",
    add_completion=True,
    no_args_is_help=False,
    rich_markup_mode="rich",
)

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context, ...global options...):
    # 1. Handle --version
    # 2. Create CLIContext, inject into ctx.obj
    # 3. If no subcommand → run interactive menu
    # 4. Else startup() connects DB, etc.
    
    if ctx.invoked_subcommand is None:
        from .commands.ui.menu import run_menu
        raise typer.Exit(run_menu(cli_ctx))
    
    startup(cli_ctx)
```

---

## 5. Output Format Behavior

| Format | Trigger | Memory Rows | Stats | Dashboard | Menu |
|--------|---------|-------------|-------|-----------|------|
| **JSON** | `--json` | Structured array | Full object | State dict | Entries array |
| **Plain** | `--no-color` or no TTY | ASCII glyphs, ANSI colors | Text tables | Static snapshot | Numbered list |
| **Rich** | Default (TTY + Unicode) | Styled tables, vitality bars | Panels, histograms | Live TUI (Live) | Arrow nav + search |

**Auto-detection logic** (`render/base.py::get_renderer_from_context`):
```python
if ctx.json:           format = "json"
elif ctx.no_color or not ctx.use_rich:  format = "plain"
else:                  format = "rich"
```

---

## 6. Interactive Menu (`commands/ui/menu.py`)

### Rich Mode (TTY + Unicode)
- **Arrow keys** ↑↓ to navigate
- **Enter** to execute
- **/** to search/filter commands
- **q** / **Esc** to quit
- **Live rendering** via `rich.live.Live` at 30fps

### Plain Mode (No TTY / No Unicode)
- Numbered list (1-13)
- Type number + Enter
- 'q' to quit

### Menu Entries (13 Core Actions)
```
1. add a memory           → izero add "fact" [--evidence] [--tags] [--scope]
2. recall something       → izero recall "query" [-k] [-a]
3. search the store       → izero search "query" [-k] [-a]
4. list memories          → izero list [--tags] [--scope] [--limit]
5. get a memory           → izero get <card_id>
6. forget a memory        → izero forget <card_id> [-y]
7. touch (refresh)        → izero touch <card_id>
8. see stats              → izero stats [-v]
9. see tags               → izero tags [-v]
10. inspect the store     → izero inspect [--top]
11. dry-run consolidation → izero dry-run-consolidation [--limit]
12. open live dashboard   → izero dashboard [--interval] [--once]
13. exit
```

### Banner (Dynamic)
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Your memories live at: ~/.isotope_zero/db       ┃
┃ 42 cards remembered so far.                     ┃
┃ What would you like to do?                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## 7. Enhanced Dashboard (`commands/ui/dashboard.py`)

### Rich Live Mode
- **Auto-refresh** every N seconds (default 2s)
- **Full TUI** with `rich.live.Live(screen=True)`
- **Ctrl-C** to exit cleanly
- **Resize handling** via rich

### Dashboard Panels
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ isotope_zero · ~/.isotope_zero/db                   ┃
┃ cards: 42      size: 1.2 MB     mode: REAL ONNX     ┃
┃ vitality: ████████▒▒░░░░░░░░░░  fresh 28 / aging 9 / decay 5  ┃
┃ tokens: 15,234 total  ·  ~3,200 reclaimable         ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ recent:                                           ┃
┃  · "User prefers dark mode"          (0.2d)       ┃
┃  · "API key rotation every 90 days"  (1.5d)       ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ decay candidates (5):                             ┃
┃  · "Old config from v0.9"            v=0.12 45d   ┃
┃  · "Deprecated endpoint notes"       v=0.21 38d   ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ refresh 2s  ·  Ctrl-C to exit  ·  [?] help        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Plain/JSON Modes
- `--once` → single static frame
- `--json` → machine-readable state dict
- Non-TTY → single static frame

---

## 8. Integration Commands

### `izero plugin` (Subcommands)
```bash
izero plugin install [--editor vscode|cursor|windsurf|zed] [--target DIR] [--dry-run]
izero plugin uninstall [--editor vscode|...] [-y]
izero plugin status [--editor vscode|...]
```

### `izero mcp`
```bash
izero mcp  # Starts MCP server on stdio
```

### `izero hook` (Claude Code Lifecycle)
```bash
izero hook pre-tool-use --tool <name> --tool-input <json> --tool-output <json>
izero hook post-tool-use ...
izero hook notification ...
izero hook stop ...
```
Reads stdin for full hook payload, writes result to stdout (JSON or human).

---

## 9. Dev Commands

### `izero benchmark`
```bash
izero benchmark [--cards 10000] [--queries 100]
```
Outputs latency percentiles (p50, p95, p99), recall@k, throughput.

### `izero debug`
```bash
izero debug store      # DB integrity, schema, indexes
izero debug embedder   # Embedder health, dimensions, mode
izero debug consolidation  # Dry-run with verbose scoring
```

---

## 10. Library Choices

| Layer | Library | Rationale |
|-------|---------|-----------|
| **CLI Framework** | `typer` | Type-safe, auto-completion, rich help, async-ready |
| **TUI Rendering** | `rich` | Panels, tables, live, markup, themes, cross-platform |
| **Console I/O** | `rich.console.Console` | Unified color/theme/glyph handling |
| **Config** | `tomli` (stdlib 3.11+) / `tomli_w` | TOML for human-editable config |
| **Completion** | `typer` built-in + custom completers | Shell completion for tags, scopes, IDs |
| **JSON** | `orjson` (optional) / `json` (stdlib) | Fast serialization, fallback to stdlib |
| **DB** | `sqlite3` (stdlib) | Zero-dep, ACID, FTS5 for BM25 |

**Zero hard dependencies beyond typer + rich** (both pure Python).  
`orjson` is optional — graceful fallback to stdlib `json`.

---

## 11. Backward Compatibility

### Preserved Commands (Exact Same Interface)
| Command | Flags | Behavior |
|---------|-------|----------|
| `izero add` | `fact`, `-e`, `-t`, `-s`, `--id` | ✅ |
| `izero recall` | `query`, `-k`, `-a` | ✅ |
| `izero search` | `query`, `-k`, `-a` | ✅ |
| `izero list` | `-t`, `-s`, `-l` | ✅ |
| `izero get` | `card_id` | ✅ |
| `izero forget` | `card_id`, `-y` | ✅ |
| `izero touch` | `card_id` | ✅ |
| `izero stats` | `-v` | ✅ |
| `izero tags` | `-v` | ✅ |
| `izero inspect` | `--top` | ✅ |
| `izero dry-run-consolidation` | `--limit` | ✅ |
| `izero dashboard` | `-i`, `--once` | ✅ |
| `izero plugin` | `install/uninstall/status` | ✅ |
| `izero mcp` | (none) | ✅ |
| `izero hook` | `event`, `--tool`, `--tool-input`, etc. | ✅ |
| `izero benchmark` | `--cards`, `--queries` | ✅ |
| `izero debug` | `store/embedder/consolidation` | ✅ |

### Global Flags (All Commands)
- `--json` → Machine output
- `-v/--verbose` → Technical detail
- `--no-color` → Force plain renderer
- `--db PATH` → Override database
- `--dry-run` → Preview only (applicable commands)
- `-y/--force` → Skip confirmations

---

## 12. Implementation Priority

### Phase 1: Foundation (Week 1)
1. `theme.py` — Complete palette, semantic aliases
2. `glyphs.py` — Unicode/ASCII, vitality bar, box builder
3. `render/base.py` — Protocol, components dataclasses
4. `render/json_renderer.py` — Passthrough
5. `render/plain_renderer.py` — ANSI, ASCII glyphs
6. `render/rich_renderer.py` — Rich panels, tables, bars
7. `core/context.py` — CLIContext, option factories, injection

### Phase 2: Commands (Week 2)
8. `commands/memory/` — add, get, list, forget, touch
9. `commands/query/` — recall, search
10. `commands/insight/` — stats, tags, inspect, active, dry-run
11. `commands/ui/dashboard.py` — Live TUI + plain/json modes
12. `commands/ui/menu.py` — Arrow nav + search + plain fallback
13. `commands/integrations/` — plugin, mcp, hook
14. `commands/dev/` — benchmark, debug

### Phase 3: Polish (Week 3)
13. Shell completion (tags, scopes, card IDs)
14. Config file support (`~/.config/izero/config.toml`)
15. Man page generation (`typer-cli` completion)
16. Tests for all renderers + commands

---

## 13. Visual Reference

### Memory Row (Rich)
```
 1.  User prefers dark mode for coding        ·  prefs,ui  ·  0.9234  ·  ████████▒▒  0.87
 2.  API keys rotate every 90 days            ·  security  ·  0.8871  ·  ████████▒▒  0.82
 3.  Old config from v0.9 deprecated          ·  legacy    ·  0.1234  ·  ░░░░░░░░░░  0.12
```

### Memory Card (Rich, Verbose)
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ User prefers dark mode for coding                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ evidence:  "Switched to dark theme in VS Code"       ┃
┃ tags:      prefs, ui                                 ┃
┃ id:        a1b2c3d4e5f6                              ┃
┃ score:     0.9234                                    ┃
┃ vitality:  ████████▒▒░░░░░░░░░░  0.87                ┃
┃ created:   2026-07-15 14:32:11                       ┃
┃ accessed:  2026-08-09 09:15:42                       ┃
┃ accesses:  23                                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Stats (Rich, Verbose)
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 42 memories  ·  1.2 MB  ·  REAL ONNX  ·  ~15K tokens  ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ tag distribution:                                    ┃
┃  prefs      12                                       ┃
┃  security   8                                        ┃
┃  ui         7                                        ┃
┃  legacy     3                                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ vitality histogram:                                  ┃
┃  fresh (>=0.66)    ████████████████████  28          ┃
┃  aging (0.33-0.66) ████████▒▒▒▒▒▒▒▒▒▒  9              ┃
┃  decayed (<0.33)   ░░░░░░░░░░░░░░░░░░  5              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## 14. Configuration File (Optional)

`~/.config/izero/config.toml`:
```toml
[database]
path = "~/.isotope_zero/isotope_zero.db"

[embedding]
mode = "auto"  # auto | real | fallback | none

[cli]
default_format = "rich"  # rich | plain | json
verbose = false
no_color = false

[dashboard]
interval = 2.0
show_decay = true
show_recent = 5

[menu]
show_on_start = true
search_enabled = true
```

---

## 15. Testing Strategy

| Layer | Approach |
|-------|----------|
| **Renderers** | Snapshot tests: compare output strings for known inputs across all 3 formats |
| **Commands** | Integration tests: invoke via `typer.testing.CliRunner`, assert JSON structure |
| **Theme** | Visual regression: render dashboard/menu in CI, compare PNG (optional) |
| **Glyphs** | Unit tests: `can_unicode()` mocking, fallback behavior |
| **Vitality Bar** | Property tests: proportional segments sum to width, correct colors |

---

## 16. Migration Notes

### From Current Architecture
- **Renderer protocol unchanged** — existing `JsonRenderer`, `PlainRenderer`, `RichRenderer` implementations map 1:1
- **Theme colors updated** — semantic names now map to black/beige/white palette
- **Glyphs enhanced** — added heavy box, radio, checkbox, more arrows
- **Commands reorganized** — flat `commands/` → domain-subdir structure (`memory/`, `query/`, etc.)
- **Menu/dashboard rewritten** — now use shared `ui/components.py` primitives

### Breaking Changes (None)
- All command signatures preserved
- All global flags preserved
- JSON output schema preserved
- Config env vars preserved (`ISOTOPE_ZERO_DB`)

---

*Design complete. Ready for implementation.*
