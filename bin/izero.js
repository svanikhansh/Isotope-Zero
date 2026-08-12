#!/usr/bin/env bun
import { runTUI } from '../src/tui/App.js';

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === 'tui') {
    await runTUI();
    return;
  }

  const command = args[0];

  switch (command) {
    case 'help':
      console.log(`
isotope-zero - Sub-millisecond, local-first cognitive memory layer

Usage: izero <command> [options]

Commands:
  tui           Launch the Terminal UI (default)
  help          Show this help message
  version       Show current version
  add           Add a new memory
  search        Search memories
  list          List all memories
  inspect       Inspect decay and statistics
  consolidate   Run memory consolidation
  serve         Start web dashboard server
  config        Manage configuration

Options:
  --theme <name>    Set color theme (ocean-blue, forest-green, sunset-orange, monochrome)
  --no-color        Disable colored output
  --json            Output as JSON
      `);
      break;
    case 'version':
      console.log('isotope-zero v1.4.0');
      break;
    default:
      console.error(`Unknown command: ${command}`);
      console.error('Run "izero help" for usage information.');
      process.exit(1);
  }
}

main().catch((err) => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
