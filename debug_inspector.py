"""
debug_inspector.py — SimpleMem Debug Inspector
===============================================

Inspect stored memory entries: overview, search by substring on the
restatement text, detect duplicate or suspiciously short entries, and
export everything to JSON or a pandas DataFrame.

CLI usage:
  python debug_inspector.py                        # compact overview table
  python debug_inspector.py --entry <id_prefix>    # full detail for one entry
  python debug_inspector.py --search <text>        # substring search on restatement
  python debug_inspector.py --dupes                # flag near-duplicate entries
  python debug_inspector.py --short [N]            # flag entries shorter than N chars (default 80)
  python debug_inspector.py --export entries.json  # dump all to JSON
  python debug_inspector.py --repl                 # interactive REPL

Python / notebook usage:
  from debug_inspector import SimpleMemInspector
  insp = SimpleMemInspector()
  insp.overview()
  insp.show_entry("abc12")
  insp.search("charity race")
  insp.check_dupes()
  df = insp.to_dataframe()
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config
from database.vector_store import VectorStore
from models.memory_entry import MemoryEntry


# ── colour helpers ──────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
GREEN   = "\033[92m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"

def _c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET

def _header(title: str, width: int = 72) -> str:
    bar = "━" * width
    return f"\n{_c(bar, BOLD, CYAN)}\n{_c('  ' + title, BOLD, CYAN)}\n{_c(bar, BOLD, CYAN)}"

def _wrap(text: str, indent: int = 4, width: int = 82) -> str:
    prefix = " " * indent
    return textwrap.fill(
        text, width=width, initial_indent=prefix, subsequent_indent=prefix
    )


# ── inspector ────────────────────────────────────────────────────────────────

class SimpleMemInspector:
    """
    Debug inspector for the SimpleMem VectorStore.

    Parameters
    ----------
    db_path    : path to LanceDB (defaults to config.LANCEDB_PATH)
    table_name : table name   (defaults to config.MEMORY_TABLE_NAME)
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        self.store = VectorStore(
            db_path=db_path or getattr(config, "LANCEDB_PATH", "./lancedb_data"),
            table_name=table_name or getattr(config, "MEMORY_TABLE_NAME", "memory_entries"),
        )

    # ── data ─────────────────────────────────────────────────────────────

    def _all(self) -> List[MemoryEntry]:
        return self.store.get_all_entries()

    # ── overview ─────────────────────────────────────────────────────────

    def overview(self) -> None:
        """Compact table: ID · restatement preview."""
        entries = self._all()
        if not entries:
            print(_c("\n  [empty store — no entries found]\n", YELLOW))
            return

        print(_header(f"SimpleMem Store  ({len(entries)} entries)"))

        ID_W, PREV_W = 8, 100
        print(
            f"  {_c('ID',BOLD):<{ID_W+9}}  "
            f"{_c('Restatement',BOLD)}"
        )
        print("  " + "─" * (ID_W + PREV_W + 4))

        for e in entries:
            eid  = (e.entry_id or "")[:ID_W]
            prev = e.lossless_restatement.replace("\n", " ")[:PREV_W]
            if len(e.lossless_restatement) > PREV_W:
                prev += "…"
            print(
                f"  {_c(eid, DIM):<{ID_W}}  "
                f"{_c(prev, DIM)}"
            )

        print()
        print(
            f"  {_c('Stats:', BOLD)} "
            f"avg restatement length: "
            f"{sum(len(e.lossless_restatement) for e in entries)//len(entries)} chars"
        )
        print()

    # ── single entry detail ───────────────────────────────────────────────

    def show_entry(self, id_prefix: str) -> None:
        """Print full detail for one entry (match by ID prefix)."""
        matches = [e for e in self._all() if (e.entry_id or "").startswith(id_prefix)]
        if not matches:
            print(_c(f"\n  [no entry matching '{id_prefix}']\n", RED))
            return
        print(_header(f"Entry Detail  [{matches[0].entry_id}]"))
        _print_detail(matches[0])

    # ── filtered views ────────────────────────────────────────────────────

    def search(self, text: str) -> None:
        """Substring search on lossless_restatement (case-insensitive)."""
        txt = text.lower()
        hits = [e for e in self._all() if txt in e.lossless_restatement.lower()]
        print(_header(f"Search '{text}'  ({len(hits)} entries)"))
        for e in hits:
            # Highlight the match in the restatement
            stmt = e.lossless_restatement
            idx = stmt.lower().find(txt)
            if idx >= 0:
                before = stmt[max(0, idx-40):idx]
                match  = stmt[idx:idx+len(text)]
                after  = stmt[idx+len(text):idx+len(text)+80]
                snippet = f"…{before}{_c(match, BOLD, YELLOW)}{after}…"
            else:
                snippet = stmt[:120]
            print(f"\n  {_c((e.entry_id or '')[:8], DIM)}")
            print(f"    {snippet}")
        print()

    # ── quality checks ────────────────────────────────────────────────────

    def check_dupes(self, sim_threshold: float = 0.92) -> None:
        """
        Flag pairs of entries whose lossless_restatement text is suspiciously
        similar (using simple character-level Jaccard on word sets, no embeddings
        needed so this works without a running model).
        """
        entries = self._all()
        if len(entries) < 2:
            print(_c("\n  [need at least 2 entries to check duplicates]\n", YELLOW))
            return

        print(_header(f"Duplicate Check  ({len(entries)} entries, threshold={sim_threshold})"))

        def _jaccard(a: str, b: str) -> float:
            sa = set(a.lower().split())
            sb = set(b.lower().split())
            if not sa and not sb:
                return 1.0
            return len(sa & sb) / len(sa | sb)

        found = 0
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                sim = _jaccard(
                    entries[i].lossless_restatement,
                    entries[j].lossless_restatement,
                )
                if sim >= sim_threshold:
                    found += 1
                    ei, ej = entries[i], entries[j]
                    print(
                        f"\n  {_c('DUPE', BOLD, YELLOW)}  "
                        f"sim={sim:.3f}  "
                        f"{_c((ei.entry_id or '')[:8], DIM)} × "
                        f"{_c((ej.entry_id or '')[:8], DIM)}"
                    )
                    print(f"    A: {ei.lossless_restatement[:120]}")
                    print(f"    B: {ej.lossless_restatement[:120]}")

        if found == 0:
            print(_c("\n  ✓ No near-duplicates found.\n", GREEN))
        else:
            print(f"\n  {_c(f'{found} duplicate pair(s) found', BOLD, YELLOW)}\n")

    def check_short(self, min_chars: int = 80) -> None:
        """Flag entries whose restatement is shorter than min_chars characters."""
        entries = self._all()
        short = [e for e in entries if len(e.lossless_restatement) < min_chars]
        print(_header(f"Short Entries  (< {min_chars} chars, {len(short)} found)"))
        if not short:
            print(_c("\n  ✓ All entries meet the minimum length.\n", GREEN))
            return
        for e in short:
            n = len(e.lossless_restatement)
            print(
                f"\n  {_c('SHORT', BOLD, YELLOW)}  "
                f"{_c((e.entry_id or '')[:8], DIM)}  "
                f"{_c(f'{n} chars', RED)}"
            )
            print(f"    {e.lossless_restatement}")
        print()

    # ── export ────────────────────────────────────────────────────────────

    def export_json(self, path: str) -> None:
        """Dump all entries to a JSON file."""
        entries = self._all()
        data = [
            {
                "entry_id":              e.entry_id,
                "lossless_restatement":  e.lossless_restatement,
                "restatement_len":       len(e.lossless_restatement),
            }
            for e in entries
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(_c(f"\n  ✓ Exported {len(data)} entries → {path}\n", GREEN))

    def to_dataframe(self):
        """Return all entries as a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pip install pandas")
        entries = self._all()
        return pd.DataFrame([
            {
                "entry_id":             e.entry_id,
                "restatement_len":      len(e.lossless_restatement),
                "lossless_restatement": e.lossless_restatement,
            }
            for e in entries
        ])

    # ── REPL ─────────────────────────────────────────────────────────────

    def repl(self) -> None:
        """Interactive REPL for exploring the store without restarting Python."""
        print(_c("\nSimpleMem Debug REPL — 'help' for commands, 'quit' to exit\n", BOLD, CYAN))
        while True:
            try:
                line = input(_c("smdbg> ", BOLD, GREEN)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            parts = line.split(maxsplit=1)
            cmd, arg = parts[0].lower(), parts[1] if len(parts) > 1 else ""
            try:
                if cmd in ("q", "quit", "exit"):
                    break
                elif cmd == "help":
                    _repl_help()
                elif cmd == "overview":
                    self.overview()
                elif cmd == "entry":
                    self.show_entry(arg) if arg else print("  Usage: entry <id_prefix>")
                elif cmd == "search":
                    self.search(arg) if arg else print("  Usage: search <text>")
                elif cmd == "dupes":
                    thresh = float(arg) if arg else 0.92
                    self.check_dupes(thresh)
                elif cmd == "short":
                    n = int(arg) if arg else 80
                    self.check_short(n)
                elif cmd == "export":
                    self.export_json(arg or "entries_debug.json")
                else:
                    print(f"  Unknown command '{cmd}' — type 'help'")
            except Exception as exc:
                print(_c(f"  Error: {exc}", RED))


# ── helpers ──────────────────────────────────────────────────────────────────

def _print_detail(e: MemoryEntry) -> None:
    ind = "    "
    print(f"\n  {_c('▸', BOLD, BLUE)} {_c((e.entry_id or '')[:8], DIM)}")
    print(f"{ind}{_c('Length     :', BOLD)} {len(e.lossless_restatement)} chars")
    print(f"\n{ind}{_c('Restatement:', BOLD)}")
    print(_wrap(e.lossless_restatement, indent=6))
    print()


def _repl_help() -> None:
    print(_c("""
  Commands:
    overview           — compact table of all entries
    entry  <id>        — full detail for one entry (ID prefix)
    search <text>      — substring search on restatement text
    dupes  [threshold] — flag near-duplicate entries (default 0.92)
    short  [N]         — flag entries shorter than N chars (default 80)
    export [path]      — dump all entries to JSON
    quit               — exit
""", DIM))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="SimpleMem Debug Inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python debug_inspector.py
              python debug_inspector.py --search "charity race"
              python debug_inspector.py --entry abc123
              python debug_inspector.py --dupes
              python debug_inspector.py --short 60
              python debug_inspector.py --export entries.json
              python debug_inspector.py --repl
        """),
    )
    p.add_argument("--db",     default=None, help="LanceDB path override")
    p.add_argument("--table",  default=None, help="Table name override")
    p.add_argument("--entry",  metavar="ID",   help="Full detail for entry ID prefix")
    p.add_argument("--search", metavar="TEXT", help="Substring search on restatement")
    p.add_argument("--dupes",  nargs="?", const=0.92, type=float, metavar="THRESH",
                   help="Flag near-duplicates (default threshold=0.92)")
    p.add_argument("--short",  nargs="?", const=80, type=int, metavar="N",
                   help="Flag entries shorter than N chars (default 80)")
    p.add_argument("--export", metavar="PATH", help="Export all entries to JSON")
    p.add_argument("--repl",   action="store_true", help="Interactive REPL")
    args = p.parse_args()

    insp = SimpleMemInspector(db_path=args.db, table_name=args.table)

    if   args.repl:            insp.repl()
    elif args.entry:           insp.show_entry(args.entry)
    elif args.search:          insp.search(args.search)
    elif args.dupes is not None: insp.check_dupes(args.dupes)
    elif args.short is not None: insp.check_short(args.short)
    elif args.export:          insp.export_json(args.export)
    else:                      insp.overview()


if __name__ == "__main__":
    main()
