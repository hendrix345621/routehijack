"""Mini terminal UI shared across every red-team script.

Goal: every step of the pipeline should be *legible at a glance*. You see:
  - which step is running and where it is in the sequence
  - a progress bar for any long-running loop
  - per-cell intermediate stats (sample size, ASR running, etc.)
  - **sample conversations** as the model produces them
  - a final summary table

Sample conversations are the transparency anchor — a "refused" boolean from a
string detector can lie. Reading the actual completion is how you catch:
  - false positives (a safety-flavored compliance counted as refusal)
  - false negatives (a vague-but-harmful response counted as refusal)
  - obvious bugs (the model emitting <eos> or garbage)

All transcript output also lands on disk as markdown so reviewers can scroll
through them in an IDE.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


def _make_console() -> Console:
    """Build a Console that handles Unicode on Windows.

    The default Console pipes through `sys.stdout`, which on Windows is cp1252
    and chokes on the box-drawing characters Rich uses for rules and panels.
    We re-wrap stdout in a UTF-8 writer with replacement so the UI is legible
    on any terminal."""
    stream = sys.stdout
    encoding = getattr(stream, "encoding", "") or ""
    if encoding.lower() not in ("utf-8", "utf8"):
        try:
            stream = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except Exception:
            stream = sys.stdout
    return Console(file=stream, highlight=False, force_terminal=stream.isatty())


# Shared console — one per process. Reused by every script so spacing is consistent.
_console = _make_console()


def console() -> Console:
    """Access the shared rich Console (for callers needing finer control)."""
    return _console


# ─────────────────────────── Step headers ───────────────────────────


def step_header(step_num: int, title: str, *, total: int | None = None) -> None:
    """Big banner. Use once at the start of each script.

    Example:
        ui.step_header(2, "RouteAudit suffix search", total=10)
        ═══ Step 2/10 ── RouteAudit suffix search ═══
    """
    if total is not None:
        line = f"Step {step_num}/{total} ── {title}"
    else:
        line = f"Step {step_num} ── {title}"
    _console.rule(f"[bold cyan]{line}[/bold cyan]")


def section(title: str) -> None:
    """Subsection within a step."""
    _console.print(f"\n[bold yellow]▸ {title}[/bold yellow]")


def info(msg: str) -> None:
    _console.print(f"  [dim]{msg}[/dim]")


def ok(msg: str) -> None:
    _console.print(f"  [green]✓[/green] {msg}")


def warn(msg: str) -> None:
    _console.print(f"  [yellow]⚠[/yellow] {msg}")


def fail(msg: str) -> None:
    _console.print(f"  [red]✗[/red] {msg}")


# ─────────────────────────── Progress ───────────────────────────


@contextmanager
def progress_bar(total: int, desc: str = "") -> Iterator:
    """Yield a `(progress, task_id)` pair. Caller calls `progress.advance(task_id)`."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
    ) as prog:
        task_id = prog.add_task(desc or "", total=total)
        yield prog, task_id


def iter_with_progress(items: Iterable, desc: str = "") -> Iterator:
    """`for x in ui.iter_with_progress(stuff, "doing X"): ...`"""
    items = list(items)
    with progress_bar(len(items), desc=desc) as (prog, tid):
        for item in items:
            yield item
            prog.advance(tid)


# ─────────────────────────── Key-value summary ───────────────────────────


def kv_panel(title: str, items: dict, *, border_style: str = "cyan") -> None:
    """Render a key-value table inside a titled panel.

    Use for end-of-step summaries (e.g. "Identification done — top layers, ...").
    """
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("k", style="bold")
    table.add_column("v")
    for k, v in items.items():
        if isinstance(v, float):
            v = f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"
        table.add_row(str(k), str(v))
    _console.print(Panel(table, title=title, border_style=border_style))


def results_table(title: str, rows: list[dict], columns: list[str]) -> None:
    """Print a tidy results table with row striping.

    `rows` is a list of dicts. `columns` is the column order (subset of dict keys).
    Numeric values are formatted to 3 decimal places when between 0 and 1.
    """
    table = Table(title=title, header_style="bold magenta", row_styles=["", "dim"])
    for c in columns:
        table.add_column(c)
    for r in rows:
        formatted = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.3f}" if 0 <= v <= 1 else f"{v:.2f}"
            formatted.append(str(v))
        table.add_row(*formatted)
    _console.print(table)


# ─────────────────────────── Sample transcripts ───────────────────────────


@dataclass
class Transcript:
    """One model exchange — what we're showing the user for transparency."""
    cell: str                          # e.g. "combined_def_vs_routeaudit"
    prompt: str
    completion: str
    refused: bool                      # the cheap string-based verdict
    judged_harmful: Optional[bool] = None   # set after HarmBench grader
    notes: dict = field(default_factory=dict)


def show_transcript(t: Transcript, *, prompt_max_chars: int = 200,
                    completion_max_chars: int = 600) -> None:
    """Live-print one transcript as a coloured panel."""
    verdict = "REFUSED" if t.refused else "COMPLIED"
    color = "green" if t.refused else "red"

    body = Text()
    body.append("PROMPT  ", style="bold")
    body.append(_clip(t.prompt, prompt_max_chars), style="cyan")
    body.append("\n\nMODEL   ", style="bold")
    body.append(_clip(t.completion, completion_max_chars))
    if t.judged_harmful is not None:
        body.append("\n\nJUDGE   ", style="bold")
        body.append("HARMFUL" if t.judged_harmful else "SAFE",
                    style="red" if t.judged_harmful else "green")
    title = f"[{color}]{verdict}[/{color}]  ·  {t.cell}"
    _console.print(Panel(body, title=title, border_style=color))


def _clip(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


# ─────────────────────────── Transcript persistence ───────────────────────────


class TranscriptLog:
    """Append-only markdown log of transcripts for one cell.

    One file per cell, e.g.:
        artifacts/transcripts/combined_def_vs_routeaudit.md
    """

    def __init__(self, cell_name: str, base_dir: str | Path = "artifacts/transcripts"):
        self.cell_name = cell_name
        self.path = Path(base_dir) / f"{cell_name}.md"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Wipe at start of cell — caller may want appending; if so, call `append_existing()`.
        self.path.write_text(f"# Transcripts — `{cell_name}`\n\n", encoding="utf-8")
        self._count = 0
        # Mirror as JSONL too, for downstream programmatic re-grading.
        self.jsonl_path = self.path.with_suffix(".jsonl")
        self.jsonl_path.write_text("", encoding="utf-8")

    def append(self, t: Transcript) -> None:
        self._count += 1
        verdict = "**REFUSED**" if t.refused else "**COMPLIED**"
        chunk = (
            f"## Sample {self._count}  ·  {verdict}\n\n"
            f"**Prompt**\n\n```\n{t.prompt}\n```\n\n"
            f"**Completion**\n\n```\n{t.completion}\n```\n\n"
        )
        if t.judged_harmful is not None:
            chunk += f"**HarmBench judge:** {'harmful' if t.judged_harmful else 'safe'}\n\n"
        if t.notes:
            chunk += f"**Notes:** `{json.dumps(t.notes)}`\n\n"
        chunk += "---\n\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(chunk)
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "cell": t.cell, "prompt": t.prompt, "completion": t.completion,
                "refused": t.refused, "judged_harmful": t.judged_harmful, "notes": t.notes,
            }) + "\n")

    @property
    def count(self) -> int:
        return self._count


# ─────────────────────────── Final summary ───────────────────────────


def big_banner(text: str, *, style: str = "bold cyan") -> None:
    _console.rule(f"[{style}]{text}[/{style}]", style=style)


def print_done(msg: str = "Done.") -> None:
    _console.rule(f"[bold green]✓ {msg}[/bold green]", style="green")
