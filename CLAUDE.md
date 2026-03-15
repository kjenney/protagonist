# protagonist

A system to build personalized videos by editing existing videos and including new audio with subtitles.

## Stack
- Textual 7.x for TUI
- External CSS in app.tcss (live reload via `textual run --dev`)
- Tests via Textual Pilot API + pytest

## Dev workflow
- Run: `textual run --dev app.py`
- Test: `pytest tests/`
- MCP: textual-mcp-server is configured in .mcp.json