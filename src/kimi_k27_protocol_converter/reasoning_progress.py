"""Build a concise user-visible progress message from Kimi reasoning."""


_MAX_PROGRESS_CHARS = 240


def build_visible_progress(reasoning_text: str) -> str:
    """Return only the final non-empty reasoning line, never the full reasoning."""
    lines = [line.strip() for line in reasoning_text.splitlines() if line.strip()]
    if not lines:
        return ""
    progress = lines[-1]
    if len(progress) <= _MAX_PROGRESS_CHARS:
        return progress
    return progress[:_MAX_PROGRESS_CHARS - 1] + "…"
