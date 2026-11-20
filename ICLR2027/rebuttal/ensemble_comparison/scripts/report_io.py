"""Respect the user's explicit control over the experiment Markdown document."""
import json
import os
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
DOCUMENT=RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md'
PREFERENCES=RUN/'configs/reporting_preferences.json'


def markdown_enabled():
    try:
        return json.loads(PREFERENCES.read_text()).get('markdown_enabled') is True
    except (FileNotFoundError,ValueError,TypeError):
        return False


def read_markdown(path=None):
    """A title fallback exists in memory only, so empty-document renders work."""
    path=Path(path) if path is not None else DOCUMENT
    value=path.read_text() if path.exists() else ''
    return value if value.strip() else '# Ensemble 对照平均结果\n'


def write_markdown(text,path=None):
    """No report may repopulate the document before explicit user instruction."""
    if not markdown_enabled():
        print('MD_WRITE_SKIPPED: user requested chat-only results until explicitly re-enabled',flush=True)
        return False
    path=Path(path) if path is not None else DOCUMENT
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(text)
    temp.replace(path)
    return True
