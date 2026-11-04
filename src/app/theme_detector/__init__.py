import os
from typing import Optional
import streamlit.components.v1 as components

_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
_theme_sync_component = components.declare_component("theme_sync", path=_PARENT_DIR)


def sync_native_theme(current_theme: Optional[str] = None, key: str = "native_theme_sync") -> Optional[str]:
    """Sync with Streamlit's native theme (System / Light / Dark) from the browser.

    Returns 'light' or 'dark' if detected from the frontend, or None if not ready.
    """
    try:
        val = _theme_sync_component(current_theme=current_theme, default=current_theme, key=key)
        if isinstance(val, str) and val.lower() in ("light", "dark"):
            return val.lower()
    except Exception:
        pass
    return None
