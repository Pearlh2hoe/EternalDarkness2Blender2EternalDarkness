"""Shared logo icons (menu entries, dialogs, sidebar) for both the importer and the exporter."""
import os

_icons = None


def load():
    global _icons
    try:
        import bpy.utils.previews
        _icons = bpy.utils.previews.new()
        _icons.load("menu", os.path.join(os.path.dirname(__file__), "logo_menu.png"), 'IMAGE')
        _icons.load("dialog", os.path.join(os.path.dirname(__file__), "logo_dialog.png"), 'IMAGE')
    except Exception:
        _icons = None


def unload():
    global _icons
    if _icons is not None:
        try:
            import bpy.utils.previews
            bpy.utils.previews.remove(_icons)
        except Exception:
            pass
        _icons = None


def kw(name="menu"):
    """keyword arguments for layout.operator()/label(): the logo when available"""
    try: return {"icon_value": _icons[name].icon_id}
    except Exception: return {}


def dialog_header(layout, title):
    """big logo + title at the top of a file dialog"""
    k = kw("dialog")
    if k:
        row = layout.row(); row.alignment = 'CENTER'
        row.template_icon(icon_value=k["icon_value"], scale=5.0)
    layout.label(text=title)
