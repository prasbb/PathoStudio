"""Shared test configuration.

Set headless Qt + software OpenGL before napari/vispy are imported so the napari
canvas can be constructed without a display on macOS (no segfault).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")