"""Virchow's Suite — offline research-grade pathology annotation tool.

M0 scope: launchable local viewer, manual polygon annotations with multiple
labels/colors, a modality selector (H&E), a model selector (None/Mock with
real models disabled), a SIMULATED mock AI overlay, and save/reopen of
annotations with AI-vs-human provenance preserved. No network access, no
model downloads, no inference on patient data at runtime.
"""

__version__ = "0.1.0"
APP_NAME = "PathoSuite"