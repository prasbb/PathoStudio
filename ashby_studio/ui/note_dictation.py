"""Optional, offline microphone dictation for pathologist ROI notes.

Requires a user-provided whisper.cpp `whisper-cli` binary, a user-provided
compatible GGML model, and `sounddevice` for microphone capture. No service,
model download, or network connection is used by this module.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import wave

from qtpy.QtCore import QSettings, QThread, QTimer, Signal
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

_MAX_SECONDS = 90
_RATE = 16000
# Keep the thread object alive even when the main window closes mid-inference.
_ACTIVE_WORKERS: set[QThread] = set()


class _TranscribeWorker(QThread):
    transcribed = Signal(str)
    failed = Signal(str)

    def __init__(self, binary: Path, model: Path, audio: bytes, parent=None):
        super().__init__(parent)
        self._binary = binary
        self._model = model
        self._audio = audio

    def run(self):
        try:
            with tempfile.TemporaryDirectory(prefix="ashby-dictation-") as directory:
                base = Path(directory)
                wav_path = base / "recording.wav"
                prefix = base / "transcript"
                with wave.open(str(wav_path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)  # signed 16-bit little endian PCM
                    output.setframerate(_RATE)
                    output.writeframes(self._audio)
                # No shell, no remote API: whisper.cpp reads local weights/audio.
                completed = subprocess.run(
                    [str(self._binary), "-m", str(self._model), "-f", str(wav_path),
                     "-l", "en", "-otxt", "-of", str(prefix)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True, timeout=180, check=False,
                )
                if completed.returncode:
                    detail = (completed.stderr or "").strip().splitlines()
                    raise RuntimeError(detail[-1][:250] if detail else "Local speech engine failed")
                output_path = Path(f"{prefix}.txt")
                if not output_path.is_file():
                    raise RuntimeError("Local speech engine did not produce a transcript")
                result = output_path.read_text(encoding="utf-8").strip()
                if not result:
                    raise RuntimeError("No speech was recognized. Try speaking clearly into the microphone.")
                self.transcribed.emit(result)
        except subprocess.TimeoutExpired:
            self.failed.emit("Local transcription timed out after three minutes.")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            self._audio = b""


class NoteDictationControls(QWidget):
    """Attach to an existing RoiDetailDock without replacing its implementation.

    The current ROI identity is captured when recording starts, and a transcript
    is inserted only if that same ROI is still selected. Saving remains manual.
    """

    def __init__(self, roi_dock, parent=None):
        super().__init__(parent or roi_dock)
        self._roi = roi_dock
        self._settings = QSettings("AshbyPathologyStudio", "AshbyPathologyStudio")
        self._audio_stream = None
        self._audio_chunks: list[bytes] = []
        self._roi_id: str | None = None
        self._worker: _TranscribeWorker | None = None
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self.stop_recording)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        actions = QHBoxLayout()
        actions.setSpacing(5)
        self.start_btn = QPushButton("Dictate")
        self.start_btn.setToolTip("Record from the microphone; transcribe entirely offline")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.setup_btn = QPushButton("Speech setup…")
        actions.addWidget(self.start_btn)
        actions.addWidget(self.stop_btn)
        actions.addWidget(self.setup_btn)
        layout.addLayout(actions)
        self.status = QLabel("Optional: local whisper.cpp; transcription stays editable until saved.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        self.setup_btn.clicked.connect(self.configure)

    def configure(self) -> bool:
        QMessageBox.information(
            self, "Local speech-to-text setup",
            "This optional feature records microphone audio on this computer and uses "
            "a locally installed whisper.cpp command-line tool and speech model. "
            "Ashby does not provide, license, or download either file. "
            "Verify the licenses and permissions for your own files before use. "
            "Temporary recordings are deleted after transcription."
        )
        previous_binary = str(self._settings.value("dictation/whisper_cli", ""))
        binary, _ = QFileDialog.getOpenFileName(
            self, "Choose local whisper.cpp whisper-cli executable", previous_binary or str(Path.home())
        )
        if not binary:
            return False
        previous_model = str(self._settings.value("dictation/model", ""))
        model, _ = QFileDialog.getOpenFileName(
            self, "Choose local whisper.cpp GGML model", previous_model or str(Path.home()),
            "Model files (*.bin);;All files (*)"
        )
        if not model:
            return False
        error = self._validate(Path(binary), Path(model))
        if error:
            QMessageBox.warning(self, "Invalid local speech setup", error)
            return False
        self._settings.setValue("dictation/whisper_cli", binary)
        self._settings.setValue("dictation/model", model)
        self.status.setText("Local speech engine configured. Select an ROI and click Dictate.")
        return True

    @staticmethod
    def _validate(binary: Path, model: Path) -> str:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            return "Choose an installed, executable whisper.cpp whisper-cli program."
        if not model.is_file():
            return "Choose an existing whisper.cpp-compatible GGML model file."
        return ""

    def start_recording(self):
        if self._audio_stream is not None or (self._worker and self._worker.isRunning()):
            return
        if self._roi.current_roi_id is None:
            QMessageBox.information(self, "Select ROI", "Select exactly one ROI before dictating notes.")
            return
        binary = Path(str(self._settings.value("dictation/whisper_cli", "")))
        model = Path(str(self._settings.value("dictation/model", "")))
        if self._validate(binary, model):
            if not self.configure():
                return
        try:
            import sounddevice as sd  # Optional microphone backend; never imported at app startup.
        except ImportError:
            QMessageBox.warning(
                self, "Microphone backend missing",
                "Local dictation requires the optional Python package 'sounddevice' "
                "in your Ashby environment. Install it separately if you wish to use dictation."
            )
            return
        self._audio_chunks = []
        self._roi_id = self._roi.current_roi_id

        def receive_audio(indata, frames, time_info, status):
            # PortAudio callback; do not call Qt or edit widgets from this thread.
            self._audio_chunks.append(bytes(indata))

        try:
            stream = sd.RawInputStream(
                samplerate=_RATE, channels=1, dtype="int16", callback=receive_audio
            )
            stream.start()
            self._audio_stream = stream
        except Exception as exc:  # noqa: BLE001
            self._roi_id = None
            QMessageBox.warning(
                self, "Microphone unavailable", f"Could not start recording: {exc}\n\n"
                "On macOS, allow microphone access for the application or Terminal "
                "in System Settings → Privacy & Security → Microphone."
            )
            return
        self.start_btn.setEnabled(False)
        self.setup_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Recording locally… Click Stop to transcribe (maximum 90 seconds).")
        self._timeout.start(_MAX_SECONDS * 1000)

    def stop_recording(self):
        stream = self._audio_stream
        if stream is None:
            return
        self._timeout.stop()
        self._audio_stream = None
        try:
            stream.stop()
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Microphone stopped unexpectedly: {exc}")
        finally:
            stream.close()
        audio = b"".join(self._audio_chunks)
        self._audio_chunks.clear()
        self.stop_btn.setEnabled(False)
        if len(audio) < 2 * _RATE // 2:  # < 0.5 s
            self.status.setText("Recording too short; no transcript created.")
            self.start_btn.setEnabled(True)
            self.setup_btn.setEnabled(True)
            return
        binary = Path(str(self._settings.value("dictation/whisper_cli")))
        model = Path(str(self._settings.value("dictation/model")))
        self.status.setText("Transcribing offline…")
        worker = _TranscribeWorker(binary, model, audio)
        self._worker = worker
        _ACTIVE_WORKERS.add(worker)
        worker.transcribed.connect(self._on_text)
        worker.failed.connect(self._on_error)
        worker.finished.connect(lambda: _ACTIVE_WORKERS.discard(worker))
        worker.finished.connect(self._on_finished)
        worker.start()

    def _on_text(self, text: str):
        if self._roi.current_roi_id != self._roi_id:
            self.status.setText("ROI selection changed; transcript was not inserted into another ROI.")
            return
        editor = self._roi.notes_edit
        cursor = editor.textCursor()
        cursor.movePosition(QTextCursor.End)
        existing = editor.toPlainText()
        cursor.insertText((" " if existing and not existing.endswith((" ", "\n")) else "") + text)
        editor.setTextCursor(cursor)
        editor.setFocus()
        self.status.setText("Transcript inserted. Check accuracy, then click Save notes.")

    def _on_error(self, error: str):
        self.status.setText(f"Dictation failed: {error}")

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.setup_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._worker = None
        self._roi_id = None


def attach_note_dictation(roi_dock) -> NoteDictationControls:
    """Add a compact row immediately after Notes in the existing ROI form."""
    control = NoteDictationControls(roi_dock)
    layout = roi_dock.layout()
    if isinstance(layout, QFormLayout):
        row = -1
        for i in range(layout.rowCount()):
            item = layout.itemAt(i, QFormLayout.FieldRole)
            if item and item.widget() is getattr(roi_dock, "notes_edit", None):
                row = i
                break
        if row >= 0:
            layout.insertRow(row + 1, control)
        else:
            layout.addRow(control)
    elif layout is not None:
        layout.addWidget(control)
    else:
        raise RuntimeError("ROI Notes editor has no Qt layout")
    return control
