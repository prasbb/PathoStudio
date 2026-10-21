"""Flat, optional local checkpoint configuration; no weight downloads."""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from ..models.local_settings import read_model_paths, save_model_paths, clear_model_paths


MODEL_INFO = {
    "conch": ("CONCH", "https://huggingface.co/MahmoodLab/CONCH"),
    "musk": ("MUSK", "https://huggingface.co/xiangjx/musk"),
}


class LocalModelsDialog(QDialog):
    """Allows each researcher to configure weights they independently obtained."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Local Models")
        self.setMinimumWidth(570)
        self.changed = False
        self._controls: dict[str, dict] = {}
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        heading = QLabel(
            "Model weights are NOT distributed with Ashby. Models run locally; "
            "Ashby does not download or upload any model files."
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)
        for model_id in ("conch", "musk"):
            self._add_model_group(layout, model_id)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)
        self.setStyleSheet("QPushButton, QLineEdit, QGroupBox { border-radius: 0px; }")

    def _add_model_group(self, outer, model_id: str):
        name, url = MODEL_INFO[model_id]
        group = QGroupBox(f"{name} — user-supplied local files")
        form = QFormLayout(group)
        form.setContentsMargins(10, 16, 10, 9)
        form.setSpacing(7)

        saved = read_model_paths(model_id)
        checkpoint = QLineEdit(saved[0] if saved else "")
        checkpoint.setPlaceholderText("Select your authorized local checkpoint")
        checkpoint_row = QWidget()
        checkpoint_layout = QHBoxLayout(checkpoint_row)
        checkpoint_layout.setContentsMargins(0, 0, 0, 0)
        checkpoint_layout.addWidget(checkpoint, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda _, target=checkpoint: self._browse(target))
        checkpoint_layout.addWidget(browse)
        form.addRow("Checkpoint", checkpoint_row)

        tokenizer = QLineEdit(saved[1] if saved else "")
        if model_id == "musk":
            token_row = QWidget()
            token_layout = QHBoxLayout(token_row)
            token_layout.setContentsMargins(0, 0, 0, 0)
            token_layout.addWidget(tokenizer, 1)
            token_browse = QPushButton("Browse…")
            token_browse.clicked.connect(lambda _, target=tokenizer: self._browse(target))
            token_layout.addWidget(token_browse)
            tokenizer.setPlaceholderText("Select MUSK's tokenizer.spm")
            form.addRow("Tokenizer", token_row)

        notice = QLabel(
            f"{name} is an independently licensed third-party research model. "
            f"Review its official <a href='{url}'>access and license conditions</a> "
            "and obtain your own permission before using its weights. "
            "Acknowledgment does not grant you a license."
        )
        notice.setWordWrap(True)
        notice.setOpenExternalLinks(True)
        form.addRow(notice)

        ack = QCheckBox("I independently obtained these files and will comply with their terms.")
        ack.setChecked(saved is not None)
        ack.setToolTip("You are responsible for complying with the model owner's terms.")
        form.addRow(ack)
        status = QLabel("Configured" if saved else "Not configured")
        status.setWordWrap(True)
        button_row = QWidget()
        buttons = QHBoxLayout(button_row)
        buttons.setContentsMargins(0, 0, 0, 0)
        validate = QPushButton("Validate and Enable")
        remove = QPushButton("Remove Configuration")
        buttons.addWidget(validate)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        form.addRow(button_row)
        form.addRow(status)

        checkpoint.textEdited.connect(lambda _: ack.setChecked(False))
        tokenizer.textEdited.connect(lambda _: ack.setChecked(False))
        validate.clicked.connect(lambda _, mid=model_id: self._validate(mid))
        remove.clicked.connect(lambda _, mid=model_id: self._remove(mid))
        self._controls[model_id] = dict(checkpoint=checkpoint, tokenizer=tokenizer,
                                         ack=ack, status=status)
        outer.addWidget(group)

    def _browse(self, field: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select locally downloaded model file")
        if path:
            field.setText(path)
            # setText does not trigger textEdited; a new file always needs acknowledgment.
            for ctrl in self._controls.values():
                if field is ctrl["checkpoint"] or field is ctrl["tokenizer"]:
                    ctrl["ack"].setChecked(False)
                    break

    def _validate(self, model_id: str):
        ctrl = self._controls[model_id]
        if not ctrl["ack"].isChecked():
            QMessageBox.warning(self, "Model license acknowledgement required",
                                "Review the official model terms and confirm that you "
                                "obtained the files with appropriate permissions.")
            return
        checkpoint = ctrl["checkpoint"].text().strip()
        tokenizer = ctrl["tokenizer"].text().strip()
        if not Path(checkpoint).expanduser().is_file() or (model_id == "musk" and
                not Path(tokenizer).expanduser().is_file()):
            ctrl["status"].setText("Select existing local files before enabling.")
            return
        ctrl["status"].setText("Validating local model; this may take some time…")
        QApplication.processEvents()
        try:
            if model_id == "conch":
                from ..models.conch import ConchAdapter
                adapter = ConchAdapter(checkpoint)
            else:
                from ..models.musk import MuskAdapter
                adapter = MuskAdapter(checkpoint, tokenizer)
            if not adapter.available:
                raise RuntimeError(adapter.reason_unavailable())
            save_model_paths(model_id, checkpoint, tokenizer)
        except Exception as exc:
            ctrl["status"].setText(f"Unavailable: {exc}")
            return
        ctrl["status"].setText("Ready — enabled for local inference.")
        self.changed = True

    def _remove(self, model_id: str):
        clear_model_paths(model_id)
        ctrl = self._controls[model_id]
        ctrl["checkpoint"].clear()
        ctrl["tokenizer"].clear()
        ctrl["ack"].setChecked(False)
        ctrl["status"].setText("Not configured")
        self.changed = True
