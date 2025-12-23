# -*- coding: utf-8 -*-
"""
C++ Typing Trainer (PyQt5 + QScintilla)

Version: 0.1.3  (2025-12-24)
Versioning: MAJOR.MINOR.PATCH (SemVer)
- MAJOR: 호환성 깨지는 변경
- MINOR: 기능 추가(호환 유지)
- PATCH: 버그 수정/리팩토링 등 소규모 개선

Release Notes (v0.1.3):
- (New) 좌측 Presets 패널 추가: 자주 연습하는 코드 목록 상시 표시 + 클릭 적용
- (New) Preset Add/Rename/Delete 기능 추가 (AppData/presets.json 영구 저장)
- (Maintain) Load .txt 기능 유지 (프리셋과 별개로 즉시 로드/연습 가능)
- (Maintain) Strict Mode, Beep on Error, overlay, paste-block, metrics 안정화 유지
"""

import sys
import os
import json
import time
import tempfile
from dataclasses import dataclass

from PyQt5.QtCore import Qt, QTimer, QSettings, QStandardPaths
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QInputDialog,
    QAbstractItemView,
    QFrame,
)


from PyQt5.Qsci import QsciScintilla, QsciLexerCPP


DEFAULT_CPP = r"""#include <iostream>
#include <vector>
#include <string>

class Greeter {
public:
    explicit Greeter(std::string name) : name_(std::move(name)) {}

    void hello() const {
        std::cout << "Hello, " << name_ << "!" << std::endl;
    }

private:
    std::string name_;
};

int main() {
    Greeter g("World");
    g.hello();

    for (int i = 0; i < 3; ++i) {
        std::cout << i << std::endl;
    }

    return 0;
}
"""


# ---------------- File Helpers ----------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _safe_write_text(path: str, text: str, retries: int = 8, backoff_s: float = 0.08):
    """
    Windows에서 파일 lock/백신 스캔 등으로 os.replace가 간헐적으로 실패할 수 있어
    tmp -> replace를 재시도(backoff) 합니다.
    """
    folder = os.path.dirname(path)
    _ensure_dir(folder)

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="._tmp_", suffix=".tmp", dir=folder)
    os.close(tmp_fd)

    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

        last_err = None
        for i in range(retries):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(backoff_s * (i + 1))
        raise last_err if last_err else PermissionError("os.replace failed")
    finally:
        # if still exists, try cleanup
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _safe_write_json(path: str, data: dict):
    _safe_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


# ---------------- Preset Store ----------------
class PresetStore:
    """
    presets.json in AppDataLocation
    schema:
    {
      "version": 1,
      "presets": [{"name": "...", "text": "..."}],
      "last_selected": "name or null"
    }
    """
    def __init__(self):
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not base:
            base = os.path.join(os.path.expanduser("~"), ".cpp_typing_trainer")
        self.dir = base
        self.path = os.path.join(self.dir, "presets.json")
        _ensure_dir(self.dir)

        self.data = {
            "version": 1,
            "presets": [{"name": "Default", "text": DEFAULT_CPP}],
            "last_selected": "Default",
        }
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            self.save()
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and "presets" in d:
                self.data = d
                # minimal validation
                if not isinstance(self.data.get("presets"), list) or len(self.data["presets"]) == 0:
                    self.data["presets"] = [{"name": "Default", "text": DEFAULT_CPP}]
                if "last_selected" not in self.data:
                    self.data["last_selected"] = None
        except Exception:
            # fallback to default if file is corrupted
            self.data = {
                "version": 1,
                "presets": [{"name": "Default", "text": DEFAULT_CPP}],
                "last_selected": "Default",
            }
            self.save()

    def save(self):
        _safe_write_json(self.path, self.data)

    def list_names(self):
        return [p.get("name", "") for p in self.data.get("presets", [])]

    def get_by_name(self, name: str):
        for p in self.data.get("presets", []):
            if p.get("name") == name:
                return p
        return None

    def upsert(self, name: str, text: str):
        name = (name or "").strip()
        if not name:
            return False
        for p in self.data.get("presets", []):
            if p.get("name") == name:
                p["text"] = text or ""
                self.save()
                return True
        self.data.setdefault("presets", []).append({"name": name, "text": text or ""})
        self.save()
        return True

    def rename(self, old: str, new: str):
        new = (new or "").strip()
        if not new:
            return False
        if old == new:
            return True
        if self.get_by_name(new) is not None:
            return False
        p = self.get_by_name(old)
        if not p:
            return False
        p["name"] = new
        if self.data.get("last_selected") == old:
            self.data["last_selected"] = new
        self.save()
        return True

    def delete(self, name: str):
        presets = self.data.get("presets", [])
        kept = [p for p in presets if p.get("name") != name]
        if len(kept) == len(presets):
            return False
        self.data["presets"] = kept if kept else [{"name": "Default", "text": DEFAULT_CPP}]
        if self.data.get("last_selected") == name:
            self.data["last_selected"] = self.data["presets"][0]["name"]
        self.save()
        return True

    def set_last_selected(self, name: str | None):
        self.data["last_selected"] = name
        self.save()


@dataclass
class Metrics:
    elapsed_s: float = 0.0
    wpm: float = 0.0
    accuracy: float = 100.0
    errors: int = 0
    typed: int = 0
    total: int = 0
    progress: float = 0.0  # 0~100


class TypingEditor(QsciScintilla):
    """
    Typing practice editor:
    - Blocks paste to preserve practice integrity.
    - In strict mode, blocks cursor navigation and forces cursor to end.
    """
    def __init__(self, host=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._host = host

    def _strict_enabled(self) -> bool:
        return bool(self._host and self._host.is_strict_mode())

    def _force_cursor_end(self):
        if not self._host:
            return
        text = (self.text() or "").replace("\r\n", "\n")
        self._host._set_cursor_to_end(self, text)

    def keyPressEvent(self, event):
        # Block paste (Ctrl+V, Shift+Insert)
        if event.matches(QKeySequence.Paste):
            QMessageBox.information(self, "Paste blocked", "Paste is blocked for typing practice.")
            return
        if (event.modifiers() & Qt.ShiftModifier) and event.key() == Qt.Key_Insert:
            QMessageBox.information(self, "Paste blocked", "Paste is blocked for typing practice.")
            return

        if self._strict_enabled():
            nav_keys = {
                Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down,
                Qt.Key_Home, Qt.Key_End, Qt.Key_PageUp, Qt.Key_PageDown,
            }
            if event.key() in nav_keys:
                return
            self._force_cursor_end()

        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self._strict_enabled():
            self._force_cursor_end()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._strict_enabled():
            self._force_cursor_end()


class MainWindow(QMainWindow):
    # Indicator IDs
    IND_CORRECT = 0
    IND_WRONG = 1
    IND_CURSOR = 2

    def __init__(self):
        super().__init__()
        self.setWindowTitle("C++ Typing Trainer (PyQt5 + QScintilla)")
        self.resize(1200, 760)

        self.settings = QSettings("PoC_Algo_Typer", "CppTypingTrainer")
        self.pstore = PresetStore()

        self.target_text = DEFAULT_CPP.replace("\r\n", "\n")
        self.target_name = "Default"

        self.start_time = None
        self.running = False
        self._ignore_textchange = False
        self._prev_first_mismatch = None

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._on_tick)

        self._build_ui()

        # load last selected preset if exists
        last = self.pstore.data.get("last_selected")
        if last and self.pstore.get_by_name(last):
            p = self.pstore.get_by_name(last)
            self._apply_target(p.get("text", ""), name=p.get("name", "Preset"), from_preset=True)
            self._select_preset_in_list(last)
        else:
            self._apply_target(self.target_text, name=self.target_name, from_preset=True)
            self._select_preset_in_list("Default")

        self._reset()
        self._restore_ui_settings()
        self._refresh_preset_list(select_name=self.pstore.data.get("last_selected"))

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QWidget(self)
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # Top controls
        top = QHBoxLayout()
        outer.addLayout(top)

        self.btn_load = QPushButton("Load .txt")
        self.btn_default = QPushButton("Use Default")
        self.btn_start = QPushButton("Start")
        self.btn_reset = QPushButton("Reset")

        self.chk_strict = QCheckBox("Strict Mode")
        self.chk_beep = QCheckBox("Beep on Error")

        self.btn_load.clicked.connect(self._load_txt)
        self.btn_default.clicked.connect(self._use_default_preset)
        self.btn_start.clicked.connect(self._start)
        self.btn_reset.clicked.connect(self._reset)

        self.chk_strict.toggled.connect(self._on_option_changed)
        self.chk_beep.toggled.connect(self._on_option_changed)

        top.addWidget(self.btn_load)
        top.addWidget(self.btn_default)
        top.addSpacing(12)
        top.addWidget(self.chk_strict)
        top.addWidget(self.chk_beep)
        top.addStretch(1)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_reset)

        # Target label
        self.lbl_target = QLabel("Target: (not loaded)")
        self.lbl_target.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.lbl_target)

        # Metrics row
        met = QHBoxLayout()
        outer.addLayout(met)

        self.lbl_elapsed = QLabel("Elapsed: 0.0s")
        self.lbl_wpm = QLabel("WPM: 0.0")
        self.lbl_acc = QLabel("Accuracy: 100.0%")
        self.lbl_err = QLabel("Errors: 0")
        self.lbl_prog = QLabel("Progress: 0.0%")
        self.lbl_pos = QLabel("Pos: 0/0 (L1:C1)")

        for w in (self.lbl_elapsed, self.lbl_wpm, self.lbl_acc, self.lbl_err, self.lbl_prog, self.lbl_pos):
            w.setMinimumWidth(150)
            met.addWidget(w)
        met.addStretch(1)

        # Main horizontal splitter: [Presets] | [Editors]
        self.main_splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(self.main_splitter, stretch=1)

        # Left presets panel
        self.preset_panel = QWidget()
        self.preset_panel.setMinimumWidth(240)
        pvl = QVBoxLayout(self.preset_panel)
        pvl.setContentsMargins(8, 8, 8, 8)
        pvl.setSpacing(8)

        lbl = QLabel("Presets")
        lbl.setStyleSheet("font-weight: 600;")
        pvl.addWidget(lbl)

        self.list_presets = QListWidget()
        self.list_presets.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_presets.itemSelectionChanged.connect(self._on_preset_selected)
        pvl.addWidget(self.list_presets, stretch=1)

        pb = QHBoxLayout()
        self.btn_p_add = QPushButton("Add")
        self.btn_p_rename = QPushButton("Rename")
        self.btn_p_del = QPushButton("Delete")

        self.btn_p_add.clicked.connect(self._preset_add_current)
        self.btn_p_rename.clicked.connect(self._preset_rename)
        self.btn_p_del.clicked.connect(self._preset_delete)

        pb.addWidget(self.btn_p_add)
        pb.addWidget(self.btn_p_rename)
        pb.addWidget(self.btn_p_del)
        pvl.addLayout(pb)

        hint = QLabel("Tip: Load .txt 후 Add로 Preset 저장 가능")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        pvl.addWidget(hint)

        # A separator look (optional)
        self.preset_panel.setFrameStyle(QFrame.StyledPanel) if isinstance(self.preset_panel, QFrame) else None

        self.main_splitter.addWidget(self.preset_panel)

        # Editors splitter (source/input)
        self.editor_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.editor_splitter)
        self.main_splitter.setSizes([260, 940])

        # Source editor (read-only)
        self.src = QsciScintilla()
        self._setup_editor(self.src, readonly=True)
        self.editor_splitter.addWidget(self.src)

        # Input editor
        self.inp = TypingEditor(host=self)
        self._setup_editor(self.inp, readonly=False)
        self.editor_splitter.addWidget(self.inp)

        self.editor_splitter.setSizes([560, 540])

        # Signals
        self.inp.textChanged.connect(self._on_input_changed)

    def _setup_editor(self, ed: QsciScintilla, readonly: bool):
        font = QFont("Consolas", 11)
        ed.setUtf8(True)
        ed.setFont(font)
        ed.setMarginsFont(font)

        ed.setMarginType(0, QsciScintilla.NumberMargin)
        ed.setMarginWidth(0, "00000")
        ed.setMarginsForegroundColor(Qt.darkGray)
        ed.setMarginsBackgroundColor(Qt.lightGray)

        ed.setIndentationsUseTabs(False)
        ed.setTabWidth(4)
        ed.setIndentationGuides(True)
        ed.setAutoIndent(True)
        ed.setBackspaceUnindents(True)

        ed.setBraceMatching(QsciScintilla.SloppyBraceMatch)
        ed.setCaretLineVisible(True)

        lexer = QsciLexerCPP(ed)
        lexer.setFont(font)
        ed.setLexer(lexer)

        ed.setAutoCompletionSource(QsciScintilla.AcsAll)
        ed.setAutoCompletionThreshold(2)
        ed.setAutoCompletionCaseSensitivity(False)
        ed.setAutoCompletionReplaceWord(True)

        ed.setReadOnly(readonly)

        ed.indicatorDefine(QsciScintilla.INDIC_STRAIGHTBOX, self.IND_CORRECT)
        ed.indicatorDefine(QsciScintilla.INDIC_SQUIGGLE, self.IND_WRONG)
        ed.indicatorDefine(QsciScintilla.INDIC_FULLBOX, self.IND_CURSOR)

    # ---------------- Options ----------------
    def is_strict_mode(self) -> bool:
        return self.chk_strict.isChecked()

    def is_beep_on_error(self) -> bool:
        return self.chk_beep.isChecked()

    def _on_option_changed(self):
        self._save_ui_settings()
        if self.is_strict_mode():
            typed = (self.inp.text() or "").replace("\r\n", "\n")
            self._set_cursor_to_end(self.inp, typed)

    def _save_ui_settings(self):
        self.settings.setValue("strict_mode", self.chk_strict.isChecked())
        self.settings.setValue("beep_on_error", self.chk_beep.isChecked())

    def _restore_ui_settings(self):
        strict = self.settings.value("strict_mode", False, type=bool)
        beep = self.settings.value("beep_on_error", True, type=bool)
        self.chk_strict.setChecked(strict)
        self.chk_beep.setChecked(beep)

    # ---------------- Presets UI ----------------
    def _refresh_preset_list(self, select_name: str | None = None):
        self.list_presets.blockSignals(True)
        try:
            self.list_presets.clear()
            for name in self.pstore.list_names():
                self.list_presets.addItem(QListWidgetItem(name))
        finally:
            self.list_presets.blockSignals(False)

        if select_name:
            self._select_preset_in_list(select_name)

    def _select_preset_in_list(self, name: str):
        for i in range(self.list_presets.count()):
            if self.list_presets.item(i).text() == name:
                self.list_presets.setCurrentRow(i)
                return

    def _current_preset_name(self) -> str | None:
        items = self.list_presets.selectedItems()
        if not items:
            return None
        return items[0].text()

    def _on_preset_selected(self):
        name = self._current_preset_name()
        if not name:
            return
        p = self.pstore.get_by_name(name)
        if not p:
            return
        self.pstore.set_last_selected(name)
        self._apply_target(p.get("text", ""), name=name, from_preset=True)

    def _preset_add_current(self):
        # Add current target (source) as preset
        default_name = self.target_name if self.target_name else "NewPreset"
        name, ok = QInputDialog.getText(self, "Add Preset", "Preset name:", text=default_name)
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            return

        # Avoid duplicate: auto-suffix
        base = name
        idx = 2
        while self.pstore.get_by_name(name) is not None:
            name = f"{base} ({idx})"
            idx += 1

        self.pstore.upsert(name, self.target_text)
        self.pstore.set_last_selected(name)
        self._refresh_preset_list(select_name=name)

    def _preset_rename(self):
        old = self._current_preset_name()
        if not old:
            QMessageBox.information(self, "Rename", "Select a preset to rename.")
            return
        new, ok = QInputDialog.getText(self, "Rename Preset", "New name:", text=old)
        if not ok:
            return
        new = (new or "").strip()
        if not new:
            return
        if not self.pstore.rename(old, new):
            QMessageBox.warning(self, "Rename failed", "Name already exists or invalid.")
            return
        self._refresh_preset_list(select_name=new)
        # If the current target was that preset, update label context too
        if self.target_name == old:
            self.target_name = new
            self.lbl_target.setText(f"Target: {self.target_name}  |  {len(self.target_text)} chars")

    def _preset_delete(self):
        name = self._current_preset_name()
        if not name:
            QMessageBox.information(self, "Delete", "Select a preset to delete.")
            return
        if name == "Default":
            # allow delete but confirm stronger
            msg = "Delete 'Default' preset? (A Default will be recreated if none remain.)"
        else:
            msg = f"Delete preset '{name}'?"
        if QMessageBox.question(self, "Confirm delete", msg) != QMessageBox.Yes:
            return

        self.pstore.delete(name)
        select = self.pstore.data.get("last_selected") or (self.pstore.list_names()[0] if self.pstore.list_names() else None)
        self._refresh_preset_list(select_name=select)

    def _use_default_preset(self):
        self._select_preset_in_list("Default")
        p = self.pstore.get_by_name("Default")
        if p:
            self.pstore.set_last_selected("Default")
            self._apply_target(p.get("text", DEFAULT_CPP), name="Default", from_preset=True)

    # ---------------- Data / Flow ----------------
    def _apply_target(self, text: str, name: str, from_preset: bool = False):
        self.target_text = (text or "").replace("\r\n", "\n")
        self.target_name = name or ("Preset" if from_preset else "Loaded")
        suffix = " (Preset)" if from_preset else " (Loaded)"
        self.lbl_target.setText(f"Target: {self.target_name}{suffix}  |  {len(self.target_text)} chars")

        self._ignore_textchange = True
        try:
            self.src.setText(self.target_text)
            self.inp.setText("")
        finally:
            self._ignore_textchange = False

        self._reset()

    def _reset(self):
        self.timer.stop()
        self.running = False
        self.start_time = None
        self._prev_first_mismatch = None

        self._ignore_textchange = True
        try:
            self.inp.setText("")
        finally:
            self._ignore_textchange = False

        self._update_metrics(Metrics(total=len(self.target_text)))
        self._clear_indicators(self.src, len(self.target_text))
        self._clear_indicators(self.inp, 0)

        self._update_overlay_src(typed_text="")
        self._update_overlay_inp(typed_text="")

        self.btn_start.setEnabled(True)
        self.inp.setFocus()

    def _start(self):
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.timer.start()
        self.btn_start.setEnabled(False)
        self.inp.setFocus()

    def _load_txt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load target text",
            "",
            "Text Files (*.txt);;All Files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="cp949", errors="replace") as f:
                text = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Failed to load file:\n{e}")
            return

        name = os.path.basename(path)
        # Load는 프리셋 선택과 별개이므로 selection 해제(시각적으로도 "임시 로드" 느낌)
        self.list_presets.clearSelection()
        self._apply_target(text, name=name, from_preset=False)

    # ---------------- Typing / Metrics ----------------
    def _on_input_changed(self):
        if self._ignore_textchange:
            return

        typed = (self.inp.text() or "").replace("\r\n", "\n")

        # Auto-start when user begins typing
        if (not self.running) and typed:
            self._start()

        # Strict mode enforcement
        if self.is_strict_mode():
            new_typed = self._enforce_strict_typed(typed)
            if new_typed != typed:
                self._set_input_text_safely(new_typed)
                typed = new_typed
            self._set_cursor_to_end(self.inp, typed)

        self._update_overlay_src(typed_text=typed)
        self._update_overlay_inp(typed_text=typed)
        self._compute_and_update_metrics(typed)

        if typed == self.target_text and self.target_text:
            self.timer.stop()
            self.running = False
            self.btn_start.setEnabled(True)

    def _on_tick(self):
        typed = (self.inp.text() or "").replace("\r\n", "\n")
        self._compute_and_update_metrics(typed)

    def _compute_and_update_metrics(self, typed: str):
        target = self.target_text
        total = len(target)
        typed_len = len(typed)

        elapsed = 0.0
        if self.running and self.start_time is not None:
            elapsed = max(0.0, time.time() - self.start_time)

        overlap = min(typed_len, total)
        mismatches = 0
        first_mismatch = None
        for i in range(overlap):
            if typed[i] != target[i]:
                mismatches += 1
                if first_mismatch is None:
                    first_mismatch = i
        extra = max(0, typed_len - total)
        if extra > 0 and first_mismatch is None:
            first_mismatch = total
        mismatches += extra

        if self.is_beep_on_error():
            if self._prev_first_mismatch is None and first_mismatch is not None:
                QApplication.beep()
        self._prev_first_mismatch = first_mismatch

        if typed_len == 0:
            accuracy = 100.0
        else:
            accuracy = max(0.0, (typed_len - mismatches) / typed_len * 100.0)

        correct_prefix = 0
        for i in range(min(typed_len, total)):
            if typed[i] != target[i]:
                break
            correct_prefix += 1

        progress = 0.0 if total == 0 else (correct_prefix / total * 100.0)

        net_correct = max(0, typed_len - mismatches)
        if elapsed > 0.0:
            wpm = (net_correct / 5.0) / (elapsed / 60.0)
        else:
            wpm = 0.0

        m = Metrics(
            elapsed_s=elapsed,
            wpm=wpm,
            accuracy=accuracy,
            errors=mismatches,
            typed=typed_len,
            total=total,
            progress=progress,
        )
        self._update_metrics(m)

        pos = min(correct_prefix, max(0, total))
        line, col = self._index_to_line_col(target, pos)
        self.lbl_pos.setText(f"Pos: {pos}/{total} (L{line}:C{col})")

    def _update_metrics(self, m: Metrics):
        self.lbl_elapsed.setText(f"Elapsed: {m.elapsed_s:.1f}s")
        self.lbl_wpm.setText(f"WPM: {m.wpm:.1f}")
        self.lbl_acc.setText(f"Accuracy: {m.accuracy:.1f}%")
        self.lbl_err.setText(f"Errors: {m.errors}")
        self.lbl_prog.setText(f"Progress: {m.progress:.1f}%")

    # ---------------- Strict mode enforcement ----------------
    def _enforce_strict_typed(self, typed: str) -> str:
        target = self.target_text
        total = len(target)
        if total <= 0:
            return ""

        if len(typed) > total:
            typed = typed[:total]

        overlap = min(len(typed), total)
        fm = None
        for i in range(overlap):
            if typed[i] != target[i]:
                fm = i
                break

        if fm is not None:
            max_len = min(total, fm + 1)
            if len(typed) > max_len:
                typed = typed[:max_len]

        return typed

    def _set_input_text_safely(self, text: str):
        self._ignore_textchange = True
        try:
            self.inp.setText(text)
        finally:
            self._ignore_textchange = False

    @staticmethod
    def _set_cursor_to_end(ed: QsciScintilla, text: str):
        text = (text or "").replace("\r\n", "\n")
        lines = text.split("\n")
        line = max(0, len(lines) - 1)
        col = len(lines[-1]) if lines else 0
        ed.setCursorPosition(line, col)
        ed.ensureCursorVisible()

    # ---------------- Overlay / Indicators ----------------
    def _clear_indicators(self, ed: QsciScintilla, length: int):
        length = max(0, int(length))
        if length <= 0:
            length = 1
        for ind in (self.IND_CORRECT, self.IND_WRONG, self.IND_CURSOR):
            ed.SendScintilla(ed.SCI_SETINDICATORCURRENT, ind)
            ed.SendScintilla(ed.SCI_INDICATORCLEARRANGE, 0, length)

    def _fill_indicator(self, ed: QsciScintilla, ind_id: int, start: int, length: int):
        if length <= 0:
            return
        ed.SendScintilla(ed.SCI_SETINDICATORCURRENT, ind_id)
        ed.SendScintilla(ed.SCI_INDICATORFILLRANGE, start, length)

    def _update_overlay_src(self, typed_text: str):
        target = self.target_text
        tlen = len(target)
        plen = min(len(typed_text), tlen)

        self._clear_indicators(self.src, tlen)
        if tlen == 0:
            return

        if plen > 0:
            i = 0
            while i < plen:
                is_ok = (typed_text[i] == target[i])
                j = i + 1
                while j < plen and (typed_text[j] == target[j]) == is_ok:
                    j += 1
                if is_ok:
                    self._fill_indicator(self.src, self.IND_CORRECT, i, j - i)
                else:
                    self._fill_indicator(self.src, self.IND_WRONG, i, j - i)
                i = j

        correct_prefix = 0
        for i in range(plen):
            if typed_text[i] != target[i]:
                break
            correct_prefix += 1

        cursor_pos = min(correct_prefix, max(0, tlen - 1))
        self._fill_indicator(self.src, self.IND_CURSOR, cursor_pos, 1)

        line, _col = self.src.lineIndexFromPosition(cursor_pos)
        self.src.setFirstVisibleLine(max(0, line - 5))

    def _update_overlay_inp(self, typed_text: str):
        target = self.target_text
        tlen = len(target)
        ilen = len(typed_text)

        self._clear_indicators(self.inp, max(ilen, 1))
        if ilen == 0:
            return

        plen = min(ilen, tlen)
        if plen > 0:
            i = 0
            while i < plen:
                is_ok = (typed_text[i] == target[i])
                j = i + 1
                while j < plen and (typed_text[j] == target[j]) == is_ok:
                    j += 1
                if is_ok:
                    self._fill_indicator(self.inp, self.IND_CORRECT, i, j - i)
                else:
                    self._fill_indicator(self.inp, self.IND_WRONG, i, j - i)
                i = j

        if ilen > tlen:
            self._fill_indicator(self.inp, self.IND_WRONG, tlen, ilen - tlen)

        fm = None
        for i in range(min(ilen, tlen)):
            if typed_text[i] != target[i]:
                fm = i
                break
        if fm is None and ilen > tlen:
            fm = tlen

        if fm is None:
            cursor_pos = max(0, ilen - 1)
        else:
            cursor_pos = max(0, min(fm, ilen - 1))

        self._fill_indicator(self.inp, self.IND_CURSOR, cursor_pos, 1)

    # ---------------- Utilities ----------------
    @staticmethod
    def _index_to_line_col(text: str, idx: int):
        idx = max(0, min(idx, len(text)))
        line = text.count("\n", 0, idx) + 1
        last_nl = text.rfind("\n", 0, idx)
        if last_nl < 0:
            col = idx + 1
        else:
            col = (idx - last_nl)
        return line, col


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
