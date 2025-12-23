# -*- coding: utf-8 -*-
"""
C++ Typing Trainer (PyQt5 + QScintilla)

Version: 0.1.5  (2025-12-24)
Versioning: MAJOR.MINOR.PATCH (SemVer)

Release Notes (v0.1.5):
- (New) Dark Theme 지원 (Light/Dark 전환 + 설정 유지)
  - Qt 전역 팔레트(Fusion) + QScintilla/lexer 색상 동기화
  - margin, caret line, selection, indicator 색상까지 적용
- (Maintain) Preset reorder(Up/Down + Drag&Drop), Import/Export(Replace/Merge), Load .txt,
            Strict Mode, Beep, overlay, paste-block, metrics 유지
"""

import sys
import os
import json
import time
import tempfile
from dataclasses import dataclass

from PyQt5.QtCore import Qt, QTimer, QSettings, QStandardPaths
from PyQt5.QtGui import QFont, QKeySequence, QColor, QPalette
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
    QComboBox,
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
    folder = os.path.dirname(path)
    if folder:
        _ensure_dir(folder)

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="._tmp_", suffix=".tmp", dir=(folder if folder else None))
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
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _safe_write_json(path: str, data: dict):
    _safe_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _read_text_with_fallback(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp949", errors="replace") as f:
            return f.read()


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
            self._normalize()
            self.save()
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and "presets" in d:
                self.data = d
            else:
                self.data = {"version": 1, "presets": self._coerce_presets(d), "last_selected": None}
            self._normalize()
        except Exception:
            self.data = {"version": 1, "presets": [{"name": "Default", "text": DEFAULT_CPP}], "last_selected": "Default"}
            self.save()

    def save(self):
        self._normalize()
        _safe_write_json(self.path, self.data)

    def export_to(self, out_path: str):
        self._normalize()
        _safe_write_json(out_path, self.data)

    def import_from(self, in_path: str):
        raw = _read_text_with_fallback(in_path)
        obj = json.loads(raw)
        if isinstance(obj, dict) and "presets" in obj:
            presets = self._coerce_presets(obj.get("presets"))
            last = obj.get("last_selected")
            return {"version": 1, "presets": presets, "last_selected": last if isinstance(last, str) else None}
        presets = self._coerce_presets(obj)
        return {"version": 1, "presets": presets, "last_selected": None}

    def _coerce_presets(self, maybe) -> list:
        presets = []
        if isinstance(maybe, list):
            for it in maybe:
                if isinstance(it, dict) and "name" in it:
                    name = str(it.get("name", "")).strip()
                    text = str(it.get("text", ""))
                    if name:
                        presets.append({"name": name, "text": text})
                elif isinstance(it, str):
                    presets.append({"name": it[:24] if it.strip() else "Preset", "text": it})
        elif isinstance(maybe, dict):
            for k, v in maybe.items():
                name = str(k).strip()
                if name:
                    presets.append({"name": name, "text": str(v)})
        return presets

    def _normalize(self):
        if not isinstance(self.data, dict):
            self.data = {"version": 1, "presets": [{"name": "Default", "text": DEFAULT_CPP}], "last_selected": "Default"}

        if "presets" not in self.data or not isinstance(self.data["presets"], list):
            self.data["presets"] = []

        cleaned = []
        seen = set()
        for p in self.data["presets"]:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name", "")).strip()
            text = str(p.get("text", ""))
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            cleaned.append({"name": name, "text": text})

        if not any(p["name"] == "Default" for p in cleaned):
            cleaned.insert(0, {"name": "Default", "text": DEFAULT_CPP})

        self.data["presets"] = cleaned
        self.data.setdefault("version", 1)
        self.data.setdefault("last_selected", "Default")

        last = self.data.get("last_selected")
        if not isinstance(last, str) or last not in {p["name"] for p in cleaned}:
            self.data["last_selected"] = cleaned[0]["name"] if cleaned else "Default"

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
        self.data["presets"] = kept
        self._normalize()
        self.save()
        return True

    def set_last_selected(self, name: str | None):
        if name is None:
            return
        self.data["last_selected"] = name
        self.save()

    def reorder_by_names(self, ordered_names: list[str]):
        name_to_p = {p["name"]: p for p in self.data.get("presets", [])}
        new_list = []
        used = set()
        for n in ordered_names:
            if n in name_to_p and n not in used:
                new_list.append(name_to_p[n])
                used.add(n)
        for p in self.data.get("presets", []):
            if p["name"] not in used:
                new_list.append(p)
        self.data["presets"] = new_list
        self._normalize()
        self.save()

    def merge_in(self, imported_data: dict):
        imp_presets = imported_data.get("presets", [])
        for p in imp_presets:
            name = str(p.get("name", "")).strip()
            text = str(p.get("text", ""))
            if not name:
                continue
            final = name
            if self.get_by_name(final) is not None:
                base = final
                idx = 2
                while self.get_by_name(final) is not None:
                    final = f"{base} ({idx})"
                    idx += 1
            self.upsert(final, text)

        imp_last = imported_data.get("last_selected")
        if isinstance(imp_last, str) and self.get_by_name(imp_last):
            self.set_last_selected(imp_last)

    def replace_with(self, imported_data: dict):
        presets = imported_data.get("presets", [])
        self.data = {"version": 1, "presets": presets, "last_selected": imported_data.get("last_selected")}
        self._normalize()
        self.save()


@dataclass
class Metrics:
    elapsed_s: float = 0.0
    wpm: float = 0.0
    accuracy: float = 100.0
    errors: int = 0
    typed: int = 0
    total: int = 0
    progress: float = 0.0


class TypingEditor(QsciScintilla):
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
    IND_CORRECT = 0
    IND_WRONG = 1
    IND_CURSOR = 2

    def __init__(self):
        super().__init__()
        self.setWindowTitle("C++ Typing Trainer (PyQt5 + QScintilla)")
        self.resize(1280, 780)

        self.settings = QSettings("PoC_Algo_Typer", "CppTypingTrainer")
        self.pstore = PresetStore()

        self.target_text = DEFAULT_CPP.replace("\r\n", "\n")
        self.target_name = "Default"

        self.start_time = None
        self.running = False
        self._ignore_textchange = False
        self._prev_first_mismatch = None
        self._suppress_preset_apply = False

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._restore_ui_settings()

        # Apply theme after UI built (affects palette + editors)
        self._apply_theme(self._get_theme())

        self._refresh_preset_list(select_name=self.pstore.data.get("last_selected"))

        last = self.pstore.data.get("last_selected")
        p = self.pstore.get_by_name(last) if last else None
        if p:
            self._apply_target(p.get("text", ""), name=p.get("name", "Preset"), from_preset=True)
            self._select_preset_in_list(p.get("name"))
        else:
            self._apply_target(DEFAULT_CPP, name="Default", from_preset=True)
            self._select_preset_in_list("Default")

        self._reset()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QWidget(self)
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        top = QHBoxLayout()
        outer.addLayout(top)

        self.btn_load = QPushButton("Load .txt")
        self.btn_default = QPushButton("Use Default")
        self.btn_start = QPushButton("Start")
        self.btn_reset = QPushButton("Reset")

        self.chk_strict = QCheckBox("Strict Mode")
        self.chk_beep = QCheckBox("Beep on Error")

        # Theme selector
        self.cmb_theme = QComboBox()
        self.cmb_theme.addItems(["Light", "Dark"])
        self.cmb_theme.currentTextChanged.connect(self._on_theme_changed)

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
        top.addSpacing(12)
        top.addWidget(QLabel("Theme:"))
        top.addWidget(self.cmb_theme)
        top.addStretch(1)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_reset)

        self.lbl_target = QLabel("Target: (not loaded)")
        self.lbl_target.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.lbl_target)

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

        self.main_splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(self.main_splitter, stretch=1)

        # Left: Presets
        self.preset_panel = QWidget()
        pvl = QVBoxLayout(self.preset_panel)
        pvl.setContentsMargins(8, 8, 8, 8)
        pvl.setSpacing(8)

        lbl = QLabel("Presets")
        lbl.setStyleSheet("font-weight: 600;")
        pvl.addWidget(lbl)

        self.list_presets = QListWidget()
        self.list_presets.setSelectionMode(QAbstractItemView.SingleSelection)

        self.list_presets.setDragEnabled(True)
        self.list_presets.setAcceptDrops(True)
        self.list_presets.setDropIndicatorShown(True)
        self.list_presets.setDragDropMode(QAbstractItemView.InternalMove)

        self.list_presets.itemSelectionChanged.connect(self._on_preset_selected)
        pvl.addWidget(self.list_presets, stretch=1)

        rb = QHBoxLayout()
        self.btn_p_up = QPushButton("Up")
        self.btn_p_down = QPushButton("Down")
        self.btn_p_up.clicked.connect(self._preset_move_up)
        self.btn_p_down.clicked.connect(self._preset_move_down)
        rb.addWidget(self.btn_p_up)
        rb.addWidget(self.btn_p_down)
        pvl.addLayout(rb)

        cb = QHBoxLayout()
        self.btn_p_add = QPushButton("Add")
        self.btn_p_rename = QPushButton("Rename")
        self.btn_p_del = QPushButton("Delete")
        self.btn_p_add.clicked.connect(self._preset_add_current)
        self.btn_p_rename.clicked.connect(self._preset_rename)
        self.btn_p_del.clicked.connect(self._preset_delete)
        cb.addWidget(self.btn_p_add)
        cb.addWidget(self.btn_p_rename)
        cb.addWidget(self.btn_p_del)
        pvl.addLayout(cb)

        ib = QHBoxLayout()
        self.btn_p_import = QPushButton("Import")
        self.btn_p_export = QPushButton("Export")
        self.btn_p_import.clicked.connect(self._preset_import)
        self.btn_p_export.clicked.connect(self._preset_export)
        ib.addWidget(self.btn_p_import)
        ib.addWidget(self.btn_p_export)
        pvl.addLayout(ib)

        self.main_splitter.addWidget(self.preset_panel)

        # Right: editors
        self.editor_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.editor_splitter)
        self.main_splitter.setSizes([280, 1000])

        self.src = QsciScintilla()
        self._setup_editor(self.src, readonly=True)
        self.editor_splitter.addWidget(self.src)

        self.inp = TypingEditor(host=self)
        self._setup_editor(self.inp, readonly=False)
        self.editor_splitter.addWidget(self.inp)
        self.editor_splitter.setSizes([560, 540])

        self.inp.textChanged.connect(self._on_input_changed)

        # Persist drag-drop reorder
        try:
            self.list_presets.model().rowsMoved.connect(self._on_preset_rows_moved)
        except Exception:
            pass

    def _setup_editor(self, ed: QsciScintilla, readonly: bool):
        font = QFont("Consolas", 11)
        ed.setUtf8(True)
        ed.setFont(font)
        ed.setMarginsFont(font)

        ed.setMarginType(0, QsciScintilla.NumberMargin)
        ed.setMarginWidth(0, "00000")

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

    # ---------------- Theme ----------------
    def _get_theme(self) -> str:
        t = self.settings.value("theme", "Light")
        return t if t in ("Light", "Dark") else "Light"

    def _save_theme(self, theme: str):
        self.settings.setValue("theme", theme)

    def _on_theme_changed(self, theme: str):
        theme = theme if theme in ("Light", "Dark") else "Light"
        self._save_theme(theme)
        self._apply_theme(theme)

    def _apply_theme(self, theme: str):
        # Keep combobox in sync (avoid recursion)
        if self.cmb_theme.currentText() != theme:
            self.cmb_theme.blockSignals(True)
            try:
                self.cmb_theme.setCurrentText(theme)
            finally:
                self.cmb_theme.blockSignals(False)

        app = QApplication.instance()
        if not app:
            return

        # Use Fusion for consistent palette behavior
        try:
            app.setStyle("Fusion")
        except Exception:
            pass

        if theme == "Dark":
            pal = QPalette()
            pal.setColor(QPalette.Window, QColor(30, 30, 30))
            pal.setColor(QPalette.WindowText, QColor(220, 220, 220))
            pal.setColor(QPalette.Base, QColor(25, 25, 25))
            pal.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
            pal.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
            pal.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
            pal.setColor(QPalette.Text, QColor(220, 220, 220))
            pal.setColor(QPalette.Button, QColor(45, 45, 45))
            pal.setColor(QPalette.ButtonText, QColor(220, 220, 220))
            pal.setColor(QPalette.Highlight, QColor(70, 110, 160))
            pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            app.setPalette(pal)

            self._apply_scintilla_theme(
                paper=QColor(25, 25, 25),
                ink=QColor(220, 220, 220),
                margin_bg=QColor(45, 45, 45),
                margin_fg=QColor(180, 180, 180),
                caretline_bg=QColor(40, 40, 40),
                selection_bg=QColor(70, 110, 160),
                selection_fg=QColor(255, 255, 255),
                indic_correct=QColor(0, 200, 0),
                indic_wrong=QColor(255, 90, 90),
                indic_cursor=QColor(90, 160, 255),
            )
        else:
            app.setPalette(app.style().standardPalette())
            self._apply_scintilla_theme(
                paper=QColor(255, 255, 255),
                ink=QColor(0, 0, 0),
                margin_bg=QColor(235, 235, 235),
                margin_fg=QColor(80, 80, 80),
                caretline_bg=QColor(245, 245, 245),
                selection_bg=QColor(180, 210, 240),
                selection_fg=QColor(0, 0, 0),
                indic_correct=QColor(0, 140, 0),
                indic_wrong=QColor(220, 60, 60),
                indic_cursor=QColor(60, 120, 220),
            )

    def _apply_scintilla_theme(
        self,
        paper: QColor,
        ink: QColor,
        margin_bg: QColor,
        margin_fg: QColor,
        caretline_bg: QColor,
        selection_bg: QColor,
        selection_fg: QColor,
        indic_correct: QColor,
        indic_wrong: QColor,
        indic_cursor: QColor,
    ):
        for ed in (getattr(self, "src", None), getattr(self, "inp", None)):
            if ed is None:
                continue

            # Editor base colors
            ed.setPaper(paper)
            ed.setColor(ink)
            ed.setCaretLineBackgroundColor(caretline_bg)

            # Margin
            ed.setMarginsBackgroundColor(margin_bg)
            ed.setMarginsForegroundColor(margin_fg)

            # Selection
            ed.setSelectionBackgroundColor(selection_bg)
            ed.setSelectionForegroundColor(selection_fg)

            # Indicators: improve visibility on dark background
            try:
                ed.indicatorSetForegroundColor(indic_correct, self.IND_CORRECT)
                ed.indicatorSetForegroundColor(indic_wrong, self.IND_WRONG)
                ed.indicatorSetForegroundColor(indic_cursor, self.IND_CURSOR)
            except Exception:
                pass

            # Lexer colors (apply to many styles to avoid mixed defaults)
            lex = ed.lexer()
            if lex is not None:
                try:
                    lex.setDefaultPaper(paper)
                    lex.setDefaultColor(ink)
                    for style in range(0, 128):
                        lex.setPaper(paper, style)
                        lex.setColor(ink, style)
                except Exception:
                    pass

            # Force redraw
            try:
                ed.repaint()
            except Exception:
                pass

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
        theme = self._get_theme()
        self.chk_strict.setChecked(strict)
        self.chk_beep.setChecked(beep)
        self.cmb_theme.setCurrentText(theme)

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

    def _select_preset_in_list(self, name: str | None):
        if not name:
            return
        for i in range(self.list_presets.count()):
            if self.list_presets.item(i).text() == name:
                self.list_presets.setCurrentRow(i)
                return

    def _current_preset_name(self) -> str | None:
        items = self.list_presets.selectedItems()
        if not items:
            return None
        return items[0].text()

    def _sync_store_order_from_list(self):
        names = [self.list_presets.item(i).text() for i in range(self.list_presets.count())]
        self.pstore.reorder_by_names(names)

    def _on_preset_rows_moved(self, *args):
        cur = self._current_preset_name()
        self._sync_store_order_from_list()
        if cur:
            self._select_preset_in_list(cur)

    def _on_preset_selected(self):
        if self._suppress_preset_apply:
            return
        name = self._current_preset_name()
        if not name:
            return
        p = self.pstore.get_by_name(name)
        if not p:
            return
        self.pstore.set_last_selected(name)
        self._apply_target(p.get("text", ""), name=name, from_preset=True)

    def _preset_move_up(self):
        row = self.list_presets.currentRow()
        if row <= 0:
            return
        self._suppress_preset_apply = True
        try:
            item = self.list_presets.takeItem(row)
            self.list_presets.insertItem(row - 1, item)
            self.list_presets.setCurrentRow(row - 1)
            self._sync_store_order_from_list()
        finally:
            self._suppress_preset_apply = False

    def _preset_move_down(self):
        row = self.list_presets.currentRow()
        if row < 0 or row >= self.list_presets.count() - 1:
            return
        self._suppress_preset_apply = True
        try:
            item = self.list_presets.takeItem(row)
            self.list_presets.insertItem(row + 1, item)
            self.list_presets.setCurrentRow(row + 1)
            self._sync_store_order_from_list()
        finally:
            self._suppress_preset_apply = False

    def _preset_add_current(self):
        default_name = self.target_name if self.target_name else "NewPreset"
        name, ok = QInputDialog.getText(self, "Add Preset", "Preset name:", text=default_name)
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            return

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
        if self.target_name == old:
            self.target_name = new
            self.lbl_target.setText(f"Target: {self.target_name} (Preset)  |  {len(self.target_text)} chars")

    def _preset_delete(self):
        name = self._current_preset_name()
        if not name:
            QMessageBox.information(self, "Delete", "Select a preset to delete.")
            return
        msg = "Delete 'Default' preset? (Default will be recreated if none remain.)" if name == "Default" else f"Delete preset '{name}'?"
        if QMessageBox.question(self, "Confirm delete", msg) != QMessageBox.Yes:
            return
        self.pstore.delete(name)
        select = self.pstore.data.get("last_selected")
        self._refresh_preset_list(select_name=select)
        if select:
            p = self.pstore.get_by_name(select)
            if p:
                self._apply_target(p.get("text", ""), name=select, from_preset=True)

    def _preset_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Presets", "presets_export.json", "JSON Files (*.json);;All Files (*.*)")
        if not path:
            return
        try:
            self.pstore.export_to(path)
            QMessageBox.information(self, "Export", f"Exported presets to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"Failed to export:\n{e}")

    def _preset_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Presets", "", "JSON Files (*.json);;All Files (*.*)")
        if not path:
            return
        try:
            imported = self.pstore.import_from(path)
        except Exception as e:
            QMessageBox.critical(self, "Import failed", f"Failed to read JSON:\n{e}")
            return

        r = QMessageBox.question(
            self,
            "Import mode",
            "Replace existing presets?\n\nYes: Replace (overwrite all)\nNo: Merge (keep existing, add imported)",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No,
        )
        if r == QMessageBox.Cancel:
            return

        try:
            if r == QMessageBox.Yes:
                self.pstore.replace_with(imported)
            else:
                self.pstore.merge_in(imported)
        except Exception as e:
            QMessageBox.critical(self, "Import failed", f"Failed to import:\n{e}")
            return

        last = self.pstore.data.get("last_selected")
        self._refresh_preset_list(select_name=last)
        if last:
            p = self.pstore.get_by_name(last)
            if p:
                self._apply_target(p.get("text", ""), name=last, from_preset=True)

        QMessageBox.information(self, "Import", "Import completed.")

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
        path, _ = QFileDialog.getOpenFileName(self, "Load target text", "", "Text Files (*.txt);;All Files (*.*)")
        if not path:
            return
        try:
            text = _read_text_with_fallback(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Failed to load file:\n{e}")
            return

        name = os.path.basename(path)
        self.list_presets.clearSelection()
        self._apply_target(text, name=name, from_preset=False)

    # ---------------- Typing / Metrics ----------------
    def _on_input_changed(self):
        if self._ignore_textchange:
            return

        typed = (self.inp.text() or "").replace("\r\n", "\n")

        if (not self.running) and typed:
            self._start()

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

        accuracy = 100.0 if typed_len == 0 else max(0.0, (typed_len - mismatches) / typed_len * 100.0)

        correct_prefix = 0
        for i in range(min(typed_len, total)):
            if typed[i] != target[i]:
                break
            correct_prefix += 1

        progress = 0.0 if total == 0 else (correct_prefix / total * 100.0)

        net_correct = max(0, typed_len - mismatches)
        wpm = (net_correct / 5.0) / (elapsed / 60.0) if elapsed > 0.0 else 0.0

        m = Metrics(elapsed_s=elapsed, wpm=wpm, accuracy=accuracy, errors=mismatches, typed=typed_len, total=total, progress=progress)
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
                self._fill_indicator(self.src, self.IND_CORRECT if is_ok else self.IND_WRONG, i, j - i)
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
                self._fill_indicator(self.inp, self.IND_CORRECT if is_ok else self.IND_WRONG, i, j - i)
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

        cursor_pos = max(0, (min(fm, ilen - 1) if fm is not None else ilen - 1))
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
