"""
C++ Typing Trainer (PyQt5 + QScintilla)

Version: 0.1.1  (2025-12-20)
Versioning: MAJOR.MINOR.PATCH (SemVer)
- MAJOR: 호환성 깨지는 변경
- MINOR: 기능 추가(호환 유지)
- PATCH: 버그 수정/리팩토링 등 소규모 개선

Release Notes (v0.1.1):
- (Fix) main guard + app.exec_() 보장: "UI가 안 뜸" 류의 실행 문제 방지
- (Improve) Source 영역에 indicator overlay (correct / wrong / cursor) 적용
- (Improve) 입력 시작 시 자동으로 타이머 시작(선택적으로 Start 버튼도 유지)
- (Improve) Metrics 계산 안정화(WPM/Accuracy/Errors/Progress)
- (Maintain) Paste 차단 유지
"""

import sys
import time
from dataclasses import dataclass

from PyQt5.QtCore import Qt, QTimer
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
    """Typing practice editor: blocks paste to preserve practice integrity."""

    def keyPressEvent(self, event):
        # Block paste (Ctrl+V, Shift+Insert)
        if event.matches(QKeySequence.Paste):
            QMessageBox.information(self, "Paste blocked", "Paste is blocked for typing practice.")
            return
        if (event.modifiers() & Qt.ShiftModifier) and event.key() == Qt.Key_Insert:
            QMessageBox.information(self, "Paste blocked", "Paste is blocked for typing practice.")
            return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    # Indicator IDs
    IND_CORRECT = 0
    IND_WRONG = 1
    IND_CURSOR = 2

    def __init__(self):
        super().__init__()
        self.setWindowTitle("C++ Typing Trainer (PyQt5 + QScintilla)")
        self.resize(1100, 720)

        self.target_text = DEFAULT_CPP.replace("\r\n", "\n")
        self.target_name = "Default"

        self.start_time = None
        self.running = False
        self._ignore_textchange = False  # guard while programmatic updates

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._apply_target(self.target_text, name=self.target_name)
        self._reset()

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

        self.btn_load.clicked.connect(self._load_txt)
        self.btn_default.clicked.connect(lambda: self._apply_target(DEFAULT_CPP, "Default"))
        self.btn_start.clicked.connect(self._start)
        self.btn_reset.clicked.connect(self._reset)

        top.addWidget(self.btn_load)
        top.addWidget(self.btn_default)
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

        # Editors
        self.splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(self.splitter, stretch=1)

        # Source editor (read-only)
        self.src = QsciScintilla()
        self._setup_editor(self.src, readonly=True)
        self.splitter.addWidget(self.src)

        # Input editor (typing)
        self.inp = TypingEditor()
        self._setup_editor(self.inp, readonly=False)
        self.splitter.addWidget(self.inp)

        self.splitter.setSizes([560, 540])

        # Signals
        self.inp.textChanged.connect(self._on_input_changed)

    def _setup_editor(self, ed: QsciScintilla, readonly: bool):
        font = QFont("Consolas", 11)
        ed.setUtf8(True)
        ed.setFont(font)
        ed.setMarginsFont(font)

        # Line numbers margin
        ed.setMarginType(0, QsciScintilla.NumberMargin)
        ed.setMarginWidth(0, "00000")
        ed.setMarginsForegroundColor(Qt.darkGray)
        ed.setMarginsBackgroundColor(Qt.lightGray)

        # Indentation & tabs
        ed.setIndentationsUseTabs(False)
        ed.setTabWidth(4)
        ed.setIndentationGuides(True)
        ed.setAutoIndent(True)
        ed.setBackspaceUnindents(True)

        # Brace matching
        ed.setBraceMatching(QsciScintilla.SloppyBraceMatch)

        # Caret line highlight
        ed.setCaretLineVisible(True)

        # Lexer
        lexer = QsciLexerCPP(ed)
        lexer.setFont(font)
        ed.setLexer(lexer)

        # Autocomplete (document + API keywords)
        ed.setAutoCompletionSource(QsciScintilla.AcsAll)
        ed.setAutoCompletionThreshold(2)
        ed.setAutoCompletionCaseSensitivity(False)
        ed.setAutoCompletionReplaceWord(True)

        ed.setReadOnly(readonly)

        # Indicators (only meaningful for source, but harmless elsewhere)
        # Styles: use Scintilla built-in indicator styles.
        # We do not hard-code colors; Scintilla default is fine. If you want custom colors later,
        # we can expose them as UI options.
        ed.indicatorDefine(QsciScintilla.INDIC_STRAIGHTBOX, self.IND_CORRECT)
        ed.indicatorDefine(QsciScintilla.INDIC_SQUIGGLE, self.IND_WRONG)
        ed.indicatorDefine(QsciScintilla.INDIC_FULLBOX, self.IND_CURSOR)

    # ---------------- Data / Flow ----------------
    def _apply_target(self, text: str, name: str):
        self.target_text = (text or "").replace("\r\n", "\n")
        self.target_name = name or "Loaded"
        self.lbl_target.setText(f"Target: {self.target_name}  |  {len(self.target_text)} chars")

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

        self._ignore_textchange = True
        try:
            self.inp.setText("")
        finally:
            self._ignore_textchange = False

        self._update_metrics(Metrics(total=len(self.target_text)))
        self._clear_indicators()
        self._update_overlay(typed_text="")

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
            # fallback for common Windows encodings
            with open(path, "r", encoding="cp949", errors="replace") as f:
                text = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Failed to load file:\n{e}")
            return

        name = path.split("/")[-1].split("\\")[-1]
        self._apply_target(text, name=name)

    # ---------------- Typing / Metrics ----------------
    def _on_input_changed(self):
        if self._ignore_textchange:
            return

        typed = (self.inp.text() or "").replace("\r\n", "\n")

        # Auto-start when user begins typing
        if (not self.running) and typed:
            self._start()

        self._update_overlay(typed_text=typed)
        self._compute_and_update_metrics(typed)

        # Stop when done
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

        # errors: per-char mismatches in overlap + any extra typed beyond target
        overlap = min(typed_len, total)
        mismatches = 0
        for i in range(overlap):
            if typed[i] != target[i]:
                mismatches += 1
        mismatches += max(0, typed_len - total)

        # accuracy
        if typed_len == 0:
            accuracy = 100.0
        else:
            accuracy = max(0.0, (typed_len - mismatches) / typed_len * 100.0)

        # progress: strict prefix correctness (first mismatch stops progress)
        correct_prefix = 0
        for i in range(min(typed_len, total)):
            if typed[i] != target[i]:
                break
            correct_prefix += 1

        progress = 0.0 if total == 0 else (correct_prefix / total * 100.0)

        # WPM: based on net correct chars (typed - errors)
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

        # Position info (use prefix position as "current expected")
        pos = min(correct_prefix, max(0, total))
        line, col = self._index_to_line_col(target, pos)
        self.lbl_pos.setText(f"Pos: {pos}/{total} (L{line}:C{col})")

    def _update_metrics(self, m: Metrics):
        self.lbl_elapsed.setText(f"Elapsed: {m.elapsed_s:.1f}s")
        self.lbl_wpm.setText(f"WPM: {m.wpm:.1f}")
        self.lbl_acc.setText(f"Accuracy: {m.accuracy:.1f}%")
        self.lbl_err.setText(f"Errors: {m.errors}")
        self.lbl_prog.setText(f"Progress: {m.progress:.1f}%")

    # ---------------- Overlay / Indicators ----------------
    def _clear_indicators(self):
        # Clear on source editor for full range
        length = len(self.target_text)
        if length <= 0:
            return
        for ind in (self.IND_CORRECT, self.IND_WRONG, self.IND_CURSOR):
            self.src.SendScintilla(self.src.SCI_SETINDICATORCURRENT, ind)
            self.src.SendScintilla(self.src.SCI_INDICATORCLEARRANGE, 0, length)

    def _update_overlay(self, typed_text: str):
        target = self.target_text
        tlen = len(target)
        plen = min(len(typed_text), tlen)

        self._clear_indicators()
        if tlen == 0:
            return

        # Mark per-run segments for correct / wrong within typed overlap
        # (Contiguous ranges are more efficient than per-char fill)
        def fill(ind_id: int, start: int, length: int):
            if length <= 0:
                return
            self.src.SendScintilla(self.src.SCI_SETINDICATORCURRENT, ind_id)
            self.src.SendScintilla(self.src.SCI_INDICATORFILLRANGE, start, length)

        if plen > 0:
            i = 0
            while i < plen:
                is_ok = (typed_text[i] == target[i])
                j = i + 1
                while j < plen and (typed_text[j] == target[j]) == is_ok:
                    j += 1
                if is_ok:
                    fill(self.IND_CORRECT, i, j - i)
                else:
                    fill(self.IND_WRONG, i, j - i)
                i = j

        # Cursor indicator: show expected position (first mismatch position = prefix)
        correct_prefix = 0
        for i in range(plen):
            if typed_text[i] != target[i]:
                break
            correct_prefix += 1

        cursor_pos = min(correct_prefix, tlen - 1)
        fill(self.IND_CURSOR, cursor_pos, 1)

        # Keep source view aligned roughly with cursor
        line, _col = self.src.lineIndexFromPosition(cursor_pos)
        self.src.setFirstVisibleLine(max(0, line - 5))

    # ---------------- Utilities ----------------
    @staticmethod
    def _index_to_line_col(text: str, idx: int):
        """
        Convert 0-based character index into 1-based (line, col).
        """
        idx = max(0, min(idx, len(text)))
        # line count by '\n'
        line = text.count("\n", 0, idx) + 1
        last_nl = text.rfind("\n", 0, idx)
        if last_nl < 0:
            col = idx + 1
        else:
            col = (idx - last_nl)
        return line, col


def main():
    # Ensure Qt app + event loop is started properly
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
