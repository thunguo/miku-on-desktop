"""回忆相册面板：把 episodic 记忆渲染成可翻阅的卡片式画廊，点击卡片查看完整详情。

新建独立面板而不是原地升级 ``memory_panel.py`` 的 episodic 标签页——那里是运维/编辑向的
三级树控件，承担"编辑标题/摘要"的心智模型，跟这里"翻一翻回忆"的沉浸式浏览场景不同。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    LineEdit,
    MessageBox,
    PushButton,
    StrongBodyLabel,
)

from miku_on_desk.brain.memory.models import Episode
from miku_on_desk.brain.memory.system import MemorySystem
from miku_on_desk.face.ui.theme import (
    HOVER_COLOR,
    SPACING_LG,
    SPACING_MD,
    TEAL_MAIN,
    border_qss,
)

_COLUMNS = 3
_SUMMARY_PREVIEW_CHARS = 80


class RecollectionCard(QWidget):
    """单张回忆卡片：标题 + 时间 + 情感/参与者标签 + 摘要预览，点击后发出事件 id。"""

    clicked = Signal(str)

    def __init__(self, episode: Episode, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._event_id = episode.id
        self.setFixedSize(200, 160)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._idle_style = border_qss(TEAL_MAIN)
        self._hover_style = border_qss(HOVER_COLOR)
        self.setStyleSheet(f"RecollectionCard {{ {self._idle_style} }}")

        layout = QVBoxLayout(self)
        title_label = StrongBodyLabel(episode.title, self)
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        date_label = CaptionLabel(_format_occurred_at(episode.occurred_at), self)
        layout.addWidget(date_label)

        if episode.emotion_tag:
            tag_label = CaptionLabel(f"心情：{episode.emotion_tag}", self)
            layout.addWidget(tag_label)

        if episode.participants:
            participants_label = CaptionLabel(f"参与者：{'、'.join(episode.participants)}", self)
            participants_label.setWordWrap(True)
            layout.addWidget(participants_label)

        summary_label = BodyLabel(_truncate(episode.summary, _SUMMARY_PREVIEW_CHARS), self)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)
        layout.addStretch()

    def enterEvent(self, event: object) -> None:
        del event
        self.setStyleSheet(f"RecollectionCard {{ {self._hover_style} }}")

    def leaveEvent(self, event: object) -> None:
        del event
        self.setStyleSheet(f"RecollectionCard {{ {self._idle_style} }}")

    def mouseReleaseEvent(self, event: object) -> None:
        del event
        self.clicked.emit(self._event_id)


class RecollectionGalleryPanel(QWidget):
    """搜索/浏览 ``memory_system.episodic`` 的事件，点击卡片查看完整详情。"""

    def __init__(self, memory_system: MemorySystem, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._memory_system = memory_system
        self.resize(720, 560)

        outer = QVBoxLayout(self)

        self._search_edit = LineEdit(self)
        self._search_edit.setPlaceholderText("搜索回忆…")
        search_button = PushButton("搜索", self)
        reset_button = PushButton("显示全部", self)
        search_button.clicked.connect(self._on_search_clicked)
        reset_button.clicked.connect(self._reload)
        search_row = QHBoxLayout()
        search_row.addWidget(self._search_edit)
        search_row.addWidget(search_button)
        search_row.addWidget(reset_button)
        outer.addLayout(search_row)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        self._grid_container = QWidget(scroll)
        self._grid = QGridLayout(self._grid_container)
        self._grid.setContentsMargins(SPACING_LG, SPACING_LG, SPACING_LG, SPACING_LG)
        self._grid.setSpacing(SPACING_MD)
        scroll.setWidget(self._grid_container)

        self._reload()

    def _reload(self) -> None:
        episodes = list(reversed(self._memory_system.episodic.list_events()))
        self._render(episodes)

    def _on_search_clicked(self) -> None:
        query = self._search_edit.text().strip()
        if not query:
            self._reload()
            return
        self._render(self._memory_system.episodic.search(query))

    def _render(self, episodes: list[Episode]) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self._empty_label = CaptionLabel("还没有回忆，多和 Miku 聊聊天吧", self._grid_container)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if not episodes:
            self._grid.addWidget(
                self._empty_label, 0, 0, 1, _COLUMNS, alignment=Qt.AlignmentFlag.AlignCenter
            )
            return
        self._empty_label.hide()

        for index, episode in enumerate(episodes):
            card = RecollectionCard(episode, self._grid_container)
            card.clicked.connect(self._show_detail)
            self._grid.addWidget(card, index // _COLUMNS, index % _COLUMNS)

    def _show_detail(self, event_id: str) -> None:
        episode = self._memory_system.episodic.get_event(event_id)
        if episode is None:
            return
        box = MessageBox("回忆详情", _format_episode_detail(episode), self)
        box.exec()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}…"


def _format_occurred_at(occurred_at: str) -> str:
    return occurred_at[:16].replace("T", " ")


def _format_episode_detail(episode: Episode) -> str:
    lines = [f"标题：{episode.title}", f"发生时间：{episode.occurred_at}"]
    if episode.summary:
        lines.append(f"摘要：{episode.summary}")
    if episode.emotion_tag:
        lines.append(f"心情：{episode.emotion_tag}")
    if episode.participants:
        lines.append(f"参与者：{'、'.join(episode.participants)}")
    if episode.event_chain:
        lines.append("事件链：\n" + "\n".join(f"- {item}" for item in episode.event_chain))
    if episode.related_events:
        lines.append(f"关联事件：{'、'.join(episode.related_events)}")
    return "\n".join(lines)
