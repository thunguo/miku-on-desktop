"""``RecollectionGalleryPanel`` 的渲染顺序/搜索/详情弹窗回归测试。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from miku_on_desk.brain.memory.models import Episode
from miku_on_desk.brain.memory.system import MemorySystem
from miku_on_desk.face.ui import recollection_gallery
from miku_on_desk.face.ui.recollection_gallery import (
    RecollectionCard,
    RecollectionGalleryPanel,
    _format_episode_detail,
)


def test_reload_shows_empty_state_message_when_no_recollections(
    qapp: QApplication, tmp_path: Path
) -> None:
    memory_system = MemorySystem(tmp_path / "memory")
    panel = RecollectionGalleryPanel(memory_system)
    panel.show()

    assert panel._empty_label.isVisibleTo(panel) is True
    assert panel._grid.count() == 1


def test_reload_hides_empty_state_message_when_recollections_exist(
    qapp: QApplication, tmp_path: Path
) -> None:
    memory_system = MemorySystem(tmp_path / "memory")
    memory_system.episodic.append_event(
        title="搬家", summary="搬到了上海。", occurred_at="2026-07-01T09:00:00+00:00"
    )
    panel = RecollectionGalleryPanel(memory_system)
    panel.show()

    assert panel._empty_label.isVisibleTo(panel) is False


def test_reload_renders_cards_with_newest_recollection_first(
    qapp: QApplication, tmp_path: Path
) -> None:
    memory_system = MemorySystem(tmp_path / "memory")
    older_id = memory_system.episodic.append_event(
        title="较早的回忆", summary="摘要一", occurred_at="2026-07-01T09:00:00+00:00"
    )
    newer_id = memory_system.episodic.append_event(
        title="较新的回忆", summary="摘要二", occurred_at="2026-07-05T09:00:00+00:00"
    )
    panel = RecollectionGalleryPanel(memory_system)

    first_card = panel._grid.itemAtPosition(0, 0).widget()
    second_card = panel._grid.itemAtPosition(0, 1).widget()
    assert isinstance(first_card, RecollectionCard)
    assert isinstance(second_card, RecollectionCard)
    assert first_card._event_id == newer_id
    assert second_card._event_id == older_id


def test_search_filters_to_matching_recollection_and_reset_restores_all(
    qapp: QApplication, tmp_path: Path
) -> None:
    memory_system = MemorySystem(tmp_path / "memory")
    memory_system.episodic.append_event(
        title="搬家到上海", summary="摘要一", occurred_at="2026-07-01T09:00:00+00:00"
    )
    memory_system.episodic.append_event(
        title="养了一只猫", summary="摘要二", occurred_at="2026-07-02T09:00:00+00:00"
    )
    panel = RecollectionGalleryPanel(memory_system)

    panel._search_edit.setText("上海")
    panel._on_search_clicked()

    cards = panel.findChildren(RecollectionCard)
    assert len(cards) == 1

    panel._search_edit.setText("")
    panel._reload()

    cards = panel.findChildren(RecollectionCard)
    assert len(cards) == 2


def test_click_card_shows_detail_via_message_box(
    qapp: QApplication, tmp_path: Path, monkeypatch: object
) -> None:
    memory_system = MemorySystem(tmp_path / "memory")
    memory_system.episodic.append_event(
        title="搬家", summary="搬到了上海。", occurred_at="2026-07-01T09:00:00+00:00"
    )
    panel = RecollectionGalleryPanel(memory_system)

    captured: list[tuple[str, str]] = []

    class _FakeMessageBox:
        def __init__(self, title: str, content: str, parent: object) -> None:
            captured.append((title, content))

        def exec(self) -> int:
            return 1

    monkeypatch.setattr(recollection_gallery, "MessageBox", _FakeMessageBox)  # type: ignore[attr-defined]

    card = panel.findChild(RecollectionCard)
    assert card is not None
    card.mouseReleaseEvent(None)

    assert len(captured) == 1
    title, content = captured[0]
    assert title == "回忆详情"
    assert "搬家" in content
    assert "搬到了上海" in content


def test_format_episode_detail_omits_empty_optional_fields() -> None:
    episode = Episode(
        id="E:001",
        month="2026-07",
        title="标题",
        occurred_at="2026-07-01T09:00:00+00:00",
        summary="摘要文本",
    )

    detail = _format_episode_detail(episode)

    assert "标题" in detail
    assert "摘要文本" in detail
    assert "心情" not in detail
    assert "参与者" not in detail
    assert "事件链" not in detail
    assert "关联事件" not in detail


def test_format_episode_detail_includes_populated_optional_fields() -> None:
    episode = Episode(
        id="E:002",
        month="2026-07",
        title="标题",
        occurred_at="2026-07-01T09:00:00+00:00",
        summary="摘要文本",
        emotion_tag="开心",
        participants=["小白", "小黑"],
        event_chain=["先做了A", "然后做了B"],
        related_events=["E:001"],
    )

    detail = _format_episode_detail(episode)

    assert "心情：开心" in detail
    assert "小白、小黑" in detail
    assert "先做了A" in detail
    assert "关联事件：E:001" in detail
