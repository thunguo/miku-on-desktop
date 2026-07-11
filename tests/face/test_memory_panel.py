"""MemoryPanel 的回归测试：五个标签页（语义/情景/情感/原始会话/诊断）各自的树形展示、选中
加载表单、保存/删除语义、搜索过滤。四层存储全是同步实现，测试函数无需 ``asyncio.run()`` 桥接。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidget, QTreeWidget, QTreeWidgetItem
from qfluentwidgets import InfoBar, MessageBox

from miku_on_desk.brain.memory.models import Fact, MemoryUnit
from miku_on_desk.brain.memory.system import MemorySystem
from miku_on_desk.face.ui.memory_panel import MemoryPanel

_DATA_ROLE = int(Qt.ItemDataRole.UserRole)


def _accept_message_box(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MessageBox, "exec", lambda self: 1)


def _reject_message_box(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MessageBox, "exec", lambda self: 0)


@pytest.fixture
def system(tmp_path: Path) -> MemorySystem:
    return MemorySystem(tmp_path / "memory")


def _make_fact(
    *,
    subject: str = "user",
    predicate: str = "name",
    value: str = "tew",
    extracted_by: str = "tool:remember",
    pinned: bool = False,
    confidence: float = 1.0,
) -> Fact:
    return Fact(
        id="",
        subject=subject,
        subject_type="person",
        predicate=predicate,
        object=value,
        object_type="concept",
        confidence=confidence,
        source=[],
        valid_from="2026-01-01T00:00:00",
        recorded_at="2026-01-01T00:00:00",
        extracted_by=extracted_by,
        status="active",
        pinned=pinned,
    )


def _find_leaf(tree: QTreeWidget, predicate: Callable[[object], bool]) -> QTreeWidgetItem | None:
    def matches(item: QTreeWidgetItem) -> bool:
        return predicate(item.data(0, _DATA_ROLE))

    def search(item: QTreeWidgetItem) -> QTreeWidgetItem | None:
        if matches(item):
            return item
        for i in range(item.childCount()):
            found = search(item.child(i))
            if found is not None:
                return found
        return None

    for i in range(tree.topLevelItemCount()):
        found = search(tree.topLevelItem(i))
        if found is not None:
            return found
    return None


def _find_leaf_by_id(tree: QTreeWidget, item_id: str) -> QTreeWidgetItem | None:
    return _find_leaf(tree, lambda data: data == item_id)


def test_registers_five_sub_interfaces(qapp: QApplication, system: MemorySystem) -> None:
    panel = MemoryPanel(system)

    assert panel.stackedWidget.count() == 5


def test_navigation_interface_is_not_hidden_for_multi_tab_panel(
    qapp: QApplication, system: MemorySystem
) -> None:
    panel = MemoryPanel(system)

    assert panel.navigationInterface.isHidden() is False


# ── 语义 ─────────────────────────────────────────────────────────────────


def test_semantic_refresh_populates_tree_grouped_by_subject(
    qapp: QApplication, system: MemorySystem
) -> None:
    id1 = system.semantic.upsert_fact(_make_fact(predicate="name", value="tew"))
    id2 = system.semantic.upsert_fact(_make_fact(predicate="role", value="engineer"))

    panel = MemoryPanel(system)

    assert _find_leaf_by_id(panel._semantic_tree, id1) is not None
    assert _find_leaf_by_id(panel._semantic_tree, id2) is not None


def test_semantic_ai_sourced_fact_leaf_has_ai_suffix(
    qapp: QApplication, system: MemorySystem
) -> None:
    ai_id = system.semantic.upsert_fact(
        _make_fact(predicate="sleep_schedule", value="喜欢熬夜", extracted_by="extractor:semantic")
    )
    manual_id = system.semantic.upsert_fact(_make_fact(predicate="name", value="tew"))

    panel = MemoryPanel(system)

    ai_leaf = _find_leaf_by_id(panel._semantic_tree, ai_id)
    manual_leaf = _find_leaf_by_id(panel._semantic_tree, manual_id)
    assert ai_leaf is not None
    assert manual_leaf is not None
    assert "（AI）" in ai_leaf.text(0)
    assert "（AI）" not in manual_leaf.text(0)


def test_semantic_pinned_fact_leaf_has_pin_marker(qapp: QApplication, system: MemorySystem) -> None:
    pinned_id = system.semantic.upsert_fact(_make_fact(predicate="name", value="tew", pinned=True))

    panel = MemoryPanel(system)

    leaf = _find_leaf_by_id(panel._semantic_tree, pinned_id)
    assert leaf is not None
    assert "📌" in leaf.text(0)


def test_semantic_selecting_leaf_loads_form_fields(
    qapp: QApplication, system: MemorySystem
) -> None:
    fact_id = system.semantic.upsert_fact(_make_fact(predicate="role", value="engineer"))
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._semantic_tree, fact_id)
    assert leaf is not None

    panel._semantic_tree.setCurrentItem(leaf)

    assert panel._semantic_subject_edit.text() == "user"
    assert panel._semantic_predicate_edit.text() == "role"
    assert panel._semantic_object_edit.text() == "engineer"


def test_semantic_save_creates_new_fact(qapp: QApplication, system: MemorySystem) -> None:
    panel = MemoryPanel(system)
    panel._semantic_subject_edit.setText("user")
    panel._semantic_predicate_edit.setText("deadline")
    panel._semantic_object_edit.setText("2026-07-10")

    panel._on_semantic_save_clicked()

    facts = system.semantic.search_facts("deadline")
    assert len(facts) == 1
    assert facts[0].object == "2026-07-10"
    assert facts[0].extracted_by == "tool:remember"


def test_semantic_save_updates_existing_fact_preserves_provenance(
    qapp: QApplication, system: MemorySystem
) -> None:
    fact_id = system.semantic.upsert_fact(
        _make_fact(predicate="name", value="old-value", extracted_by="extractor:semantic")
    )
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._semantic_tree, fact_id)
    assert leaf is not None
    panel._semantic_tree.setCurrentItem(leaf)

    panel._semantic_object_edit.setText("new-value")
    panel._on_semantic_save_clicked()

    updated = system.semantic.get_fact(fact_id)
    assert updated is not None
    assert updated.object == "new-value"
    assert updated.extracted_by == "extractor:semantic"


def test_semantic_delete_removes_fact_and_clears_form(
    qapp: QApplication, system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_message_box(monkeypatch)
    fact_id = system.semantic.upsert_fact(_make_fact(predicate="temp", value="throwaway"))
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._semantic_tree, fact_id)
    assert leaf is not None
    panel._semantic_tree.setCurrentItem(leaf)

    panel._on_semantic_delete_clicked()

    assert system.semantic.get_fact(fact_id) is None
    assert panel._semantic_subject_edit.text() == ""


def test_semantic_new_button_clears_form(qapp: QApplication, system: MemorySystem) -> None:
    fact_id = system.semantic.upsert_fact(_make_fact(predicate="name", value="tew"))
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._semantic_tree, fact_id)
    assert leaf is not None
    panel._semantic_tree.setCurrentItem(leaf)

    panel._on_semantic_new_clicked()

    assert panel._semantic_subject_edit.text() == ""
    assert panel._semantic_predicate_edit.text() == ""


def test_semantic_search_filters_tree_to_matching_facts(
    qapp: QApplication, system: MemorySystem
) -> None:
    system.semantic.upsert_fact(_make_fact(predicate="name", value="tew"))
    deadline_id = system.semantic.upsert_fact(_make_fact(predicate="deadline", value="2026-07-10"))
    panel = MemoryPanel(system)

    panel._semantic_search_edit.setText("deadline")
    panel._on_semantic_search_clicked()

    assert _find_leaf_by_id(panel._semantic_tree, deadline_id) is not None
    assert _find_leaf(panel._semantic_tree, lambda data: data is not None) is not None
    assert panel._semantic_tree.topLevelItemCount() == 1


def test_semantic_reset_restores_full_list_after_search(
    qapp: QApplication, system: MemorySystem
) -> None:
    name_id = system.semantic.upsert_fact(_make_fact(predicate="name", value="tew"))
    deadline_id = system.semantic.upsert_fact(_make_fact(predicate="deadline", value="2026-07-10"))
    panel = MemoryPanel(system)
    panel._semantic_search_edit.setText("deadline")
    panel._on_semantic_search_clicked()

    panel._refresh_semantic()

    assert _find_leaf_by_id(panel._semantic_tree, name_id) is not None
    assert _find_leaf_by_id(panel._semantic_tree, deadline_id) is not None


# ── 情景 ─────────────────────────────────────────────────────────────────


def test_episodic_refresh_populates_year_month_tree(
    qapp: QApplication, system: MemorySystem
) -> None:
    event_id = system.episodic.append_event(
        title="第一次对话", summary="打了个招呼", occurred_at="2026-07-06T10:00:00"
    )

    panel = MemoryPanel(system)

    assert _find_leaf_by_id(panel._episodic_tree, event_id) is not None


def test_episodic_selecting_event_loads_summary(qapp: QApplication, system: MemorySystem) -> None:
    event_id = system.episodic.append_event(
        title="第一次对话", summary="打了个招呼", occurred_at="2026-07-06T10:00:00"
    )
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._episodic_tree, event_id)
    assert leaf is not None

    panel._episodic_tree.setCurrentItem(leaf)

    assert panel._episodic_summary_edit.toPlainText() == "打了个招呼"


def test_episodic_save_updates_summary(qapp: QApplication, system: MemorySystem) -> None:
    event_id = system.episodic.append_event(
        title="第一次对话", summary="打了个招呼", occurred_at="2026-07-06T10:00:00"
    )
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._episodic_tree, event_id)
    assert leaf is not None
    panel._episodic_tree.setCurrentItem(leaf)

    panel._episodic_summary_edit.setPlainText("修改后的摘要")
    panel._on_episodic_save_clicked()

    updated = system.episodic.get_event(event_id)
    assert updated is not None
    assert updated.summary == "修改后的摘要"


def test_episodic_delete_removes_event(
    qapp: QApplication, system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_message_box(monkeypatch)
    event_id = system.episodic.append_event(
        title="临时事件", summary="待删除", occurred_at="2026-07-06T10:00:00"
    )
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._episodic_tree, event_id)
    assert leaf is not None
    panel._episodic_tree.setCurrentItem(leaf)

    panel._on_episodic_delete_clicked()

    assert system.episodic.get_event(event_id) is None
    assert _find_leaf_by_id(panel._episodic_tree, event_id) is None


def test_episodic_search_filters_tree_to_matching_events(
    qapp: QApplication, system: MemorySystem
) -> None:
    system.episodic.append_event(
        title="打招呼", summary="日常寒暄", occurred_at="2026-07-06T10:00:00"
    )
    deadline_id = system.episodic.append_event(
        title="截止日期讨论", summary="聊了项目排期", occurred_at="2026-07-06T11:00:00"
    )
    panel = MemoryPanel(system)

    panel._episodic_search_edit.setText("截止日期")
    panel._on_episodic_search_clicked()

    assert _find_leaf_by_id(panel._episodic_tree, deadline_id) is not None


def test_episodic_reset_restores_full_list_after_search(
    qapp: QApplication, system: MemorySystem
) -> None:
    greet_id = system.episodic.append_event(
        title="打招呼", summary="日常寒暄", occurred_at="2026-07-06T10:00:00"
    )
    deadline_id = system.episodic.append_event(
        title="截止日期讨论", summary="聊了项目排期", occurred_at="2026-07-06T11:00:00"
    )
    panel = MemoryPanel(system)
    panel._episodic_search_edit.setText("截止日期")
    panel._on_episodic_search_clicked()

    panel._refresh_episodic()

    assert _find_leaf_by_id(panel._episodic_tree, greet_id) is not None
    assert _find_leaf_by_id(panel._episodic_tree, deadline_id) is not None


# ── 情感 ─────────────────────────────────────────────────────────────────


def test_emotional_refresh_populates_flattened_tree(
    qapp: QApplication, system: MemorySystem
) -> None:
    data = system.emotional.load_preferences()
    data["habits"] = {"coffee": "喝美式"}
    system.emotional.save_preferences(data)

    panel = MemoryPanel(system)

    assert _find_leaf_by_id(panel._emotional_tree, "habits/coffee") is not None


def test_emotional_selecting_leaf_loads_path_and_value(
    qapp: QApplication, system: MemorySystem
) -> None:
    data = system.emotional.load_preferences()
    data["habits"] = {"coffee": "喝美式"}
    system.emotional.save_preferences(data)
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._emotional_tree, "habits/coffee")
    assert leaf is not None

    panel._emotional_tree.setCurrentItem(leaf)

    assert panel._emotional_path_edit.text() == "habits/coffee"
    assert "喝美式" in panel._emotional_value_edit.toPlainText()


def test_emotional_save_updates_existing_leaf(qapp: QApplication, system: MemorySystem) -> None:
    data = system.emotional.load_preferences()
    data["habits"] = {"coffee": "喝美式"}
    system.emotional.save_preferences(data)
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._emotional_tree, "habits/coffee")
    assert leaf is not None
    panel._emotional_tree.setCurrentItem(leaf)

    panel._emotional_value_edit.setPlainText('"喝拿铁"')
    panel._on_emotional_save_clicked()

    updated = system.emotional.load_preferences()
    assert updated["habits"]["coffee"] == "喝拿铁"


def test_emotional_save_creates_new_leaf(qapp: QApplication, system: MemorySystem) -> None:
    panel = MemoryPanel(system)
    panel._emotional_path_edit.setText("habits/sleep_schedule")
    panel._emotional_value_edit.setPlainText('"喜欢熬夜"')

    panel._on_emotional_save_clicked()

    data = system.emotional.load_preferences()
    assert data["habits"]["sleep_schedule"] == "喜欢熬夜"


def test_emotional_delete_removes_leaf(
    qapp: QApplication, system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_message_box(monkeypatch)
    data = system.emotional.load_preferences()
    data["habits"] = {"coffee": "喝美式"}
    system.emotional.save_preferences(data)
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._emotional_tree, "habits/coffee")
    assert leaf is not None
    panel._emotional_tree.setCurrentItem(leaf)

    panel._on_emotional_delete_clicked()

    updated = system.emotional.load_preferences()
    assert "coffee" not in updated.get("habits", {})
    assert panel._emotional_path_edit.text() == ""


def test_emotional_search_filters_tree_to_matching_paths(
    qapp: QApplication, system: MemorySystem
) -> None:
    data = system.emotional.load_preferences()
    data["habits"] = {"coffee": "喝美式", "sleep_schedule": "喜欢熬夜"}
    system.emotional.save_preferences(data)
    panel = MemoryPanel(system)

    panel._emotional_search_edit.setText("coffee")
    panel._on_emotional_search_clicked()

    assert _find_leaf_by_id(panel._emotional_tree, "habits/coffee") is not None
    assert _find_leaf_by_id(panel._emotional_tree, "habits/sleep_schedule") is None


def test_emotional_reset_restores_full_list_after_search(
    qapp: QApplication, system: MemorySystem
) -> None:
    data = system.emotional.load_preferences()
    data["habits"] = {"coffee": "喝美式", "sleep_schedule": "喜欢熬夜"}
    system.emotional.save_preferences(data)
    panel = MemoryPanel(system)
    panel._emotional_search_edit.setText("coffee")
    panel._on_emotional_search_clicked()

    panel._refresh_emotional()

    assert _find_leaf_by_id(panel._emotional_tree, "habits/coffee") is not None
    assert _find_leaf_by_id(panel._emotional_tree, "habits/sleep_schedule") is not None


# ── 原始会话 ───────────────────────────────────────────────────────────────


def _find_session_item(list_widget: QListWidget, session_id: str) -> object | None:
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        if item.data(_DATA_ROLE) == session_id:
            return item
    return None


def test_base_refresh_populates_session_list(qapp: QApplication, system: MemorySystem) -> None:
    system.base.start_session("s1", "对话一")

    panel = MemoryPanel(system)

    assert _find_session_item(panel._base_session_list, "s1") is not None


def test_base_selecting_session_shows_units(qapp: QApplication, system: MemorySystem) -> None:
    system.base.start_session("s1", "对话一")
    system.add_memory_unit(
        MemoryUnit(
            id="",
            session_id="s1",
            role="user",
            content="你好",
            created_at="2026-07-06T10:00:00",
        )
    )
    panel = MemoryPanel(system)
    item = _find_session_item(panel._base_session_list, "s1")
    assert item is not None

    panel._base_session_list.setCurrentItem(item)  # type: ignore[arg-type]

    assert "你好" in panel._base_units_view.toPlainText()


def test_base_search_filters_units_to_matching_content(
    qapp: QApplication, system: MemorySystem
) -> None:
    system.base.start_session("s1", "对话一")
    system.add_memory_unit(
        MemoryUnit(
            id="", session_id="s1", role="user", content="你好", created_at="2026-07-06T10:00:00"
        )
    )
    system.add_memory_unit(
        MemoryUnit(
            id="",
            session_id="s1",
            role="assistant",
            content="在摸鱼",
            created_at="2026-07-06T10:01:00",
        )
    )
    panel = MemoryPanel(system)
    item = _find_session_item(panel._base_session_list, "s1")
    assert item is not None
    panel._base_session_list.setCurrentItem(item)  # type: ignore[arg-type]

    panel._base_search_edit.setText("摸鱼")
    panel._on_base_search_clicked()

    assert "在摸鱼" in panel._base_units_view.toPlainText()
    assert "你好" not in panel._base_units_view.toPlainText()


def test_base_reset_restores_full_session_view_after_search(
    qapp: QApplication, system: MemorySystem
) -> None:
    system.base.start_session("s1", "对话一")
    system.add_memory_unit(
        MemoryUnit(
            id="", session_id="s1", role="user", content="你好", created_at="2026-07-06T10:00:00"
        )
    )
    panel = MemoryPanel(system)
    item = _find_session_item(panel._base_session_list, "s1")
    assert item is not None
    panel._base_session_list.setCurrentItem(item)  # type: ignore[arg-type]
    panel._base_search_edit.setText("不存在的内容")
    panel._on_base_search_clicked()

    panel._on_base_reset_clicked()

    assert "你好" in panel._base_units_view.toPlainText()


def test_base_delete_removes_session_from_list(
    qapp: QApplication, system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_message_box(monkeypatch)
    system.base.start_session("s1", "对话一")
    panel = MemoryPanel(system)
    item = _find_session_item(panel._base_session_list, "s1")
    assert item is not None
    panel._base_session_list.setCurrentItem(item)  # type: ignore[arg-type]

    panel._on_base_delete_clicked()

    assert system.base.list_sessions() == []
    assert _find_session_item(panel._base_session_list, "s1") is None


# ── 删除确认 / 校验反馈 ─────────────────────────────────────────────────────


def test_semantic_delete_does_nothing_when_confirmation_declined(
    qapp: QApplication, system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reject_message_box(monkeypatch)
    fact_id = system.semantic.upsert_fact(_make_fact(predicate="temp", value="throwaway"))
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._semantic_tree, fact_id)
    assert leaf is not None
    panel._semantic_tree.setCurrentItem(leaf)

    panel._on_semantic_delete_clicked()

    assert system.semantic.get_fact(fact_id) is not None


def test_episodic_delete_does_nothing_when_confirmation_declined(
    qapp: QApplication, system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reject_message_box(monkeypatch)
    event_id = system.episodic.append_event(
        title="临时事件", summary="待删除", occurred_at="2026-07-06T10:00:00"
    )
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._episodic_tree, event_id)
    assert leaf is not None
    panel._episodic_tree.setCurrentItem(leaf)

    panel._on_episodic_delete_clicked()

    assert system.episodic.get_event(event_id) is not None


def test_emotional_delete_does_nothing_when_confirmation_declined(
    qapp: QApplication, system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reject_message_box(monkeypatch)
    data = system.emotional.load_preferences()
    data["habits"] = {"coffee": "喝美式"}
    system.emotional.save_preferences(data)
    panel = MemoryPanel(system)
    leaf = _find_leaf_by_id(panel._emotional_tree, "habits/coffee")
    assert leaf is not None
    panel._emotional_tree.setCurrentItem(leaf)

    panel._on_emotional_delete_clicked()

    updated = system.emotional.load_preferences()
    assert updated["habits"]["coffee"] == "喝美式"


def test_base_delete_does_nothing_when_confirmation_declined(
    qapp: QApplication, system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reject_message_box(monkeypatch)
    system.base.start_session("s1", "对话一")
    panel = MemoryPanel(system)
    item = _find_session_item(panel._base_session_list, "s1")
    assert item is not None
    panel._base_session_list.setCurrentItem(item)  # type: ignore[arg-type]

    panel._on_base_delete_clicked()

    assert system.base.list_sessions() != []


def test_emotional_save_invalid_json_shows_info_bar_and_does_not_save(
    qapp: QApplication, system: MemorySystem
) -> None:
    panel = MemoryPanel(system)
    panel._emotional_path_edit.setText("habits/sleep_schedule")
    panel._emotional_value_edit.setPlainText("{not valid json")

    panel._on_emotional_save_clicked()

    data = system.emotional.load_preferences()
    assert "habits" not in data
    assert panel.findChildren(InfoBar)


# ── 诊断 ─────────────────────────────────────────────────────────────────


def test_diagnostics_refresh_shows_current_threshold_values(
    qapp: QApplication, system: MemorySystem
) -> None:
    panel = MemoryPanel(system)

    panel._refresh_diagnostics()

    text = panel._diagnostics_view.toPlainText()
    assert "0.70" in text
    assert "0.80" in text
    assert "0.75" in text


def test_diagnostics_counts_facts_filtered_by_retrieval_threshold(
    qapp: QApplication, system: MemorySystem
) -> None:
    system.semantic.upsert_fact(_make_fact(predicate="住在", value="上海", confidence=0.9))
    system.semantic.upsert_fact(_make_fact(predicate="喜欢", value="猫", confidence=0.5))
    panel = MemoryPanel(system)

    panel._refresh_diagnostics()

    text = panel._diagnostics_view.toPlainText()
    assert "活跃语义事实总数：2" in text
    assert "其中会被检索阈值过滤掉的条数：1" in text


def test_diagnostics_counts_duplicate_groups_from_similar_base_units(
    qapp: QApplication, system: MemorySystem
) -> None:
    system.base.append(
        MemoryUnit(
            id="",
            session_id="s1",
            role="user",
            content="今天天气真好呀，想出去走走。",
            created_at="2026-07-06T09:00:00+00:00",
        )
    )
    system.base.append(
        MemoryUnit(
            id="",
            session_id="s1",
            role="user",
            content="今天天气真好呀，想出去走走。",
            created_at="2026-07-06T09:01:00+00:00",
        )
    )
    panel = MemoryPanel(system)

    panel._refresh_diagnostics()

    text = panel._diagnostics_view.toPlainText()
    assert "疑似重复记忆组数（全库扫描）：1" in text
