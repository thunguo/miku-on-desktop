"""记忆管理面板：四层文件系统记忆架构（语义/情景/情感/原始会话）的浏览/搜索/编辑/删除界面。

四个子存储（``BaseStore``/``SemanticStore``/``EpisodicStore``/``EmotionalStore``）全是同步
实现（纯 ``pathlib``/JSON/Markdown 读写），Qt 槽函数可以直接调用。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QListWidgetItem,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CheckBox,
    ComboBox,
    FluentIcon,
    FluentWindow,
    InfoBar,
    LineEdit,
    ListWidget,
    MessageBox,
    PlainTextEdit,
    PrimaryPushButton,
    PushButton,
    TreeWidget,
)

from miku_on_desk.brain.memory.models import EntityType, Episode, Fact
from miku_on_desk.brain.memory.system import MemorySystem

_ENTITY_TYPES: list[EntityType] = [
    "person",
    "location",
    "organization",
    "concept",
    "event",
    "technology",
]
_MANUAL_EXTRACTED_BY = "tool:remember"
_DATA_ROLE = int(Qt.ItemDataRole.UserRole)
_MAX_DUPLICATE_SCAN_UNITS = 200


def _populate_semantic_tree(tree: TreeWidget, facts: list[Fact]) -> None:
    tree.clear()
    branches: dict[str, QTreeWidgetItem] = {}
    for fact in facts:
        branch = branches.get(fact.subject)
        if branch is None:
            branch = QTreeWidgetItem([fact.subject])
            tree.addTopLevelItem(branch)
            branches[fact.subject] = branch
        leaf_text = f"{fact.predicate}：{fact.object}"
        if fact.extracted_by != _MANUAL_EXTRACTED_BY:
            leaf_text = f"{leaf_text}（AI）"
        if fact.pinned:
            leaf_text = f"{leaf_text} 📌"
        leaf = QTreeWidgetItem([leaf_text])
        leaf.setData(0, _DATA_ROLE, fact.id)
        branch.addChild(leaf)
    tree.expandAll()


def _populate_episodic_tree(tree: TreeWidget, episodes: list[Episode]) -> None:
    tree.clear()
    year_branches: dict[str, QTreeWidgetItem] = {}
    month_branches: dict[str, QTreeWidgetItem] = {}
    for episode in episodes:
        year = episode.month[:4]
        year_branch = year_branches.get(year)
        if year_branch is None:
            year_branch = QTreeWidgetItem([year])
            tree.addTopLevelItem(year_branch)
            year_branches[year] = year_branch
        month_branch = month_branches.get(episode.month)
        if month_branch is None:
            month_branch = QTreeWidgetItem([episode.month])
            year_branch.addChild(month_branch)
            month_branches[episode.month] = month_branch
        leaf = QTreeWidgetItem([f"[{episode.id}] {episode.title}"])
        leaf.setData(0, _DATA_ROLE, episode.id)
        month_branch.addChild(leaf)
    tree.expandAll()


def _flatten_preferences(
    data: dict[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = []
    for key, value in data.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            paths.extend(_flatten_preferences(cast(dict[str, Any], value), path))
        else:
            paths.append(path)
    return paths


def _populate_emotional_tree(tree: TreeWidget, data: dict[str, Any]) -> None:
    tree.clear()
    branches: dict[tuple[str, ...], QTreeWidgetItem] = {}

    def branch_item(path: tuple[str, ...]) -> QTreeWidgetItem | None:
        if not path:
            return None
        if path in branches:
            return branches[path]
        parent = branch_item(path[:-1])
        item = QTreeWidgetItem([path[-1]])
        if parent is None:
            tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        branches[path] = item
        return item

    for path in _flatten_preferences(data):
        parent = branch_item(path[:-1])
        leaf = QTreeWidgetItem([path[-1]])
        leaf.setData(0, _DATA_ROLE, "/".join(path))
        if parent is None:
            tree.addTopLevelItem(leaf)
        else:
            parent.addChild(leaf)
    tree.expandAll()


def _get_nested(data: dict[str, Any], path: list[str]) -> Any:
    node: Any = data
    for segment in path:
        node = node[segment]
    return node


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    node = data
    for segment in path[:-1]:
        child = node.setdefault(segment, {})
        node = cast(dict[str, Any], child)
    node[path[-1]] = value


def _delete_nested(data: dict[str, Any], path: list[str]) -> None:
    node = data
    for segment in path[:-1]:
        node = cast(dict[str, Any], node[segment])
    node.pop(path[-1], None)


def _render_units(session_id: str, system: MemorySystem) -> str:
    units = system.base.list_units(session_id=session_id)
    return "\n\n".join(f"[{unit.created_at}] {unit.role}：{unit.content}" for unit in units)


def _populate_session_list(list_widget: ListWidget, system: MemorySystem) -> None:
    list_widget.clear()
    for meta in system.base.list_sessions():
        item = QListWidgetItem(f"{meta.title} — {meta.updated_at}")
        item.setData(_DATA_ROLE, meta.session_id)
        list_widget.addItem(item)


def _confirm_delete(parent: QWidget, name: str) -> bool:
    box = MessageBox("确认删除", f"确定要删除「{name}」吗？此操作不可撤销。", parent)
    return bool(box.exec())


def _count_duplicate_groups(system: MemorySystem) -> int:
    """统计疑似重复组数，并为避免 UI 卡顿仅扫描前 N 条 base 记录。"""
    units = system.base.list_units()
    if len(units) > _MAX_DUPLICATE_SCAN_UNITS:
        units = units[:_MAX_DUPLICATE_SCAN_UNITS]
    visited: set[str] = set()
    groups = 0
    for unit in units:
        if unit.id in visited:
            continue
        visited.add(unit.id)
        similar = system.base.find_semantically_similar(unit)
        if similar:
            groups += 1
            visited.update(other_id for other_id, _score in similar)
    return groups


def _render_diagnostics(system: MemorySystem) -> str:
    tuning = system.tuning
    active_facts = system.semantic.list_facts(status="active")
    base_unit_count = len(system.base.list_units())
    duplicate_groups = _count_duplicate_groups(system)
    duplicate_groups_line = f"疑似重复记忆组数（全库扫描）：{duplicate_groups}"
    if base_unit_count > _MAX_DUPLICATE_SCAN_UNITS:
        duplicate_groups_line = (
            f"{duplicate_groups_line}（基于前 {_MAX_DUPLICATE_SCAN_UNITS} 条记录的顺序截断结果）"
        )
    filtered_count = sum(
        1 for fact in active_facts if not (fact.confidence > tuning.retrieval_min_confidence)
    )
    lines = [
        f"检索置信度阈值（retrieval_min_confidence）：{tuning.retrieval_min_confidence:.2f}",
        f"写入相似度阈值（base_similarity_threshold）：{tuning.base_similarity_threshold:.2f}",
        f"情感置信度阈值（emotional_confidence_threshold）："
        f"{tuning.emotional_confidence_threshold:.2f}",
        "",
        f"活跃语义事实总数：{len(active_facts)}",
        f"其中会被检索阈值过滤掉的条数：{filtered_count}",
        f"base 层原始记录总数：{base_unit_count}",
        duplicate_groups_line,
    ]
    return "\n".join(lines)


class MemoryPanel(FluentWindow):  # type: ignore[misc]
    """五个标签页各自独立操作对应的存储层，互不共享状态。"""

    def __init__(self, memory_system: MemorySystem, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._system = memory_system

        semantic_tab = self._build_semantic_tab()
        episodic_tab = self._build_episodic_tab()
        emotional_tab = self._build_emotional_tab()
        base_tab = self._build_base_tab()
        diagnostics_tab = self._build_diagnostics_tab()

        for widget, object_name in (
            (semantic_tab, "semanticTab"),
            (episodic_tab, "episodicTab"),
            (emotional_tab, "emotionalTab"),
            (base_tab, "baseTab"),
            (diagnostics_tab, "diagnosticsTab"),
        ):
            widget.setObjectName(object_name)

        self.addSubInterface(semantic_tab, FluentIcon.LIBRARY, "语义")
        self.addSubInterface(episodic_tab, FluentIcon.CALENDAR, "情景")
        self.addSubInterface(emotional_tab, FluentIcon.HEART, "情感")
        self.addSubInterface(base_tab, FluentIcon.MESSAGE, "原始会话")
        self.addSubInterface(diagnostics_tab, FluentIcon.DEVELOPER_TOOLS, "诊断")

        self.resize(720, 560)

        self._refresh_semantic()
        self._refresh_episodic()
        self._refresh_emotional()
        self._refresh_base()

    # ── 语义 ─────────────────────────────────────────────────────────────

    def _build_semantic_tab(self) -> QWidget:
        container = QWidget(self)

        self._semantic_search_edit = LineEdit(container)
        search_button = PushButton("搜索", container)
        reset_button = PushButton("显示全部", container)
        search_button.clicked.connect(self._on_semantic_search_clicked)
        reset_button.clicked.connect(self._refresh_semantic)

        self._semantic_tree = TreeWidget(container)
        self._semantic_tree.setHeaderHidden(True)
        self._semantic_tree.currentItemChanged.connect(self._on_semantic_selection_changed)

        self._semantic_subject_edit = LineEdit(container)
        self._semantic_subject_type_combo = ComboBox(container)
        self._semantic_subject_type_combo.addItems(_ENTITY_TYPES)
        self._semantic_predicate_edit = LineEdit(container)
        self._semantic_object_edit = LineEdit(container)
        self._semantic_object_type_combo = ComboBox(container)
        self._semantic_object_type_combo.addItems(_ENTITY_TYPES)
        self._semantic_confidence_edit = LineEdit(container)
        self._semantic_confidence_edit.setText("1.0")
        self._semantic_pinned_check = CheckBox(container)

        new_button = PushButton("新建", container)
        save_button = PrimaryPushButton("保存", container)
        delete_button = PushButton("删除", container)
        new_button.clicked.connect(self._on_semantic_new_clicked)
        save_button.clicked.connect(self._on_semantic_save_clicked)
        delete_button.clicked.connect(self._on_semantic_delete_clicked)

        search_row = QHBoxLayout()
        search_row.addWidget(self._semantic_search_edit)
        search_row.addWidget(search_button)
        search_row.addWidget(reset_button)

        form = QFormLayout()
        form.addRow("主体", self._semantic_subject_edit)
        form.addRow("主体类型", self._semantic_subject_type_combo)
        form.addRow("关系", self._semantic_predicate_edit)
        form.addRow("客体", self._semantic_object_edit)
        form.addRow("客体类型", self._semantic_object_type_combo)
        form.addRow("置信度", self._semantic_confidence_edit)
        form.addRow("常驻", self._semantic_pinned_check)

        button_row = QHBoxLayout()
        button_row.addWidget(new_button)
        button_row.addWidget(save_button)
        button_row.addWidget(delete_button)

        layout = QVBoxLayout(container)
        layout.addLayout(search_row)
        layout.addWidget(self._semantic_tree)
        layout.addLayout(form)
        layout.addLayout(button_row)
        return container

    def _refresh_semantic(self) -> None:
        facts = self._system.semantic.list_facts()
        _populate_semantic_tree(self._semantic_tree, facts)

    def _on_semantic_search_clicked(self) -> None:
        query = self._semantic_search_edit.text().strip()
        if not query:
            self._refresh_semantic()
            return
        _populate_semantic_tree(self._semantic_tree, self._system.semantic.search_facts(query))

    def _on_semantic_selection_changed(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        if current is None:
            return
        fact_id = current.data(0, _DATA_ROLE)
        if fact_id is None:
            return
        fact = self._system.semantic.get_fact(fact_id)
        if fact is None:
            return
        self._load_semantic_form(fact)

    def _load_semantic_form(self, fact: Fact) -> None:
        self._semantic_subject_edit.setText(fact.subject)
        self._semantic_subject_type_combo.setCurrentText(fact.subject_type)
        self._semantic_predicate_edit.setText(fact.predicate)
        self._semantic_object_edit.setText(fact.object)
        self._semantic_object_type_combo.setCurrentText(fact.object_type)
        self._semantic_confidence_edit.setText(str(fact.confidence))
        self._semantic_pinned_check.setChecked(fact.pinned)
        self._semantic_tree.setProperty("selectedFactId", fact.id)

    def _on_semantic_new_clicked(self) -> None:
        self._semantic_subject_edit.clear()
        self._semantic_subject_type_combo.setCurrentIndex(0)
        self._semantic_predicate_edit.clear()
        self._semantic_object_edit.clear()
        self._semantic_object_type_combo.setCurrentIndex(0)
        self._semantic_confidence_edit.setText("1.0")
        self._semantic_pinned_check.setChecked(False)
        self._semantic_tree.setProperty("selectedFactId", None)
        self._semantic_tree.clearSelection()

    def _on_semantic_save_clicked(self) -> None:
        subject = self._semantic_subject_edit.text().strip()
        predicate = self._semantic_predicate_edit.text().strip()
        obj = self._semantic_object_edit.text().strip()
        if not subject or not predicate or not obj:
            return
        try:
            confidence = float(self._semantic_confidence_edit.text().strip())
        except ValueError:
            confidence = 1.0

        selected_id = self._semantic_tree.property("selectedFactId")
        existing = self._system.semantic.get_fact(selected_id) if selected_id else None

        now = datetime.now(UTC).isoformat()
        fact = Fact(
            id=existing.id if existing is not None else "",
            subject=subject,
            subject_type=cast(EntityType, self._semantic_subject_type_combo.currentText()),
            predicate=predicate,
            object=obj,
            object_type=cast(EntityType, self._semantic_object_type_combo.currentText()),
            confidence=confidence,
            source=existing.source if existing is not None else [],
            valid_from=existing.valid_from if existing is not None else now,
            recorded_at=existing.recorded_at if existing is not None else now,
            extracted_by=existing.extracted_by if existing is not None else _MANUAL_EXTRACTED_BY,
            status=existing.status if existing is not None else "active",
            context=existing.context if existing is not None else None,
            pinned=self._semantic_pinned_check.isChecked(),
        )
        self._system.semantic.upsert_fact(fact)
        self._refresh_semantic()

    def _on_semantic_delete_clicked(self) -> None:
        selected_id = self._semantic_tree.property("selectedFactId")
        if not selected_id:
            return
        current = self._semantic_tree.currentItem()
        name = current.text(0) if current is not None else selected_id
        if not _confirm_delete(self, name):
            return
        self._system.semantic.delete_fact(selected_id)
        self._on_semantic_new_clicked()
        self._refresh_semantic()

    # ── 情景 ─────────────────────────────────────────────────────────────

    def _build_episodic_tab(self) -> QWidget:
        container = QWidget(self)

        self._episodic_search_edit = LineEdit(container)
        search_button = PushButton("搜索", container)
        reset_button = PushButton("显示全部", container)
        search_button.clicked.connect(self._on_episodic_search_clicked)
        reset_button.clicked.connect(self._refresh_episodic)

        self._episodic_tree = TreeWidget(container)
        self._episodic_tree.setHeaderHidden(True)
        self._episodic_tree.currentItemChanged.connect(self._on_episodic_selection_changed)

        self._episodic_summary_edit = PlainTextEdit(container)

        save_button = PrimaryPushButton("保存摘要", container)
        delete_button = PushButton("删除事件", container)
        save_button.clicked.connect(self._on_episodic_save_clicked)
        delete_button.clicked.connect(self._on_episodic_delete_clicked)

        search_row = QHBoxLayout()
        search_row.addWidget(self._episodic_search_edit)
        search_row.addWidget(search_button)
        search_row.addWidget(reset_button)

        button_row = QHBoxLayout()
        button_row.addWidget(save_button)
        button_row.addWidget(delete_button)

        layout = QVBoxLayout(container)
        layout.addLayout(search_row)
        layout.addWidget(self._episodic_tree)
        layout.addWidget(self._episodic_summary_edit)
        layout.addLayout(button_row)
        return container

    def _refresh_episodic(self) -> None:
        episodes = self._system.episodic.list_events()
        _populate_episodic_tree(self._episodic_tree, episodes)

    def _on_episodic_search_clicked(self) -> None:
        query = self._episodic_search_edit.text().strip()
        if not query:
            self._refresh_episodic()
            return
        _populate_episodic_tree(self._episodic_tree, self._system.episodic.search(query))

    def _on_episodic_selection_changed(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        if current is None:
            return
        event_id = current.data(0, _DATA_ROLE)
        if event_id is None:
            return
        episode = self._system.episodic.get_event(event_id)
        if episode is None:
            return
        self._episodic_summary_edit.setPlainText(episode.summary)
        self._episodic_tree.setProperty("selectedEventId", episode.id)

    def _on_episodic_save_clicked(self) -> None:
        event_id = self._episodic_tree.property("selectedEventId")
        if not event_id:
            return
        self._system.episodic.update_summary(event_id, self._episodic_summary_edit.toPlainText())
        self._refresh_episodic()

    def _on_episodic_delete_clicked(self) -> None:
        event_id = self._episodic_tree.property("selectedEventId")
        if not event_id:
            return
        current = self._episodic_tree.currentItem()
        name = current.text(0) if current is not None else event_id
        if not _confirm_delete(self, name):
            return
        self._system.episodic.delete_event(event_id)
        self._episodic_summary_edit.clear()
        self._episodic_tree.setProperty("selectedEventId", None)
        self._refresh_episodic()

    # ── 情感 ─────────────────────────────────────────────────────────────

    def _build_emotional_tab(self) -> QWidget:
        container = QWidget(self)

        self._emotional_search_edit = LineEdit(container)
        search_button = PushButton("搜索", container)
        reset_button = PushButton("显示全部", container)
        search_button.clicked.connect(self._on_emotional_search_clicked)
        reset_button.clicked.connect(self._refresh_emotional)

        self._emotional_tree = TreeWidget(container)
        self._emotional_tree.setHeaderHidden(True)
        self._emotional_tree.currentItemChanged.connect(self._on_emotional_selection_changed)

        self._emotional_path_edit = LineEdit(container)
        self._emotional_value_edit = PlainTextEdit(container)

        new_button = PushButton("新建", container)
        save_button = PrimaryPushButton("保存", container)
        delete_button = PushButton("删除", container)
        new_button.clicked.connect(self._on_emotional_new_clicked)
        save_button.clicked.connect(self._on_emotional_save_clicked)
        delete_button.clicked.connect(self._on_emotional_delete_clicked)

        search_row = QHBoxLayout()
        search_row.addWidget(self._emotional_search_edit)
        search_row.addWidget(search_button)
        search_row.addWidget(reset_button)

        form = QFormLayout()
        form.addRow("路径（用 / 分隔）", self._emotional_path_edit)
        form.addRow("值（JSON）", self._emotional_value_edit)

        button_row = QHBoxLayout()
        button_row.addWidget(new_button)
        button_row.addWidget(save_button)
        button_row.addWidget(delete_button)

        layout = QVBoxLayout(container)
        layout.addLayout(search_row)
        layout.addWidget(self._emotional_tree)
        layout.addLayout(form)
        layout.addLayout(button_row)
        return container

    def _refresh_emotional(self) -> None:
        data = self._system.emotional.load_preferences()
        _populate_emotional_tree(self._emotional_tree, data)

    def _on_emotional_search_clicked(self) -> None:
        query = self._emotional_search_edit.text().strip().lower()
        data = self._system.emotional.load_preferences()
        if not query:
            _populate_emotional_tree(self._emotional_tree, data)
            return
        self._emotional_tree.clear()
        for path in _flatten_preferences(data):
            path_text = "/".join(path)
            value_text = json.dumps(_get_nested(data, list(path)), ensure_ascii=False)
            if query in path_text.lower() or query in value_text.lower():
                leaf = QTreeWidgetItem([path_text])
                leaf.setData(0, _DATA_ROLE, path_text)
                self._emotional_tree.addTopLevelItem(leaf)

    def _on_emotional_selection_changed(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        if current is None:
            return
        path_text = current.data(0, _DATA_ROLE)
        if path_text is None:
            return
        data = self._system.emotional.load_preferences()
        try:
            value = _get_nested(data, path_text.split("/"))
        except KeyError:
            return
        self._emotional_path_edit.setText(path_text)
        self._emotional_value_edit.setPlainText(json.dumps(value, ensure_ascii=False, indent=2))

    def _on_emotional_new_clicked(self) -> None:
        self._emotional_path_edit.clear()
        self._emotional_value_edit.clear()
        self._emotional_tree.clearSelection()

    def _on_emotional_save_clicked(self) -> None:
        path_text = self._emotional_path_edit.text().strip().strip("/")
        if not path_text:
            return
        try:
            value = json.loads(self._emotional_value_edit.toPlainText() or "null")
        except json.JSONDecodeError as exc:
            InfoBar.error(title="JSON 格式错误", content=str(exc), parent=self)
            return
        data = self._system.emotional.load_preferences()
        _set_nested(data, path_text.split("/"), value)
        self._system.emotional.save_preferences(data)
        self._refresh_emotional()

    def _on_emotional_delete_clicked(self) -> None:
        path_text = self._emotional_path_edit.text().strip().strip("/")
        if not path_text:
            return
        if not _confirm_delete(self, path_text):
            return
        data = self._system.emotional.load_preferences()
        _delete_nested(data, path_text.split("/"))
        self._system.emotional.save_preferences(data)
        self._on_emotional_new_clicked()
        self._refresh_emotional()

    # ── 原始会话 ─────────────────────────────────────────────────────────

    def _build_base_tab(self) -> QWidget:
        container = QWidget(self)

        self._base_search_edit = LineEdit(container)
        search_button = PushButton("搜索", container)
        reset_button = PushButton("显示全部", container)
        search_button.clicked.connect(self._on_base_search_clicked)
        reset_button.clicked.connect(self._on_base_reset_clicked)

        self._base_session_list = ListWidget(container)
        self._base_session_list.currentItemChanged.connect(self._on_base_session_changed)

        self._base_units_view = PlainTextEdit(container)
        self._base_units_view.setReadOnly(True)

        delete_button = PushButton("删除会话", container)
        delete_button.clicked.connect(self._on_base_delete_clicked)

        search_row = QHBoxLayout()
        search_row.addWidget(self._base_search_edit)
        search_row.addWidget(search_button)
        search_row.addWidget(reset_button)

        layout = QVBoxLayout(container)
        layout.addWidget(self._base_session_list)
        layout.addLayout(search_row)
        layout.addWidget(self._base_units_view)
        layout.addWidget(delete_button)
        return container

    def _refresh_base(self) -> None:
        _populate_session_list(self._base_session_list, self._system)
        self._base_units_view.clear()

    def _selected_session_id(self) -> str | None:
        item = self._base_session_list.currentItem()
        if item is None:
            return None
        return cast(str | None, item.data(_DATA_ROLE))

    def _on_base_session_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            self._base_units_view.clear()
            return
        session_id = cast(str, current.data(_DATA_ROLE))
        self._base_units_view.setPlainText(_render_units(session_id, self._system))

    def _on_base_search_clicked(self) -> None:
        session_id = self._selected_session_id()
        if session_id is None:
            return
        query = self._base_search_edit.text().strip()
        if not query:
            self._base_units_view.setPlainText(_render_units(session_id, self._system))
            return
        matches = self._system.base.search(query, session_id=session_id)
        self._base_units_view.setPlainText(
            "\n\n".join(f"[{unit.created_at}] {unit.role}：{unit.content}" for unit in matches)
        )

    def _on_base_reset_clicked(self) -> None:
        session_id = self._selected_session_id()
        if session_id is None:
            return
        self._base_units_view.setPlainText(_render_units(session_id, self._system))

    def _on_base_delete_clicked(self) -> None:
        session_id = self._selected_session_id()
        if session_id is None:
            return
        current = self._base_session_list.currentItem()
        name = current.text() if current is not None else session_id
        if not _confirm_delete(self, name):
            return
        self._system.base.delete_session(session_id)
        self._refresh_base()

    # ── 诊断 ─────────────────────────────────────────────────────────────

    def _build_diagnostics_tab(self) -> QWidget:
        container = QWidget(self)

        self._diagnostics_view = PlainTextEdit(container)
        self._diagnostics_view.setReadOnly(True)
        self._diagnostics_view.setPlainText('点击"刷新诊断"查看当前记忆质量概况。')

        refresh_button = PrimaryPushButton("刷新诊断", container)
        refresh_button.clicked.connect(self._refresh_diagnostics)

        layout = QVBoxLayout(container)
        layout.addWidget(self._diagnostics_view)
        layout.addWidget(refresh_button)
        return container

    def _refresh_diagnostics(self) -> None:
        self._diagnostics_view.setPlainText(_render_diagnostics(self._system))
