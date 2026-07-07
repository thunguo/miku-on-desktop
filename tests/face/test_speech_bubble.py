"""SpeechBubble 的回归测试：纯 Qt widget 状态机验证，不接 bridge/事件总线——那部分接线
在 overlay_window 的测试里覆盖。
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from miku_on_desk.face.ui.speech_bubble import (
    _MAX_BUBBLE_HEIGHT,
    _MIN_BUBBLE_HEIGHT,
    SpeechBubble,
)


def test_show_speech_sets_text_and_hides_buttons_and_shows_widget(qapp: QApplication) -> None:
    bubble = SpeechBubble()

    bubble.show_speech("你好")

    assert bubble.current_text() == "你好"
    assert bubble.is_awaiting_confirmation() is False
    assert bubble.isVisible() is True


def test_append_speech_accumulates_text(qapp: QApplication) -> None:
    bubble = SpeechBubble()

    bubble.show_speech("你")
    bubble.append_speech("好")

    assert bubble.current_text() == "你好"


def test_show_confirmation_sets_question_and_shows_buttons(qapp: QApplication) -> None:
    bubble = SpeechBubble()

    bubble.show_confirmation("要点击这个按钮吗？")

    assert bubble.current_text() == "要点击这个按钮吗？"
    assert bubble.is_awaiting_confirmation() is True


def test_clear_resets_text_and_hides_buttons_and_widget(qapp: QApplication) -> None:
    bubble = SpeechBubble()
    bubble.show_confirmation("要点击这个按钮吗？")

    bubble.clear()

    assert bubble.current_text() == ""
    assert bubble.is_awaiting_confirmation() is False
    assert bubble.isVisible() is False


def test_clicking_yes_button_emits_true_and_hides_buttons(qapp: QApplication) -> None:
    bubble = SpeechBubble()
    bubble.show_confirmation("要点击这个按钮吗？")
    decisions: list[bool] = []
    bubble.decision_made.connect(decisions.append)

    bubble._yes_button.click()

    assert decisions == [True]
    assert bubble.is_awaiting_confirmation() is False


def test_clicking_no_button_emits_false(qapp: QApplication) -> None:
    bubble = SpeechBubble()
    bubble.show_confirmation("要点击这个按钮吗？")
    decisions: list[bool] = []
    bubble.decision_made.connect(decisions.append)

    bubble._no_button.click()

    assert decisions == [False]


def test_show_speech_while_awaiting_confirmation_hides_buttons(qapp: QApplication) -> None:
    bubble = SpeechBubble()
    bubble.show_confirmation("要点击这个按钮吗？")

    bubble.show_speech("换个话题")

    assert bubble.is_awaiting_confirmation() is False


def test_ideal_height_for_empty_text_is_clamped_to_minimum(qapp: QApplication) -> None:
    bubble = SpeechBubble()

    assert bubble.ideal_height(300) == _MIN_BUBBLE_HEIGHT


def test_ideal_height_grows_with_longer_wrapped_text(qapp: QApplication) -> None:
    bubble = SpeechBubble()
    bubble.show_speech("短")
    short_height = bubble.ideal_height(120)
    bubble.show_speech("这是一段很长的文字 " * 20)

    long_height = bubble.ideal_height(120)

    assert long_height > short_height


def test_ideal_height_is_clamped_to_maximum_for_extremely_long_text(qapp: QApplication) -> None:
    bubble = SpeechBubble()

    bubble.show_speech("超长文字段落 " * 200)

    assert bubble.ideal_height(120) == _MAX_BUBBLE_HEIGHT


def test_ideal_height_reserves_extra_space_for_confirmation_buttons(qapp: QApplication) -> None:
    speech_bubble = SpeechBubble()
    speech_bubble.show_speech("同样的文字")
    speech_height = speech_bubble.ideal_height(200)

    confirm_bubble = SpeechBubble()
    confirm_bubble.show_confirmation("同样的文字")
    confirm_height = confirm_bubble.ideal_height(200)

    assert confirm_height > speech_height


def test_has_non_empty_stylesheet(qapp: QApplication) -> None:
    bubble = SpeechBubble()

    assert bubble.styleSheet().strip() != ""
