"""screen_analyze 工具注册的回归测试：假 PlatformBackend + 假 Provider，不碰真实
accessibility API/截图/LLM——那些分别在 hands_eyes 自己的测试里覆盖。

核心回归场景是当年触发这次重构的 bug 本身：视觉定位（query 有值但 accessibility elements
里无匹配时的兜底）必须固定调用 Qwen 原生 point-grounding（单轮，无网格叠加），即使配置里
还有其它 provider 在同一层级也配置了模型——不能被 `ModelRouter.resolve()` 的 provider
枚举顺序抢先命中。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import (
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    ToolDefinition,
    ToolUseBlock,
)
from miku_on_desk.brain.tools.builtin import screen_analyze as screen_analyze_module
from miku_on_desk.brain.tools.builtin.screen_analyze import register_screen_analyze_tool
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistry
from miku_on_desk.config.settings import ModelRouterConfig, ModelTier, ProviderConfig, ProviderName
from miku_on_desk.hands_eyes.backend import ForegroundAppInfo, PlatformBackend, UIElement

_FAKE_SCREEN_SIZE = (1200, 800)


class _FakeBackend(PlatformBackend):
    def __init__(
        self,
        elements_by_pid: dict[int, list[UIElement]] | None = None,
        bounds_by_pid: dict[int, tuple[int, int, int, int]] | None = None,
    ) -> None:
        self._elements_by_pid = elements_by_pid or {}
        self._bounds_by_pid = bounds_by_pid or {}
        self.list_elements_calls: list[int] = []
        self.get_window_bounds_calls: list[int] = []

    def list_elements(self, pid: int) -> list[UIElement]:
        self.list_elements_calls.append(pid)
        return self._elements_by_pid.get(pid, [])

    def get_window_bounds(self, pid: int) -> tuple[int, int, int, int] | None:
        self.get_window_bounds_calls.append(pid)
        return self._bounds_by_pid.get(pid)

    def open_app(self, name: str) -> None:
        raise NotImplementedError

    def get_idle_seconds(self) -> float:
        return 0.0

    def get_foreground_app_info(self) -> ForegroundAppInfo | None:
        return None


class _FakeProvider(Provider):
    """按调用顺序依次返回 ``results`` 里的结果，用完后重复最后一个。"""

    def __init__(self, result: StreamResult | list[StreamResult]) -> None:
        self._results = [result] if isinstance(result, StreamResult) else list(result)
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        on_content: OnContent | None = None,
        on_thinking: OnThinking | None = None,
        idle_timeout_s: float = 120.0,
        hard_timeout_s: float = 600.0,
    ) -> StreamResult:
        self.calls.append({"model": model, "messages": list(messages), "tools": tools})
        index = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[index]


class _RaisingProvider(Provider):
    """stream() 直接抛出裸异常（非 ToolExecutionError），验证调用点的 except Exception 生效。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        on_content: OnContent | None = None,
        on_thinking: OnThinking | None = None,
        idle_timeout_s: float = 120.0,
        hard_timeout_s: float = 600.0,
    ) -> StreamResult:
        self.calls.append({"model": model})
        raise RuntimeError("网络连接失败")


def _make_registry(tmp_path: Path) -> ToolRegistry:
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    sandbox = PathSandbox(cwd=cwd, output_dir=tmp_path / "output", data_dir=tmp_path / "data")
    policy = PolicyEngine(
        trusted_mode=True,
        allowed_tools=frozenset(),
        denied_tools=frozenset(),
        default_decision=Decision.ALLOW,
        path_sandbox=sandbox,
        read_tracker=ReadTracker(),
    )
    return ToolRegistry(policy, ReadTracker())


def _qwen_router() -> ModelRouter:
    config = ModelRouterConfig()
    config.qwen = ProviderConfig(api_key="key", models={ModelTier.FAST: "qwen3-vl-plus"})
    return ModelRouter(config)


def _register_qwen(tmp_path: Path, backend: PlatformBackend, provider: Provider) -> ToolRegistry:
    registry = _make_registry(tmp_path)
    register_screen_analyze_tool(
        backend=backend,
        router=_qwen_router(),
        providers={ProviderName.QWEN: provider},
        registry=registry,
    )
    return registry


def _stub_capture(monkeypatch: pytest.MonkeyPatch, image: Image.Image | None = None) -> None:
    """默认截图/坐标原点打桩：绝大多数测试不关心真实截图内容，只需要一个稳定的假图。"""
    fallback_image = image or Image.new("RGB", _FAKE_SCREEN_SIZE)
    monkeypatch.setattr(screen_analyze_module, "capture_screen", lambda: fallback_image)
    monkeypatch.setattr(screen_analyze_module, "capture_origin", lambda: (0, 0))


async def test_returns_accessibility_elements_with_source_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    element = UIElement(role="button", label="确定", center_x=1, center_y=2, width=3, height=4)
    backend = _FakeBackend({42: [element]})
    provider = _FakeProvider(StreamResult(success=True, content="不应该被调用"))
    registry = _register_qwen(tmp_path, backend, provider)
    _stub_capture(monkeypatch)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"pid": 42}), session_id="s1"
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload["elements"] == [
        {
            "role": "button",
            "text": "确定",
            "center_x": 1,
            "center_y": 2,
            "width": 3,
            "height": 4,
            "source": "accessibility",
        }
    ]
    assert provider.calls == []


async def test_no_pid_returns_empty_elements_without_calling_accessibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    provider = _FakeProvider(StreamResult(success=True, content="不应该被调用"))
    registry = _register_qwen(tmp_path, backend, provider)
    _stub_capture(monkeypatch)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={}), session_id="s1"
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload["elements"] == []
    assert backend.list_elements_calls == []


async def test_query_match_skips_vision_grounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    element = UIElement(role="button", label="确定", center_x=1, center_y=2, width=3, height=4)
    backend = _FakeBackend({42: [element]})
    provider = _FakeProvider(StreamResult(success=True, content="不应该被调用"))
    registry = _register_qwen(tmp_path, backend, provider)
    _stub_capture(monkeypatch)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"pid": 42, "query": "确定"}),
        session_id="s1",
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload["elements"][0]["match_score"] == 1.0
    assert "vision_grounding" not in payload
    assert provider.calls == []


async def test_empty_string_query_is_treated_as_no_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    element = UIElement(role="button", label="确定", center_x=1, center_y=2, width=3, height=4)
    backend = _FakeBackend({42: [element]})
    provider = _FakeProvider(StreamResult(success=True, content="不应该被调用"))
    registry = _register_qwen(tmp_path, backend, provider)
    _stub_capture(monkeypatch)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"pid": 42, "query": ""}),
        session_id="s1",
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert "match_score" not in payload["elements"][0]
    assert "vision_grounding" not in payload
    assert provider.calls == []


async def test_accessibility_failure_degrades_silently_and_keeps_elements_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend({42: []})

    def _raise_accessibility_error(pid: int) -> list[UIElement]:
        raise RuntimeError("辅助功能权限未授予")

    monkeypatch.setattr(backend, "list_elements", _raise_accessibility_error)
    provider = _FakeProvider(StreamResult(success=True, content="不应该被调用"))
    registry = _register_qwen(tmp_path, backend, provider)
    _stub_capture(monkeypatch)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"pid": 42}), session_id="s1"
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload["elements"] == []


async def test_pid_with_window_bounds_crops_screenshot_for_vision_grounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bounds = (100, 100, 50, 50)
    backend = _FakeBackend({42: []}, bounds_by_pid={42: bounds})
    provider = _FakeProvider(
        StreamResult(success=True, content=json.dumps({"found": True, "point_2d": [500, 500]}))
    )
    registry = _register_qwen(tmp_path, backend, provider)
    image = Image.new("RGB", _FAKE_SCREEN_SIZE, color=(0, 0, 0))
    _stub_capture(monkeypatch, image=image)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"pid": 42, "query": "确定按钮"}),
        session_id="s1",
    )

    assert result.is_error is False
    assert backend.get_window_bounds_calls == [42]
    payload = json.loads(result.content)
    assert payload["vision_grounding"] == {
        "found": True,
        "x": 125,
        "y": 125,
        "confidence": "native_point",
    }


async def test_query_no_match_with_qwen_provider_uses_single_call_native_point_grounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    provider = _FakeProvider(
        StreamResult(success=True, content=json.dumps({"found": True, "point_2d": [500, 250]}))
    )
    registry = _register_qwen(tmp_path, backend, provider)
    image = Image.new("RGB", _FAKE_SCREEN_SIZE, color=(0, 0, 0))
    _stub_capture(monkeypatch, image=image)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"query": "确定按钮"}), session_id="s1"
    )

    assert result.is_error is False
    assert len(provider.calls) == 1
    payload = json.loads(result.content)
    assert payload["vision_grounding"] == {
        "found": True,
        "x": 600,
        "y": 200,
        "confidence": "native_point",
    }


async def test_qwen_point_grounding_not_found_returns_found_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    provider = _FakeProvider(StreamResult(success=True, content=json.dumps({"found": False})))
    registry = _register_qwen(tmp_path, backend, provider)
    image = Image.new("RGB", _FAKE_SCREEN_SIZE, color=(0, 0, 0))
    _stub_capture(monkeypatch, image=image)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"query": "确定按钮"}), session_id="s1"
    )

    assert result.is_error is False
    assert len(provider.calls) == 1
    payload = json.loads(result.content)
    vision = payload["vision_grounding"]
    assert vision["found"] is False
    assert "note" in vision
    assert "x" not in vision


async def test_vision_grounding_uses_qwen_even_when_other_provider_configured_at_same_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """核心回归测试：复现用户报告的 bug——Moonshot（openai 槽位）与 Qwen 同时配置在 FAST 层级时，
    视觉定位必须只调用 Qwen，不能被 `ModelRouter.resolve()` 的 provider 枚举顺序抢先命中。
    """
    backend = _FakeBackend()
    qwen_provider = _FakeProvider(
        StreamResult(success=True, content=json.dumps({"found": True, "point_2d": [500, 250]}))
    )
    other_provider = _FakeProvider(StreamResult(success=True, content="不应该被调用"))
    config = ModelRouterConfig()
    config.qwen = ProviderConfig(api_key="key", models={ModelTier.FAST: "qwen3-vl-plus"})
    config.openai = ProviderConfig(api_key="key", models={ModelTier.FAST: "moonshot-v1"})
    registry = _make_registry(tmp_path)
    register_screen_analyze_tool(
        backend=backend,
        router=ModelRouter(config),
        providers={ProviderName.QWEN: qwen_provider, ProviderName.OPENAI: other_provider},
        registry=registry,
    )
    image = Image.new("RGB", _FAKE_SCREEN_SIZE, color=(0, 0, 0))
    _stub_capture(monkeypatch, image=image)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"query": "确定按钮"}), session_id="s1"
    )

    assert result.is_error is False
    assert len(qwen_provider.calls) == 1
    assert other_provider.calls == []
    payload = json.loads(result.content)
    assert payload["vision_grounding"]["found"] is True


async def test_vision_grounding_degrades_when_qwen_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    other_provider = _FakeProvider(StreamResult(success=True, content="不应该被调用"))
    config = ModelRouterConfig()
    config.anthropic = ProviderConfig(api_key="key", models={ModelTier.FAST: "claude-haiku"})
    registry = _make_registry(tmp_path)
    register_screen_analyze_tool(
        backend=backend,
        router=ModelRouter(config),
        providers={ProviderName.ANTHROPIC: other_provider},
        registry=registry,
    )
    _stub_capture(monkeypatch)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"query": "确定按钮"}), session_id="s1"
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload["elements"] == []
    assert payload["vision_grounding"]["found"] is False
    assert "未配置 Qwen3-VL" in payload["vision_grounding"]["error"]
    assert other_provider.calls == []


async def test_vision_grounding_degrades_on_invalid_json_without_failing_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    element = UIElement(role="button", label="其它", center_x=1, center_y=2, width=3, height=4)
    backend = _FakeBackend({42: [element]})
    provider = _FakeProvider(StreamResult(success=True, content="这里有个按钮，大概在左上角"))
    registry = _register_qwen(tmp_path, backend, provider)
    image = Image.new("RGB", _FAKE_SCREEN_SIZE, color=(0, 0, 0))
    _stub_capture(monkeypatch, image=image)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"pid": 42, "query": "确定按钮"}),
        session_id="s1",
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload["elements"] != []
    assert payload["vision_grounding"]["found"] is False
    assert "error" in payload["vision_grounding"]


async def test_vision_grounding_degrades_on_unexpected_exception_without_failing_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    provider = _RaisingProvider()
    registry = _register_qwen(tmp_path, backend, provider)
    _stub_capture(monkeypatch)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="screen_analyze", input={"query": "确定按钮"}), session_id="s1"
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload["elements"] == []
    assert payload["vision_grounding"]["found"] is False
    assert "网络连接失败" in payload["vision_grounding"]["error"]
