"""``screen_analyze`` 工具：理解当前屏幕内容，返回 OS accessibility tree 枚举到的 `elements[]`
列表。

`query` 有实际内容时，先在 `elements` 里做纯 Python 文本模糊匹配（`_match_score`，无 LLM
参与），命中的元素附加 `match_score` 字段并排到前面；完全匹配不到时才退回 Qwen3-VL 原生视觉
定位（`vision_grounding` 字段，单轮，直接输出归一化坐标点）——不使用 OCR，也不使用其它
provider 的通用网格标注兜底：accessibility 枚举不到语义、又匹配不到文本时，唯一能"看懂截图"
的就是 Qwen 这类视觉模型本身，没有必要为了兼容"没配 Qwen 但配了其它支持读图的 provider"这种
场景去多维护一套网格 Set-of-Mark 流程。因此视觉定位调用固定要求 `ProviderName.QWEN`
（`ModelRouter.resolve_provider()`），未配置 Qwen 时给出清晰的错误提示，而不是静默换用一个
根本不支持这种坐标定位协议的 provider。视觉定位失败（网络错误、返回格式不对等）不会让整个
工具调用报错——仍返回已收集到的 `elements`，只是 `vision_grounding.found` 为 false 并附带
`error` 说明。

提供 `pid` 时会用其对应窗口边界裁剪截图（ROI），减少无关区域、加快视觉定位处理、提高准确率，
`elements` 也只在提供 `pid` 时才会枚举（accessibility API 需要目标进程 pid）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any

from PIL import Image
from pydantic import BaseModel, ValidationError

from miku_on_desk.brain.model_router import ModelRouter, NoModelAvailableError
from miku_on_desk.brain.providers.base import (
    ImageBlock,
    Message,
    Provider,
    TextBlock,
    ToolDefinition,
)
from miku_on_desk.brain.tools.registry import (
    ToolExecutionError,
    ToolHandler,
    ToolRegistration,
    ToolRegistry,
)
from miku_on_desk.config.settings import ModelTier, ProviderName
from miku_on_desk.hands_eyes.backend import PlatformBackend, element_to_dict
from miku_on_desk.hands_eyes.capture import capture_origin, capture_screen, crop_to_bounds
from miku_on_desk.hands_eyes.vision_fallback import encode_image_as_base64

logger = logging.getLogger(__name__)

_SCREEN_ANALYZE_TOOL_NAME = "screen_analyze"
_VISION_TIER = ModelTier.FAST
_NOT_FOUND_NOTE = "未能在截图中定位到匹配的元素，建议换一种描述或换个 pid 重试。"

_POINT_SYSTEM_PROMPT = (
    "你是屏幕元素定位器。给你一张截图和一个查询描述，找到最匹配的元素中心点。"
    "坐标用 0-1000 的归一化整数表示（0 是左/上边缘，1000 是右/下边缘）。"
    '严格输出 JSON：{"found": true 或 false, "point_2d": [x, y]}，不要输出其他文字。'
    "found 为 false 时 point_2d 可以省略。"
)

_MATCH_THRESHOLD = 0.6


class ScreenAnalyzeInput(BaseModel):
    pid: int | None = None
    query: str | None = None


class _PointSelection(BaseModel):
    found: bool
    point_2d: tuple[int, int] | None = None


def _normalize_for_match(text: str) -> str:
    return "".join(text.split()).lower().replace("　", "")


def _match_score(query: str, text: str) -> float:
    q, t = _normalize_for_match(query), _normalize_for_match(text)
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0
    if q in t or t in q:
        overlap = min(len(q), len(t)) / max(len(q), len(t))
        return 0.85 + 0.15 * overlap
    return SequenceMatcher(None, q, t).ratio()


def _try_parse_json(raw: str | None) -> Any | None:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_json_substring(raw: str) -> str | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw[start : end + 1]


def _parse_point_selection(raw: str) -> _PointSelection:
    data = _try_parse_json(raw)
    if data is None:
        data = _try_parse_json(_extract_json_substring(raw))
    if data is None:
        raise ToolExecutionError(f"视觉模型未返回合法 JSON：{raw!r}")
    try:
        return _PointSelection.model_validate(data)
    except ValidationError as exc:
        raise ToolExecutionError(f"视觉模型返回的坐标点字段不合法：{exc}") from exc


async def _run_point_grounding(
    *,
    query: str,
    provider: Provider,
    model_id: str,
    image: Image.Image,
    roi_offset: tuple[int, int],
) -> dict[str, Any]:
    media_type, data = encode_image_as_base64(image)
    message = Message(
        role="user",
        content=[ImageBlock(media_type=media_type, data=data), TextBlock(text=f"查询：{query}")],
    )
    result = await provider.stream(
        model=model_id, system=_POINT_SYSTEM_PROMPT, messages=[message], tools=[]
    )
    if not result.success:
        raise ToolExecutionError(f"视觉定位调用失败：{result.error}")
    selection = _parse_point_selection(result.content)
    if not selection.found or selection.point_2d is None:
        return {"found": False, "note": _NOT_FOUND_NOTE}

    norm_x, norm_y = selection.point_2d
    width, height = image.size
    x = round(norm_x / 1000 * width) + roi_offset[0]
    y = round(norm_y / 1000 * height) + roi_offset[1]
    return {"found": True, "x": x, "y": y, "confidence": "native_point"}


async def _run_vision_grounding(
    *,
    query: str,
    router: ModelRouter,
    providers: Mapping[ProviderName, Provider],
    image: Image.Image,
    roi_offset: tuple[int, int],
) -> dict[str, Any]:
    try:
        resolved = router.resolve_provider(ProviderName.QWEN, _VISION_TIER)
    except NoModelAvailableError as exc:
        raise ToolExecutionError(f"视觉定位不可用：未配置 Qwen3-VL（{exc}）") from exc

    provider = providers.get(resolved.provider)
    if provider is None:
        raise ToolExecutionError('视觉定位所需的 provider "qwen" 未注册')

    logger.debug("视觉定位使用 provider=%s model=%s", resolved.provider.value, resolved.model_id)
    return await _run_point_grounding(
        query=query,
        provider=provider,
        model_id=resolved.model_id,
        image=image,
        roi_offset=roi_offset,
    )


def _make_screen_analyze_handler(
    *,
    backend: PlatformBackend,
    router: ModelRouter,
    providers: Mapping[ProviderName, Provider],
    match_threshold: float = _MATCH_THRESHOLD,
) -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            parsed = ScreenAnalyzeInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, capture_screen)
        roi_offset = await loop.run_in_executor(None, capture_origin)

        if parsed.pid is not None:
            bounds = await loop.run_in_executor(None, backend.get_window_bounds, parsed.pid)
            if bounds is not None:
                image, roi_offset = crop_to_bounds(image, bounds, roi_offset)

        elements: list[dict[str, Any]] = []
        if parsed.pid is not None:
            try:
                raw_elements = await loop.run_in_executor(None, backend.list_elements, parsed.pid)
                elements = [element_to_dict(e) for e in raw_elements]
            except Exception as exc:
                logger.warning("accessibility 枚举失败（pid=%s）：%s", parsed.pid, exc)
        logger.debug("accessibility 枚举到 %d 个元素（pid=%s）", len(elements), parsed.pid)

        payload: dict[str, Any] = {"elements": elements}

        if parsed.query:
            scored = [
                (element, _match_score(parsed.query, str(element.get("text", ""))))
                for element in elements
            ]
            matched = [
                {**element, "match_score": score}
                for element, score in scored
                if score >= match_threshold
            ]
            if matched:
                logger.debug("query 命中 %d 个元素，跳过视觉定位", len(matched))
                matched.sort(key=lambda e: e["match_score"], reverse=True)
                unmatched = [element for element, score in scored if score < match_threshold]
                payload["elements"] = matched + unmatched
            else:
                logger.debug("query 未命中任何元素，退回视觉定位")
                try:
                    payload["vision_grounding"] = await _run_vision_grounding(
                        query=parsed.query,
                        router=router,
                        providers=providers,
                        image=image,
                        roi_offset=roi_offset,
                    )
                except Exception as exc:
                    logger.warning("视觉定位失败，保留已收集的 elements：%s", exc)
                    payload["vision_grounding"] = {"found": False, "error": str(exc)}

        return json.dumps(payload, ensure_ascii=False)

    return handler


def register_screen_analyze_tool(
    *,
    backend: PlatformBackend,
    router: ModelRouter,
    providers: Mapping[ProviderName, Provider],
    registry: ToolRegistry,
    match_threshold: float = _MATCH_THRESHOLD,
) -> None:
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name=_SCREEN_ANALYZE_TOOL_NAME,
                description=(
                    "理解当前屏幕内容。提供 pid 时返回该进程 accessibility tree 中可交互元素的 "
                    "elements 列表（不提供 pid 则为空列表）。提供 query 时会额外在 elements 里做"
                    "文本模糊匹配并标注 match_score（按分数降序排在前面）；完全匹配不到时会退回 "
                    "Qwen3-VL 视觉定位直接给出估算坐标（vision_grounding 字段，附 "
                    "found/confidence 标记，找不到时 found 为 false）。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "pid": {
                            "type": "integer",
                            "description": (
                                "目标进程 pid，通常来自 computer_input 的 open_app 返回值；"
                                "提供后还会用于裁剪截图到该窗口范围并枚举其 accessibility 元素"
                            ),
                        },
                        "query": {
                            "type": "string",
                            "description": (
                                "描述想找的目标元素（如\"确定按钮\"），会在 elements 列表里做"
                                "文本模糊匹配并标注 match_score；找不到匹配时才会退回 Qwen3-VL "
                                "视觉定位直接给出估算坐标（vision_grounding 字段，附 "
                                "found/confidence 标记）"
                            ),
                        },
                    },
                },
            ),
            handler=_make_screen_analyze_handler(
                backend=backend, router=router, providers=providers, match_threshold=match_threshold
            ),
        )
    )
