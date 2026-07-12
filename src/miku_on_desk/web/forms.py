"""AppSettings 的 HTML 表单渲染/解析：Providers（API Key/分层模型）与 Persona
（人格设定）——kiosk 精简本机面板不呈现这些复杂配置，改由这里承载的局域网 Web 页面编辑。
用 f-string + ``html.escape`` 拼接，不引入模板引擎，延续 ``face/hooks/server.py``
"不为这类低频、结构简单的场景引入新框架"的风格。

Permissions/Skills/Memory/MCP 等配置目前仍需通过桌面版 ``SettingsPanel`` 或直接编辑
磁盘上的 settings.json 调整——这里先覆盖 Phase 3 验收必需的 Provider/Persona 两块，
其余留作后续按需扩展，不在这一轮里一次性搬完。
"""

from __future__ import annotations

import html

from miku_on_desk.config.settings import AppSettings, ModelTier, PersonaConfig, ProviderConfig

_TIERS: tuple[ModelTier, ...] = (ModelTier.MINI, ModelTier.FAST, ModelTier.MEDIUM, ModelTier.HEAVY)
_PROVIDER_LABELS: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "gemini": "Gemini",
    "qwen": "Qwen",
}

_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>miku-on-desk 设置</title>
<style>
body {{ font-family: sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }}
fieldset {{ margin-bottom: 1.5rem; }}
label {{ display: block; margin-top: 0.5rem; }}
input[type=text], input[type=password], textarea {{ width: 100%; box-sizing: border-box; }}
.tier-row {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
.tier-row label {{ flex: 1; min-width: 120px; }}
.banner {{ background: #e6ffed; border: 1px solid #34c759; padding: 0.5rem 1rem;
  margin-bottom: 1rem; }}
</style>
</head>
<body>
<h1>miku-on-desk 设置</h1>
{banner}
<form method="POST">
{provider_fieldsets}
{persona_fieldset}
<button type="submit">保存</button>
</form>
</body>
</html>
"""

_SAVED_BANNER = (
    '<p class="banner">配置已保存。部分改动（如新增/更换 Provider）需要重启'
    " miku-on-desk-kiosk 才能生效。</p>"
)


def _e(value: str) -> str:
    return html.escape(value, quote=True)


def _render_provider_fieldset(name: str, config: ProviderConfig) -> str:
    label = _PROVIDER_LABELS.get(name, name)
    tier_inputs = "\n".join(
        f'<label>{tier.value} 模型<input type="text" '
        f'name="provider_{name}_model_{tier.value}" value="{_e(config.models.get(tier, ""))}">'
        "</label>"
        for tier in _TIERS
    )
    api_key_input = (
        f'<input type="password" name="provider_{name}_api_key" '
        f'value="{_e(config.api_key or "")}">'
    )
    base_url_input = (
        f'<input type="text" name="provider_{name}_base_url" '
        f'value="{_e(config.base_url or "")}">'
    )
    return f"""<fieldset>
<legend>{_e(label)}</legend>
<label>API Key{api_key_input}</label>
<label>Base URL（可留空用默认）{base_url_input}</label>
<div class="tier-row">
{tier_inputs}
</div>
</fieldset>"""


def _render_persona_fieldset(persona: PersonaConfig) -> str:
    personality_input = (
        f'<textarea name="persona_personality" rows="3">{_e(persona.personality)}</textarea>'
    )
    return f"""<fieldset>
<legend>人格设定</legend>
<label>名字<input type="text" name="persona_name" value="{_e(persona.name)}"></label>
<label>角色定位<input type="text" name="persona_role" value="{_e(persona.role)}"></label>
<label>说话风格{personality_input}</label>
</fieldset>"""


def render_settings_page(settings: AppSettings, *, saved: bool = False) -> str:
    provider_fieldsets = "\n".join(
        _render_provider_fieldset(name, getattr(settings.model_router, name))
        for name in _PROVIDER_LABELS
    )
    return _PAGE_TEMPLATE.format(
        banner=_SAVED_BANNER if saved else "",
        provider_fieldsets=provider_fieldsets,
        persona_fieldset=_render_persona_fieldset(settings.persona),
    )


def apply_settings_form(settings: AppSettings, fields: dict[str, list[str]]) -> AppSettings:
    """把解析后的表单字段（``urllib.parse.parse_qs`` 的输出格式）合并进一份
    ``settings`` 的深拷贝，返回新对象——不修改传入的 ``settings``，调用方决定何时落盘。
    只更新表单里渲染出的字段，其余（Permissions/Skills/Memory/MCP 等）原样保留。
    """
    updated = settings.model_copy(deep=True)

    def _first(key: str) -> str | None:
        values = fields.get(key)
        return values[0] if values else None

    for name in _PROVIDER_LABELS:
        provider = getattr(updated.model_router, name)
        api_key = _first(f"provider_{name}_api_key")
        if api_key is not None:
            provider.api_key = api_key.strip() or None
        base_url = _first(f"provider_{name}_base_url")
        if base_url is not None:
            provider.base_url = base_url.strip() or None
        for tier in _TIERS:
            model_name = _first(f"provider_{name}_model_{tier.value}")
            if model_name is None:
                continue
            model_name = model_name.strip()
            if model_name:
                provider.models[tier] = model_name
            else:
                provider.models.pop(tier, None)

    persona_name = _first("persona_name")
    if persona_name is not None and persona_name.strip():
        updated.persona.name = persona_name.strip()
    persona_role = _first("persona_role")
    if persona_role is not None and persona_role.strip():
        updated.persona.role = persona_role.strip()
    persona_personality = _first("persona_personality")
    if persona_personality is not None and persona_personality.strip():
        updated.persona.personality = persona_personality.strip()

    return updated
