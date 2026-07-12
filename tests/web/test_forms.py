"""``web/forms.py`` 的表单渲染/解析回归测试：渲染出的 HTML 转义、字段名与
``apply_settings_form`` 解析后写回 ``AppSettings`` 的映射关系。"""

from __future__ import annotations

from urllib.parse import parse_qs

from miku_on_desk.config.settings import AppSettings, ModelTier
from miku_on_desk.web.forms import apply_settings_form, render_settings_page


def test_render_settings_page_escapes_api_key_and_persona_fields() -> None:
    settings = AppSettings()
    settings.model_router.anthropic.api_key = '"><script>alert(1)</script>'
    settings.persona.name = "<b>初音</b>"

    page = render_settings_page(settings)

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;b&gt;初音&lt;/b&gt;" in page


def test_render_settings_page_without_saved_flag_omits_banner() -> None:
    page = render_settings_page(AppSettings(), saved=False)

    assert "配置已保存" not in page


def test_render_settings_page_with_saved_flag_shows_banner() -> None:
    page = render_settings_page(AppSettings(), saved=True)

    assert "配置已保存" in page


def test_apply_settings_form_updates_provider_api_key_and_models() -> None:
    settings = AppSettings()
    fields = parse_qs(
        "provider_anthropic_api_key=sk-test-123"
        "&provider_anthropic_model_medium=claude-sonnet-5"
        "&provider_anthropic_base_url=https%3A%2F%2Fexample.com"
    )

    updated = apply_settings_form(settings, fields)

    assert updated.model_router.anthropic.api_key == "sk-test-123"
    assert updated.model_router.anthropic.base_url == "https://example.com"
    assert updated.model_router.anthropic.models[ModelTier.MEDIUM] == "claude-sonnet-5"


def test_apply_settings_form_blank_api_key_clears_it() -> None:
    settings = AppSettings()
    settings.model_router.openai.api_key = "sk-existing"
    fields = parse_qs("provider_openai_api_key=", keep_blank_values=True)

    updated = apply_settings_form(settings, fields)

    assert updated.model_router.openai.api_key is None


def test_apply_settings_form_blank_model_tier_removes_it() -> None:
    settings = AppSettings()
    settings.model_router.openai.models[ModelTier.FAST] = "gpt-5-mini"
    fields = parse_qs("provider_openai_model_fast=", keep_blank_values=True)

    updated = apply_settings_form(settings, fields)

    assert ModelTier.FAST not in updated.model_router.openai.models


def test_apply_settings_form_updates_persona_fields() -> None:
    settings = AppSettings()
    fields = parse_qs("persona_name=新名字&persona_role=新角色&persona_personality=新风格")

    updated = apply_settings_form(settings, fields)

    assert updated.persona.name == "新名字"
    assert updated.persona.role == "新角色"
    assert updated.persona.personality == "新风格"


def test_apply_settings_form_does_not_mutate_input_settings() -> None:
    settings = AppSettings()
    fields = parse_qs("provider_anthropic_api_key=sk-test-123")

    apply_settings_form(settings, fields)

    assert settings.model_router.anthropic.api_key is None


def test_apply_settings_form_ignores_unrelated_fields() -> None:
    settings = AppSettings()
    settings.skills_dir = None
    fields = parse_qs("mcp_servers=should_be_ignored")

    updated = apply_settings_form(settings, fields)

    assert updated.skills_dir is None
