"""``web/settings_server.py`` 的回归测试：真实 HTTP 往返（``urllib.request``），覆盖
GET 渲染表单、POST 保存并复用 vault 加密、保存后原样保留未出现在表单里的字段。

每个测试自己起一个 ``port=0``（操作系统分配空闲端口）的 ``SettingsServer``，用完在
finally 里 ``stop()``，避免端口/线程泄漏到下一个测试——跟 ``test_server.py``
（``face/hooks/server.py`` 的测试）同一个模式。
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from miku_on_desk.brain.secrets.vault import SecretVault
from miku_on_desk.config.settings import (
    AppSettings,
    load_settings_with_vault,
    save_settings_with_vault,
)
from miku_on_desk.web.settings_server import SettingsServer


@pytest.fixture
def vault_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "secrets.db", tmp_path / "secrets.key"


@pytest.fixture
def vault(vault_paths: tuple[Path, Path]) -> Iterator[SecretVault]:
    v = SecretVault(*vault_paths)
    try:
        yield v
    finally:
        v.close()


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    path = tmp_path / "settings.json"
    AppSettings().save(path)
    return path


@pytest.fixture
def server(settings_path: Path, vault_paths: tuple[Path, Path]) -> Iterator[SettingsServer]:
    srv = SettingsServer(settings_path, vault_paths, host="127.0.0.1", port=0)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def _get(srv: SettingsServer) -> tuple[int, str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/", timeout=5) as response:
        return response.status, response.read().decode("utf-8")


def _post(srv: SettingsServer, form: dict[str, str]) -> tuple[int, str]:
    body = urllib.parse.urlencode(form).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{srv.port}/",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read().decode("utf-8")


def test_port_is_assigned_by_os_and_usable(server: SettingsServer) -> None:
    assert server.port != 0


def test_get_returns_html_form_with_current_settings(
    server: SettingsServer, settings_path: Path, vault: SecretVault
) -> None:
    settings = load_settings_with_vault(settings_path, vault)
    settings.persona.name = "测试角色"
    save_settings_with_vault(settings, settings_path, vault)

    status, body = _get(server)

    assert status == 200
    assert "测试角色" in body
    assert "<form" in body


def test_post_saves_provider_api_key_encrypted_via_vault(
    server: SettingsServer, settings_path: Path, vault: SecretVault
) -> None:
    status, body = _post(
        server,
        {
            "provider_anthropic_api_key": "sk-test-secret",
            "provider_anthropic_model_medium": "claude-sonnet-5",
        },
    )

    assert status == 200
    assert "配置已保存" in body

    reloaded = load_settings_with_vault(settings_path, vault)
    assert reloaded.model_router.anthropic.api_key == "sk-test-secret"

    on_disk = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "sk-test-secret" not in json.dumps(on_disk)
    assert on_disk["model_router"]["anthropic"]["api_key"].startswith("vault-ref:")


def test_post_preserves_fields_not_present_in_form(
    server: SettingsServer, settings_path: Path, vault: SecretVault
) -> None:
    settings = load_settings_with_vault(settings_path, vault)
    settings.skills_dir = Path("/tmp/my-skills")
    save_settings_with_vault(settings, settings_path, vault)

    _post(server, {"persona_name": "新名字"})

    reloaded = load_settings_with_vault(settings_path, vault)
    assert reloaded.persona.name == "新名字"
    assert reloaded.skills_dir == Path("/tmp/my-skills")
