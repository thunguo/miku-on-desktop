"""局域网 Web 管理页面：读写复用现有 ``AppSettings``/``load_settings_with_vault``/
``save_settings_with_vault``，参照 ``face/hooks/server.py`` 的实现风格，用标准库
``http.server.ThreadingHTTPServer``（不引入 FastAPI/Flask 等新依赖）。只由
``kiosk_main.py`` 启动，桌面入口 ``main.py`` 不 import 这个模块。

保存后不做进程内热重载/文件监听——改动需要重启 ``miku-on-desk-kiosk`` 才能生效，这是
刻意选的最简方案：Provider/Persona 这类配置本来就不是高频改动项，先不为热更新引入额外
复杂度（``watchfiles`` 监听 + 跨线程通知 Brain 重新装配 provider/router），等真正需要
时再补。

没有做任何身份验证——局域网内任何人都能看到/修改这里的表单，包括 Provider API Key。
``HookServer`` 的 token 鉴权模式在这里没有直接复用，因为那个 token 是给"本机安装的
CLI 工具"用的机器凭证，不是给"局域网里另一台设备上的人"用的用户凭证；这里更合适的是
一个用户自己设的访问密码，属于后续可以按需补的加固项，不在这一轮里做。

``SecretVault`` 内部的 sqlite3 连接是线程绑定的（同一个连接不能跨线程复用），而
``ThreadingHTTPServer`` 每个请求都在新线程里处理——因此这里不接受一个共享的
``SecretVault`` 实例，而是接受 ``(db_path, key_path)`` 路径元组，每次处理请求时在
当前线程内现开一个 vault，用完即关，天然避免跨线程复用同一个 sqlite3 连接。
"""

from __future__ import annotations

import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from miku_on_desk.brain.secrets.vault import SecretVault
from miku_on_desk.config.settings import load_settings_with_vault, save_settings_with_vault
from miku_on_desk.web.forms import apply_settings_form, render_settings_page

logger = logging.getLogger(__name__)


def _make_handler(
    settings_path: Path, vault_paths: tuple[Path, Path]
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass  # 静默：跟 face/hooks/server.py 一样，不需要访问日志噪音

        def do_GET(self) -> None:
            vault = SecretVault(*vault_paths)
            try:
                settings = load_settings_with_vault(settings_path, vault)
            finally:
                vault.close()
            self._respond_html(render_settings_page(settings))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            # keep_blank_values=True：留空一个字段（如清空 API Key）也要能生效，
            # parse_qs 默认会直接丢弃空值字段，那样清空操作在表单里永远提交不出去。
            fields = parse_qs(body, keep_blank_values=True)
            vault = SecretVault(*vault_paths)
            try:
                settings = load_settings_with_vault(settings_path, vault)
                updated = apply_settings_form(settings, fields)
                save_settings_with_vault(updated, settings_path, vault)
            finally:
                vault.close()
            self._respond_html(render_settings_page(updated, saved=True))

        def _respond_html(self, body_text: str) -> None:
            body = body_text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class SettingsServer:
    """``port=0`` 让操作系统分配空闲端口（供测试用），真实运行时传固定端口。

    默认监听 ``0.0.0.0``（跟只监听 ``127.0.0.1`` 的 ``HookServer`` 不同）——这个服务
    本来就是设计给局域网内其它设备（手机/电脑浏览器）访问的，监听本机回环地址会让它
    完全打不开。
    """

    def __init__(
        self,
        settings_path: Path,
        vault_paths: tuple[Path, Path],
        *,
        host: str = "0.0.0.0",
        port: int = 8766,
    ) -> None:
        handler = _make_handler(settings_path, vault_paths)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_port

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
