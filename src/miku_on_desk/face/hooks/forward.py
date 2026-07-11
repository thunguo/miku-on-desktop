"""Codex CLI / Gemini CLI 本地 command hook → 本项目 hook sidecar 的转发层。

Codex CLI 与 Gemini CLI 的 hook 机制目前都只支持本地 ``command`` 类型执行、JSON 走
stdin（没有 Claude Code 那种可以直接配置 URL 的 ``http`` hook 类型），所以需要一个能
在 PATH 上被调用的小程序，读 stdin 的原始 JSON、加上来源标记后转发给
``face/hooks/server.py`` 的 ``/pet-event``。见 ``pyproject.toml`` 的
``miku-on-desk-hook-forward`` console script。

转发失败（sidecar 没启动、网络异常等）永远不能影响调用方 CLI 自身的 hook 判定——本
程序只在 stderr 记一行日志,始终以退出码 0 收场，stdout 保持空,避免被 Codex/Gemini 的
"parse stdout JSON 作为决策"语义误读成某种 block/override 指令。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from argparse import ArgumentParser

_TIMEOUT_SECONDS = 3.0


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="miku-on-desk-hook-forward")
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--source", default="unknown")
    args = parser.parse_args(argv)

    raw = sys.stdin.read()
    body = _with_source(raw, args.source)
    request = urllib.request.Request(
        args.url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.token}",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"miku-on-desk hook forward failed: {exc}", file=sys.stderr)
    return 0


def _with_source(raw: str, source: str) -> bytes:
    """给转发的 payload 补一个 ``source`` 字段,让 sidecar 知道是哪个 CLI 发来的事件；
    payload 本身格式非法时原样转发,交给 ``HookEvent.from_raw`` 的 JSON 解析去处理失败。
    """
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return raw.encode("utf-8")
    if not isinstance(payload, dict):
        return raw.encode("utf-8")
    payload.setdefault("source", source)
    return json.dumps(payload).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
