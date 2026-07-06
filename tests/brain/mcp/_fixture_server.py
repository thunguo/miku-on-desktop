"""供 MCP 集成测试使用的最小 server：暴露 echo/fail 两个工具（远程 transport 下另加
echo_header，用于验证自定义 HTTP header 有没有被 client 正确带上）。

不是 mock——是一个真实、可被官方 SDK 的 `stdio_client`/`sse_client`/`streamable_http_client`
当子进程/远程 server 连接、走完整 MCP 握手（initialize → tools/list → tools/call）的独立
进程，用来验证 ``MCPServerConnection``/``MCPHost`` 真的能对上官方 SDK 的线上协议行为，而不是
对着一个手写的假连接对象断言。

可通过命令行参数 ``--transport``/``--port`` 选择 stdio/sse/streamable-http 三种 transport
之一启动，默认沿用原来的无参数 stdio 调用方式（``tests/brain/mcp/test_host.py`` 与
``test_client.py`` 现有用法不受影响）。

``echo_header`` 读取自定义 header 的方式：给工具函数加一个 ``ctx: Context`` 参数，FastMCP
按类型注解自动注入当前请求的 ``Context``；``ctx.request_context.request`` 是 SSE/Streamable
HTTP 两种远程 transport 下框架无关的原始 Starlette ``Request`` 对象（stdio 下恒为
``None``，因为没有 HTTP 请求）——直接通过 spike 脚本对着真实起的 streamable-http server 验证
过这条链路能拿到 client 端 `headers=` 传入的值,不是凭空猜测的 API。
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import Context, FastMCP


def build_server(transport: str, port: int | None = None) -> FastMCP:
    kwargs = {} if port is None else {"port": port}
    mcp = FastMCP("fixture", **kwargs)  # type: ignore[arg-type]

    @mcp.tool()
    def echo(text: str) -> str:
        """原样返回输入文本。"""
        return text

    @mcp.tool()
    def fail(reason: str) -> str:
        """总是抛出异常，用于验证 isError 结果路径。"""
        raise ValueError(reason)

    if transport != "stdio":

        @mcp.tool()
        def echo_header(header_name: str, ctx: Context) -> str:  # type: ignore[type-arg]
            """返回上一次请求里指定 HTTP header 的值，用于验证自定义 header 是否透传成功。"""
            request = ctx.request_context.request
            if request is None:
                return "<no-request>"
            return str(request.headers.get(header_name, "<missing>"))

    return mcp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="stdio")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    server = build_server(args.transport, args.port)
    server.run(transport=args.transport)
