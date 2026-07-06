"""路径沙箱：校验工具要访问的文件路径是否落在允许的根目录集合内。

取舍是"宽松但不失控"：cwd、输出目录、用户数据目录这几个固定根之外，额外把用户主目录下几个
常见工作目录（``code``/``Desktop``/``Documents`` 等，仅当目录真实存在时才纳入）也放进允许
列表——让"去 ~/code 里改个文件"这种常见请求不需要用户先手动加一条自定义允许目录。POSIX 下
``/tmp`` 在 macOS 上是指向 ``/private/tmp`` 的符号链接，但 Python 的 ``tempfile.gettempdir()``
在两个平台上都已经返回规范路径，不需要额外处理这个符号链接。

拒绝时的原因文本特意写清楚"不要重试同一路径"：避免模型反复重试同一个已知会被拒绝的路径——
模型看到普通的 permission denied 容易在同一条被拒路径上反复重试，期待不同的结果。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from miku_on_desk.config.settings import EnvBootstrap

_HOME_SUBDIRS = (
    "code",
    "Code",
    "projects",
    "Projects",
    "Desktop",
    "Documents",
    "Downloads",
    "workspace",
    "Workspace",
    "dev",
    "Dev",
    "src",
)


@dataclass(frozen=True)
class PathSandboxResult:
    allowed: bool
    reason: str | None = None


def _normalize(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


class PathSandbox:
    """允许根目录集合在构造时一次性算好，后续 ``check`` 不重新扫描文件系统。"""

    def __init__(
        self,
        *,
        cwd: Path,
        output_dir: Path,
        data_dir: Path,
        extra_dirs: list[Path] | None = None,
    ) -> None:
        roots = {_normalize(cwd), _normalize(output_dir), _normalize(data_dir)}
        home = Path.home()
        for sub in _HOME_SUBDIRS:
            candidate = home / sub
            if candidate.exists():
                roots.add(_normalize(candidate))
        roots.add(_normalize(Path(tempfile.gettempdir())))
        for extra in extra_dirs or []:
            roots.add(_normalize(extra))
        self._roots = frozenset(roots)

    def check(self, path: Path) -> PathSandboxResult:
        normalized = _normalize(path)
        if normalized in self._roots or any(root in normalized.parents for root in self._roots):
            return PathSandboxResult(allowed=True)
        return PathSandboxResult(
            allowed=False,
            reason=(
                f'路径 "{path}" 不在允许的目录范围内。不要用同一路径重试 read_file/write_file，'
                "它会再次失败——改用 exec_command 配合 cat/ls 访问沙箱外的文件，"
                "或请用户在设置里把这个目录加入允许列表。"
            ),
        )


def default_path_sandbox(
    bootstrap: EnvBootstrap | None = None, extra_dirs: list[Path] | None = None
) -> PathSandbox:
    bootstrap = bootstrap or EnvBootstrap()
    data_dir = bootstrap.resolve_data_dir()
    return PathSandbox(
        cwd=Path.cwd(),
        output_dir=data_dir / "outputs",
        data_dir=data_dir,
        extra_dirs=extra_dirs,
    )
