"""离线一次性运行的桌宠像素画生成工具 CLI 入口：真正的生成流水线在
``miku_on_desk.character_generation`` 里，供本脚本与 GUI 的角色创建对话框共享。

不注册为 console-script：需要真实 API 计费，且极少运行，与随 app 一起分发的代码
性质不同。用法：``uv run python scripts/generate_pet_sprites.py --help``。

设计角色而非复刻任何现有模型：本脚本生成的是 AI 原创像素画，不以
`assets/legacy_pmx/` 下任何模型的渲染截图作为生成参考图（详见
`assets/legacy_pmx/README.md` 中的授权说明）。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from miku_on_desk.character_generation import (
    GenerationConfig,
    GenerationProgress,
    generate_character,
)

_DEFAULT_CHARACTER_DESCRIPTION = (
    "一只原创像素风虚拟歌手偶像桌宠角色：青绿色双马尾（发梢系深粉色蝴蝶结）、圆脸大眼睛、"
    "超Q版比例（约2头身，头部占身体一半以上）、简洁清爽的水手服风格未来感服装，配色以青绿/"
    "白/浅灰为主，整体呆萌可爱"
)


def _format_progress(progress: GenerationProgress) -> str:
    if progress.stage == "reference":
        return "生成基准参考图..."
    if progress.stage == "strip":
        return (
            f"生成状态动画 {progress.completed_states}/{progress.total_states}："
            f"{progress.detail}"
        )
    if progress.stage == "assemble":
        return "拼装 spritesheet..."
    return "运行 QA 检查..."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="miku_pixel", help="宠物资产名称，决定输出子目录")
    parser.add_argument(
        "--description",
        default=_DEFAULT_CHARACTER_DESCRIPTION,
        help="角色设计的自然语言描述，用于生成基准参考图",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="输出目录，默认 assets/pets/<name>"
    )
    parser.add_argument("--model", default="gpt-image-1")
    parser.add_argument("--api-key", default=None, help="默认读取 OPENAI_API_KEY 环境变量")
    parser.add_argument("--base-url", default=None, help="自定义 API 中转地址，默认官方地址")
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=None,
        help="可选：上传一张参考图，让基准参考图结合该图与文字描述生成，而非纯文字生成",
    )
    parser.add_argument(
        "--background",
        default="transparent",
        choices=("transparent", "opaque", "auto"),
        help="传给图像生成 API 的 background 参数，部分模型/中转不支持 transparent",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = args.output_dir or repo_root / "assets" / "pets" / args.name

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "错误：未提供 API Key，请传入 --api-key 或设置 OPENAI_API_KEY 环境变量", file=sys.stderr
        )
        return 1

    config = GenerationConfig(
        pet_name=args.name,
        description=args.description,
        output_dir=output_dir,
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        reference_image_path=args.reference_image,
        background=args.background,
    )

    _sheet, _meta, problems = generate_character(
        config, on_progress=lambda progress: print(_format_progress(progress))
    )
    for problem in problems:
        print(f"QA 警告：{problem}", file=sys.stderr)

    sheet_path = config.output_dir / "spritesheet.png"
    meta_path = config.output_dir / "pet.json"
    print(f"已写入 {sheet_path}")
    print(f"已写入 {meta_path}")

    if problems:
        print(
            f"QA 未完全通过（{len(problems)} 项警告），建议人工检查后决定是否重新生成",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
