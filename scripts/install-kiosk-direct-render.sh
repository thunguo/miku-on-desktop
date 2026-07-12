#!/usr/bin/env bash
set -euo pipefail

# 把当前 Pi OS 图形桌面替换为直接写 /dev/fb0 的 Miku kiosk。脚本只适用于本仓库当前
# 部署约定（/home/thunguo/miku-on-desk）；回退请运行 restore-desktop.sh。

readonly REPO_DIR="/home/thunguo/miku-on-desk"
readonly SERVICE_NAME="miku-kiosk.service"
readonly STATE_DIR="/etc/miku-on-desk"
readonly STATE_FILE="${STATE_DIR}/display-mode.env"

if [[ "${EUID}" -ne 0 ]]; then
    echo "请通过 sudo 运行：sudo $0" >&2
    exit 1
fi

if [[ ! -x "${REPO_DIR}/.venv/bin/miku-on-desk-kiosk" ]]; then
    echo "未找到 ${REPO_DIR}/.venv/bin/miku-on-desk-kiosk；请先在树莓派完成 uv sync。" >&2
    exit 1
fi

if [[ ! -r /dev/fb0 || ! -r /dev/input/event4 ]]; then
    echo "未发现预期的 framebuffer 或 ADS7846 触摸设备。" >&2
    exit 1
fi

install -d -m 0755 "${STATE_DIR}"
if systemctl is-enabled --quiet lightdm.service; then
    printf 'LIGHTDM_WAS_ENABLED=1\n' >"${STATE_FILE}"
else
    printf 'LIGHTDM_WAS_ENABLED=0\n' >"${STATE_FILE}"
fi

install -m 0644 "${REPO_DIR}/deploy/${SERVICE_NAME}" "/etc/systemd/system/${SERVICE_NAME}"
systemctl daemon-reload
systemctl disable --now lightdm.service
systemctl enable --now "${SERVICE_NAME}"

echo "已启用直接 framebuffer kiosk。日志：journalctl -u ${SERVICE_NAME} -f"
echo "回退桌面：sudo ${REPO_DIR}/scripts/restore-desktop.sh"
