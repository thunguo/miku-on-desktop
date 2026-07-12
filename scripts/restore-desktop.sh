#!/usr/bin/env bash
set -euo pipefail

# 停止 LinuxFB kiosk 并恢复安装前已启用的 LightDM 桌面，供 SSH 排障后的现场恢复使用。

readonly SERVICE_NAME="miku-kiosk.service"
readonly STATE_FILE="/etc/miku-on-desk/display-mode.env"

if [[ "${EUID}" -ne 0 ]]; then
    echo "请通过 sudo 运行：sudo $0" >&2
    exit 1
fi

systemctl disable --now "${SERVICE_NAME}" || true
if [[ -r "${STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_FILE}"
fi

if [[ "${LIGHTDM_WAS_ENABLED:-1}" == "1" ]]; then
    systemctl enable --now lightdm.service
    echo "已恢复 LightDM 桌面。"
else
    echo "安装前 LightDM 未启用；没有启动桌面。"
fi
