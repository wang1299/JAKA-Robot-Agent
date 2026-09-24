#!/bin/bash
# 机器人（上位机）到模型服务器的 SSH 隧道，支持自动重连。
#
# 用法：
#   ./tunnel.sh start
#   ./tunnel.sh stop
#   ./tunnel.sh status
#   ./tunnel.sh restart
#
# 前置（只需做一次）：配置免密登录，否则自动重连会卡在密码输入：
#   ssh-keygen -t ed25519 -N ""                         # 没有密钥时执行
#   ssh-copy-id root@100.125.32.77                      # 把机器人公钥放到服务器
#   ssh root@100.125.32.77 'echo ok'                     # 应直接返回 ok
set -euo pipefail

REMOTE_USER="root"
# 两台设备属于同一 Tailnet；使用服务器固定 Tailscale IP，避免 DNS 配置变化影响启动。
REMOTE_HOST="100.125.32.77"
REMOTE_PORT="22"
LOCAL_PORT="8000"
REMOTE_TARGET="localhost:8000"
# 独立的 Agent 决策通道；视觉和旧流程继续使用 8000。
AGENT_LOCAL_PORT="8001"
AGENT_REMOTE_TARGET="localhost:8001"
IDENTITY_FILE="$HOME/.ssh/id_ed25519"

PID_FILE="$HOME/.minicpm-tunnel.pid"
LOG_FILE="$HOME/.minicpm-tunnel.log"

SSH_OPTS=(
  -N
  -T
  -L "127.0.0.1:${LOCAL_PORT}:${REMOTE_TARGET}"
  -L "127.0.0.1:${AGENT_LOCAL_PORT}:${AGENT_REMOTE_TARGET}"
  -p "${REMOTE_PORT}"
  -i "${IDENTITY_FILE}"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
  -o ExitOnForwardFailure=yes
  -o StrictHostKeyChecking=accept-new
)

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

# 内部模式：SSH 断开后等待 3 秒再连接。
if [[ "${1:-}" == "__loop__" ]]; then
  while true; do
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" || true
    echo "[$(date '+%F %T')] SSH 隧道断开，3 秒后重连……" >&2
    sleep 3
  done
fi

if [[ "${1:-}" == "foreground" ]]; then
  exec ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}"
fi

start() {
  if is_running; then
    echo "隧道已在运行（PID $(cat "${PID_FILE}")）"
    return 0
  fi

  nohup "$0" __loop__ > "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  sleep 1

  if ! is_running; then
    echo "隧道启动失败，请查看 ${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
  fi

  echo "隧道已启动（PID $(cat "${PID_FILE}")）"
  echo "本地服务：http://127.0.0.1:${LOCAL_PORT}"
  echo "Agent 决策：http://127.0.0.1:${AGENT_LOCAL_PORT}"
  echo "日志：${LOG_FILE}"
}

stop() {
  if [[ -f "${PID_FILE}" ]]; then
    kill "$(cat "${PID_FILE}")" 2>/dev/null || true
  fi
  # 只清理与本脚本转发规则完全匹配的 SSH 子进程。
  pkill -f "ssh .*-L 127.0.0.1:${LOCAL_PORT}:${REMOTE_TARGET}.*-p ${REMOTE_PORT}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  echo "隧道已停止"
}

status() {
  if is_running; then
    echo "隧道运行中（PID $(cat "${PID_FILE}")）"
  else
    echo "隧道未运行"
    return 1
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "用法：$0 {start|stop|restart|status|foreground}"; exit 1 ;;
esac
