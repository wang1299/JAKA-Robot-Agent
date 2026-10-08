#!/usr/bin/env bash
# Two loopback-only forwards. Host/user/key belong in a private SSH alias.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${JAKA_TUNNEL_CONFIG:-$SCRIPT_DIR/configs/tunnel.local.env}"
if [[ -f "$CONFIG" ]]; then
  set -a
  # This is a trusted local shell config, never a downloaded file.
  source "$CONFIG"
  set +a
elif [[ -n "${JAKA_TUNNEL_CONFIG:-}" ]]; then
  echo "Tunnel config file is missing" >&2
  exit 1
fi
HOST="${JAKA_SSH_HOST:-jaka-model-server}"
DEST="$HOST"
[[ -z "${JAKA_SSH_USER:-}" ]] || DEST="${JAKA_SSH_USER}@${HOST}"
VISION_LOCAL="${JAKA_VISION_LOCAL_PORT:-8000}"
VISION_REMOTE="${JAKA_VISION_REMOTE_PORT:-8000}"
AGENT_LOCAL="${JAKA_AGENT_LOCAL_PORT:-8001}"
AGENT_REMOTE="${JAKA_AGENT_REMOTE_PORT:-8001}"
for port in "$VISION_LOCAL" "$VISION_REMOTE" "$AGENT_LOCAL" "$AGENT_REMOTE"; do
  if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo "Invalid tunnel port" >&2; exit 1
  fi
done
[[ "$VISION_LOCAL" != "$AGENT_LOCAL" ]] || { echo "Local ports must differ" >&2; exit 1; }
SSH=(ssh -N -T -o BatchMode=yes -o StrictHostKeyChecking=yes
  -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3
  -o ExitOnForwardFailure=yes
  -L "127.0.0.1:${VISION_LOCAL}:127.0.0.1:${VISION_REMOTE}"
  -L "127.0.0.1:${AGENT_LOCAL}:127.0.0.1:${AGENT_REMOTE}")
[[ -z "${JAKA_SSH_PORT:-}" ]] || SSH+=(-p "$JAKA_SSH_PORT")
[[ -z "${JAKA_SSH_IDENTITY:-}" ]] || SSH+=(-i "$JAKA_SSH_IDENTITY")
SSH+=("$DEST")
PID_FILE="${JAKA_TUNNEL_PID_FILE:-$HOME/.jaka-model-tunnel.pid}"
LOG_FILE="${JAKA_TUNNEL_LOG_FILE:-$HOME/.jaka-model-tunnel.log}"
running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid command
  read -r pid < "$PID_FILE"
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$command" == *"$SCRIPT_DIR/tunnel.sh __loop__"* ]]
}
action="${1:-status}"
case "$action" in
  foreground) exec "${SSH[@]}" ;;
  __loop__)
    child=""
    trap '[[ -z "$child" ]] || kill "$child" 2>/dev/null || true; exit 0' TERM INT
    while true; do
      "${SSH[@]}" & child=$!
      wait "$child" || true
      child=""
      sleep 3 & child=$!
      wait "$child" || true
      child=""
    done ;;
  start|stop|restart|status) ;;
  *) echo "Usage: $0 {start|stop|restart|status|foreground}" >&2; exit 2 ;;
esac
# When the optional unit exists, avoid launching a second background tunnel.
if command -v systemctl >/dev/null 2>&1 &&
   systemctl cat jaka-model-tunnel.service >/dev/null 2>&1; then
  if [[ "$action" == status ]]; then exec systemctl status jaka-model-tunnel.service; fi
  exec sudo systemctl "$action" jaka-model-tunnel.service
fi
stop_tunnel() {
  if running; then
    local pid; read -r pid < "$PID_FILE"
    kill "$pid"
    rm -f -- "$PID_FILE"
    echo "Tunnel stopped"
  else echo "Tunnel is not running"; fi
}
start_tunnel() {
  if running; then echo "Tunnel is already running"; return; fi
  if command -v ss >/dev/null 2>&1; then
    for port in "$VISION_LOCAL" "$AGENT_LOCAL"; do
      if [[ -n "$(ss -H -ltn "sport = :$port")" ]]; then
        echo "A local model port is occupied; check existing services" >&2; exit 1
      fi
    done
  fi
  nohup bash "$SCRIPT_DIR/tunnel.sh" __loop__ >> "$LOG_FILE" 2>&1 &
  echo "$!" > "$PID_FILE"
  echo "Tunnel supervisor started; verify model endpoints (see docs/models.md)"
}
case "$action" in
  start) start_tunnel ;;
  stop) stop_tunnel ;;
  restart) stop_tunnel; start_tunnel ;;
  status) if running; then echo "Tunnel supervisor is running"; else echo "Tunnel is not running"; exit 1; fi ;;
esac
