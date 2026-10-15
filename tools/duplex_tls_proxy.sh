#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-status}"
case "${ACTION}" in
    start|stop|status) ;;
    *) echo "Usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/.tmp/duplex_tls_proxy"
CERT_DIR="${DUPLEX_TLS_CERT_DIR:-/home/administrator/.local/share/gemma4-omni/tls}"
CERT_FILE="${CERT_DIR}/server.crt"
KEY_FILE="${CERT_DIR}/server.key"
PID_FILE="${RUNTIME_DIR}/proxy.pid"
LOG_FILE="${RUNTIME_DIR}/proxy.log"
LISTEN_HOST="${DUPLEX_TLS_LISTEN_HOST:-0.0.0.0}"
LISTEN_PORT="${DUPLEX_TLS_LISTEN_PORT:-8443}"
UPSTREAM_HOST="127.0.0.1"
UPSTREAM_PORT="8000"
HELPER="${PROJECT_DIR}/duplex_kernel/duplex_tls_proxy.py"

die() {
    echo "[duplex-tls] $*" >&2
    exit 1
}

proxy_pid() {
    [[ -s "${PID_FILE}" ]] || return 1
    local candidate
    candidate="$(cat "${PID_FILE}")"
    [[ "${candidate}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${candidate}" 2>/dev/null || return 1
    ps -p "${candidate}" -o args= | grep -Fq "${HELPER}" || return 1
    printf '%s\n' "${candidate}"
}

case "${ACTION}" in
    status)
        if current_pid="$(proxy_pid)"; then
            echo "TLS proxy attivo: https://192.168.0.186:${LISTEN_PORT}/ (PID ${current_pid})"
        else
            echo "TLS proxy non attivo sulla porta ${LISTEN_PORT}."
            exit 1
        fi
        ;;
    stop)
        if current_pid="$(proxy_pid)"; then
            kill "${current_pid}"
            for _ in $(seq 1 30); do
                kill -0 "${current_pid}" 2>/dev/null || break
                sleep 0.1
            done
            if kill -0 "${current_pid}" 2>/dev/null; then
                kill -TERM "${current_pid}"
            fi
            echo "TLS proxy fermato (PID ${current_pid})."
        else
            echo "Nessun TLS proxy gestito da questo launcher risulta attivo."
        fi
        if [[ -e "${PID_FILE}" ]]; then unlink "${PID_FILE}"; fi
        ;;
    start)
        [[ "$(id -u)" == "1000" ]] || die "Avviare il proxy come UID 1000."
        [[ -f "${HELPER}" ]] || die "Helper mancante: ${HELPER}"
        if current_pid="$(proxy_pid)"; then
            echo "TLS proxy già attivo: https://192.168.0.186:${LISTEN_PORT}/ (PID ${current_pid})"
            exit 0
        fi
        command -v openssl >/dev/null || die "openssl non installato."
        command -v python3 >/dev/null || die "python3 non installato."
        mkdir -p "${RUNTIME_DIR}" "${CERT_DIR}"
        chmod 700 "${RUNTIME_DIR}" "${CERT_DIR}"
        if [[ ! -s "${CERT_FILE}" || ! -s "${KEY_FILE}" ]]; then
            umask 077
            openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 365 \
                -keyout "${KEY_FILE}" -out "${CERT_FILE}" \
                -subj "/CN=192.168.0.186" \
                -addext "subjectAltName=IP:192.168.0.186" \
                -addext "basicConstraints=critical,CA:FALSE" \
                -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
                -addext "extendedKeyUsage=serverAuth"
        fi
        chmod 600 "${KEY_FILE}"
        chmod 644 "${CERT_FILE}"
        openssl x509 -in "${CERT_FILE}" -noout -checkip 192.168.0.186 >/dev/null \
            || die "Il certificato esistente non include 192.168.0.186 come SAN; non lo sovrascrivo."

        nohup python3 -u "${HELPER}" \
            --listen-host "${LISTEN_HOST}" --listen-port "${LISTEN_PORT}" \
            --upstream-host "${UPSTREAM_HOST}" --upstream-port "${UPSTREAM_PORT}" \
            --certfile "${CERT_FILE}" --keyfile "${KEY_FILE}" \
            >>"${LOG_FILE}" 2>&1 </dev/null &
        started_pid=$!
        printf '%s\n' "${started_pid}" >"${PID_FILE}"
        ready=0
        for _ in $(seq 1 60); do
            if ! kill -0 "${started_pid}" 2>/dev/null; then break; fi
            if curl -kfsS --max-time 2 "https://127.0.0.1:${LISTEN_PORT}/health" >/dev/null; then
                ready=1
                break
            fi
            sleep 0.25
        done
        if [[ "${ready}" != 1 ]]; then
            echo "TLS proxy non pronto; ultimi log:" >&2
            tail -n 30 "${LOG_FILE}" >&2 || true
            die "Avvio TLS proxy fallito sulla porta ${LISTEN_PORT}."
        fi
        fingerprint="$(openssl x509 -in "${CERT_FILE}" -noout -fingerprint -sha256 | cut -d= -f2)"
        echo "TLS proxy pronto: https://192.168.0.186:${LISTEN_PORT}/ (PID ${started_pid})"
        echo "Certificato self-signed SHA256: ${fingerprint}"
        ;;
esac
