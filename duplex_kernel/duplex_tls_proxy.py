"""Small TLS terminator that forwards HTTP and WebSocket bytes to the local API."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import ssl
from asyncio import StreamReader, StreamWriter
from pathlib import Path

LOG = logging.getLogger("duplex_tls_proxy")
COPY_CHUNK_BYTES = 64 * 1024


async def _copy_stream(reader: StreamReader, writer: StreamWriter) -> None:
    while chunk := await reader.read(COPY_CHUNK_BYTES):
        writer.write(chunk)
        await writer.drain()


async def _close_writer(writer: StreamWriter | None) -> None:
    if writer is None:
        return
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()


async def _proxy_connection(
    client_reader: StreamReader,
    client_writer: StreamWriter,
    *,
    upstream_host: str,
    upstream_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    upstream_writer: StreamWriter | None = None
    tasks: list[asyncio.Task[None]] = []
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(
            upstream_host, upstream_port
        )
        tasks = [
            asyncio.create_task(_copy_stream(client_reader, upstream_writer)),
            asyncio.create_task(_copy_stream(upstream_reader, client_writer)),
        ]
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    except (ConnectionError, OSError, ssl.SSLError) as exc:
        LOG.info("connection %r closed: %s", peer, type(exc).__name__)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await _close_writer(upstream_writer)
        await _close_writer(client_writer)


async def start_proxy_server(
    *,
    listen_host: str,
    listen_port: int,
    upstream_host: str = "127.0.0.1",
    upstream_port: int = 8000,
    ssl_context: ssl.SSLContext | None,
) -> asyncio.AbstractServer:
    async def handle(reader: StreamReader, writer: StreamWriter) -> None:
        await _proxy_connection(
            reader,
            writer,
            upstream_host=upstream_host,
            upstream_port=upstream_port,
        )

    return await asyncio.start_server(
        handle,
        host=listen_host,
        port=listen_port,
        ssl=ssl_context,
        limit=COPY_CHUNK_BYTES,
    )


def make_tls_server_context(certfile: str | Path, keyfile: str | Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
    return context


async def run_proxy(args: argparse.Namespace) -> None:
    context = make_tls_server_context(args.certfile, args.keyfile)
    server = await start_proxy_server(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_host=args.upstream_host,
        upstream_port=args.upstream_port,
        ssl_context=context,
    )
    sockets = server.sockets or []
    addresses = ", ".join(str(sock.getsockname()) for sock in sockets)
    LOG.info("TLS proxy listening on %s and forwarding to %s:%d", addresses, args.upstream_host, args.upstream_port)
    async with server:
        await server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8443)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=8000)
    parser.add_argument("--certfile", required=True)
    parser.add_argument("--keyfile", required=True)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run_proxy(build_parser().parse_args()))


if __name__ == "__main__":
    main()
