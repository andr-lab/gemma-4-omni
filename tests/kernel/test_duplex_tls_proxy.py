import asyncio

from duplex_kernel.duplex_tls_proxy import start_proxy_server


def test_tls_proxy_forwards_http_upgrade_and_websocket_bytes_bidirectionally():
    async def exercise():
        async def upstream(reader, writer):
            request = await reader.readuntil(b"\r\n\r\n")
            assert request.startswith(b"GET /v1/realtime?duplex=1 HTTP/1.1\r\n")
            writer.write(
                b"HTTP/1.1 101 Switching Protocols\r\n"
                b"Connection: Upgrade\r\nUpgrade: websocket\r\n\r\n"
            )
            await writer.drain()
            frame = await reader.readexactly(6)
            writer.write(b"\x81\x04pong" + frame)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream_server = await asyncio.start_server(upstream, "127.0.0.1", 0)
        upstream_port = upstream_server.sockets[0].getsockname()[1]
        proxy_server = await start_proxy_server(
            listen_host="127.0.0.1",
            listen_port=0,
            upstream_host="127.0.0.1",
            upstream_port=upstream_port,
            ssl_context=None,
        )
        proxy_port = proxy_server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
            writer.write(
                b"GET /v1/realtime?duplex=1 HTTP/1.1\r\n"
                b"Host: localhost\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\n"
            )
            await writer.drain()
            response = await reader.readuntil(b"\r\n\r\n")
            assert response.startswith(b"HTTP/1.1 101 Switching Protocols\r\n")

            client_frame = b"\x82\x04ping"
            writer.write(client_frame)
            await writer.drain()
            reply = await reader.readexactly(12)
            assert reply == b"\x81\x04pong" + client_frame
            writer.close()
            await writer.wait_closed()
        finally:
            proxy_server.close()
            upstream_server.close()
            await proxy_server.wait_closed()
            await upstream_server.wait_closed()

    asyncio.run(exercise())
