import socket

import pytest


def _guard(*args, **kwargs):
    raise RuntimeError("network access is not allowed in search unit tests")


@pytest.fixture(autouse=True)
def no_name_resolution_or_outbound_connections(monkeypatch):
    """Search unit tests never resolve hosts or open outbound connections.

    `socket.socket.connect` stays usable here because the Windows asyncio loop behind
    TestClient connects a local self-pipe; see `no_socket_connect` for the strict variant."""
    monkeypatch.setattr(socket, "create_connection", _guard)
    monkeypatch.setattr(socket, "getaddrinfo", _guard)


@pytest.fixture
def no_socket_connect(monkeypatch):
    """Strict guard for pure (non-API) tests: no socket may connect at all."""
    monkeypatch.setattr(socket.socket, "connect", _guard)
