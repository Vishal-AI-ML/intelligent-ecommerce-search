from fastapi.testclient import TestClient

from ecommerce_search.api.app import create_app


def test_factory_registers_health_route(make_settings):
    app = create_app(make_settings())
    assert "/health" in app.openapi()["paths"]


def test_lifespan_creates_and_disposes_engine(make_settings):
    app = create_app(make_settings())
    with TestClient(app):
        engine = app.state.engine
        assert app.state.session_factory is not None
        disposed = []
        original = engine.dispose
        engine.dispose = lambda *a, **k: (disposed.append(True), original(*a, **k))[1]
    assert disposed == [True]


def test_engine_is_lazy_and_does_not_connect_at_startup(make_settings):
    # Port 1 is unreachable; startup must still succeed (health reports the failure).
    app = create_app(make_settings(postgres_port=1))
    with TestClient(app):
        pass
