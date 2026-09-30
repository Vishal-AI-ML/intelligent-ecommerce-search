from types import SimpleNamespace

import pytest

from ecommerce_search.api.dependencies import get_db_session


class TrackingSession:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _request(session: TrackingSession):
    state = SimpleNamespace(session_factory=lambda: session)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_get_db_session_closes_session_after_normal_completion():
    session = TrackingSession()
    dependency = get_db_session(_request(session))
    assert next(dependency) is session
    assert session.close_calls == 0
    with pytest.raises(StopIteration):
        next(dependency)
    assert session.close_calls == 1


def test_get_db_session_closes_session_after_exception():
    session = TrackingSession()
    dependency = get_db_session(_request(session))
    assert next(dependency) is session
    with pytest.raises(RuntimeError, match="boom"):
        dependency.throw(RuntimeError("boom"))
    assert session.close_calls == 1
