"""Real loopback HTTP checks; all domain calls use a non-network fake backend."""
from __future__ import annotations

from contextlib import contextmanager
import base64
import hashlib
import http.client as http_client
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

import pytest

from meta_data_extraction.agent import pilot_http as http

TOKEN = "test-only-AbCdEfGhIjKlMnOpQrStUvWx0123456789"
_SOCKET_CONNECT = socket.socket.connect
_ACTIVE_PORTS = set()


@pytest.fixture(autouse=True)
def ephemeral_loopback_only(no_network, monkeypatch):
    """Keep DNS/egress forbidden; permit only the server created by this test."""
    def connect(sock, address):
        if address[0] != "127.0.0.1" or address[1] not in _ACTIVE_PORTS:
            raise AssertionError("Only this test's ephemeral loopback server is permitted")
        return _SOCKET_CONNECT(sock, address)

    def create(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, **kwargs):
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            connection.settimeout(timeout)
        try:
            connection.connect(address)
        except BaseException:
            connection.close()
            raise
        return connection

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", create)


class FakeService:
    def __init__(self):
        self.calls = []
        self.runs = {}
        self.failure = None

    def _call(self, name, *values):
        self.calls.append((name, *values))
        if self.failure:
            raise self.failure
        return {"operation": name, "new_live_requests": 0, "values": list(values)}

    def health(self):
        return self._call("health")

    def start(self, run_id, *, max_steps=None):
        self.runs[run_id] = {"run_id": run_id, "saved": True}
        return self._call("start", run_id, max_steps)

    def status(self, run_id):
        self._call("status", run_id)
        return self.runs.get(run_id, {"run_id": run_id})

    def review(self, run_id):
        return self._call("review", run_id)

    def submit(self, run_id, response):
        return self._call("submit", run_id, response)

    def export(self, run_id, expected_state_id):
        return self._call("export", run_id, expected_state_id)


@contextmanager
def serving(service=None, log=None):
    backend = service or FakeService()
    log = log if log is not None else io.StringIO()
    server = http.make_server(backend, TOKEN, port=0, log_stream=log)
    _ACTIVE_PORTS.add(server.server_address[1])
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield server, backend, log
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()
        _ACTIVE_PORTS.remove(server.server_address[1])
        assert not worker.is_alive()


def request(server, path="/health", method="GET", body=None, headers=None, auth=True):
    connection = http_client.HTTPConnection(*server.server_address, timeout=3)
    supplied = {"Authorization": "Bearer " + TOKEN} if auth else {}
    if body is not None:
        supplied["Content-Type"] = "application/json"
        if not isinstance(body, (bytes, str)):
            body = json.dumps(body)
    supplied.update(headers or {})
    connection.request(method, path, body=body, headers=supplied)
    result = connection.getresponse()
    status, response_headers, raw = result.status, dict(result.getheaders()), result.read()
    connection.close()
    return status, response_headers, raw


def raw_request(server, wire):
    with socket.create_connection(server.server_address, timeout=3) as connection:
        connection.sendall(wire)
        connection.shutdown(socket.SHUT_WR)
        result = b""
        while chunk := connection.recv(65536):
            result += chunk
        return result


def test_authenticated_loopback_full_route_contract():
    with serving() as (server, backend, _):
        assert request(server)[0] == 200
        assert request(server, "/runs", "POST", {"run_id": "m10-example", "max_steps": 1})[0] == 200
        assert request(server, "/runs/m10-example")[0] == 200
        assert request(server, "/runs/m10-example/review")[0] == 200
        response = {"outcome": "unresolved", "rationale": "insufficient evidence"}
        assert request(server, "/runs/m10-example/review", "POST", response)[0] == 200
        assert request(server, "/runs/m10-example/export", "POST", {"expected_state_id": "review-state:example"})[0] == 200
        assert backend.calls == [("health",), ("start", "m10-example", 1), ("status", "m10-example"),
                                 ("review", "m10-example"), ("submit", "m10-example", response),
                                 ("export", "m10-example", "review-state:example")]


def test_transport_restart_keeps_backend_state_and_requires_fresh_auth():
    backend = FakeService()
    with serving(backend) as (server, _, _):
        assert request(server, "/runs", "POST", {"run_id": "m10-restart"})[0] == 200
    with serving(backend) as (server, _, _):
        assert request(server, "/runs/m10-restart", auth=False)[0] == 401
        assert json.loads(request(server, "/runs/m10-restart")[2]) == {"run_id": "m10-restart", "saved": True}


@pytest.mark.parametrize("path,method,body", [
    ("/health", "GET", None), ("/runs", "POST", {"run_id": "m10-test"}),
    ("/runs/m10-test", "GET", None), ("/runs/m10-test/review", "GET", None),
    ("/runs/m10-test/review", "POST", {}), ("/runs/m10-test/export", "POST", {})])
def test_every_api_route_requires_bearer(path, method, body):
    with serving() as (server, backend, _):
        status, headers, raw = request(server, path, method, body, auth=False)
        assert status == 401 and json.loads(raw) == {"error": "authentication_required"}
        assert headers["WWW-Authenticate"] == "Bearer"
        assert not backend.calls


@pytest.mark.parametrize("authorization", ["", TOKEN, "Basic " + TOKEN, "bearer " + TOKEN, "Bearer wrong-token"])
def test_invalid_authentication_never_dispatches(authorization):
    with serving() as (server, backend, _):
        assert request(server, headers={"Authorization": authorization})[0] == 401
        assert not backend.calls


@pytest.mark.parametrize("host", ["evil.example", "localhost:8765", "127.0.0.2:8765", "127.0.0.1", "127.0.0.1:0", "[::1]:8765"])
def test_rebinding_hosts_rejected_even_for_static_page(host):
    with serving() as (server, backend, _):
        assert request(server, "/", headers={"Host": host})[0] == 403
        assert not backend.calls


@pytest.mark.parametrize("origin", ["null", "https://evil.example", "http://localhost:8765", "http://127.0.0.1:0", "https://127.0.0.1:8765"])
def test_cross_origin_rejected(origin):
    with serving() as (server, backend, _):
        assert request(server, headers={"Origin": origin})[0] == 403
        assert not backend.calls


def test_exact_same_origin_accepted_and_security_headers_present():
    with serving() as (server, _, _):
        status, headers, _ = request(server, headers={"Origin": f"http://127.0.0.1:{server.server_address[1]}"})
        assert status == 200
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
        assert not any(name.startswith("Access-Control-") for name in headers)


def test_public_ui_has_exact_script_style_csp_and_no_unsafe_dom_or_storage():
    with serving() as (server, backend, _):
        status, headers, raw = request(server, "/", auth=False)
        assert status == 200 and not backend.calls
        assert headers["Content-Type"].startswith("text/html")
        policy = headers["Content-Security-Policy"]
        assert "unsafe-inline" not in policy and "frame-ancestors 'none'" in policy
        for tag in (b"script", b"style"):
            matches = re.findall(b"<" + tag + b">(.*?)</" + tag + b">", raw, flags=re.DOTALL)
            assert len(matches) == 1
            digest = base64.b64encode(hashlib.sha256(matches[0]).digest()).decode()
            assert "'sha256-" + digest + "'" in policy
        for forbidden in (b"innerHTML", b"localStorage", b"sessionStorage", b"document.cookie", b"<script src=", b"onclick="):
            assert forbidden not in raw
        assert b'input id="token" type="password"' in raw
        assert b'outcome:""' in raw and b"attestations:[]" in raw
        assert b'input id="confirm-response" type="checkbox"' in raw


@pytest.mark.parametrize("payload,code", [
    (b'{"run_id":"a","run_id":"b"}', "duplicate_json_key"),
    (b'{"value":NaN}', "nonfinite_json"), (b'{"value":1e999}', "nonfinite_json"),
    (b'{"value":Infinity}', "nonfinite_json"), (b'[]', "json_object_required"),
    (b'{', "invalid_json"), (b'\xff', "invalid_json"),
    (b'{"a":' * 33 + b'0' + b'}' * 33, "json_too_deep")])
def test_strict_request_json_rejects_before_dispatch(payload, code):
    with serving() as (server, backend, _):
        status, _, raw = request(server, "/runs", "POST", payload)
        assert status == 400 and json.loads(raw)["error"] == code
        assert not backend.calls


@pytest.mark.parametrize("body", [
    {}, {"run_id": 5}, {"run_id": "../data/gold"}, {"run_id": "m10-a", "source": "data/gold/a.json"},
    {"run_id": "m10-a", "model": "live"}, {"run_id": "m10-a", "max_steps": True},
    {"run_id": "m10-a", "max_steps": 0}, {"run_id": "m10-a", "max_steps": 13},
    {"run_id": "m10-a", "max_steps": 1.2}, {"run_id": "m10-a", "max_steps": "1"}])
def test_start_shape_and_bounds(body):
    with serving() as (server, backend, _):
        assert request(server, "/runs", "POST", body)[0] == 400
        assert not backend.calls


@pytest.mark.parametrize("body", [{}, {"expected_state_id": 4}, {"expected_state_id": "x", "output": "outputs/x"}])
def test_export_has_no_output_path_override(body):
    with serving() as (server, backend, _):
        assert request(server, "/runs/m10-example/export", "POST", body)[0] == 400
        assert not backend.calls


@pytest.mark.parametrize("path", ["/.env", "/data/gold/x", "/runs/../health", "/health?token=SECRET", "/%2e%2e/.env", "/runs\\test", "http://evil.example/health"])
def test_unknown_or_ambiguous_paths_never_dispatch(path):
    with serving() as (server, backend, log):
        assert request(server, path)[0] in {400, 403, 404}
        assert not backend.calls
        assert path not in log.getvalue() and "SECRET" not in log.getvalue()


@pytest.mark.parametrize("extra,expected", [
    ("Content-Length: 2\r\nContent-Length: 2\r\n", 400),
    ("Transfer-Encoding: chunked\r\n", 400),
    ("Authorization: Bearer other\r\n", 400),
    ("Content-Type: application/json\r\nContent-Type: text/plain\r\n", 400),
    ("Expect: 100-continue\r\n", 417),
    ("Content-Length: -1\r\nContent-Type: application/json\r\n", 400),
    ("Content-Length: 999999999\r\nContent-Type: application/json\r\n", 413),
    ("Content-Type: application/json\r\n", 411),
    ("Content-Length: 2\r\nContent-Type: text/plain\r\n", 415),
])
def test_raw_framing_and_content_bounds(extra, expected):
    with serving() as (server, backend, _):
        wire = f"POST /runs HTTP/1.1\r\nHost: 127.0.0.1:{server.server_address[1]}\r\nAuthorization: Bearer {TOKEN}\r\n{extra}\r\n{{}}".encode()
        assert raw_request(server, wire).startswith(f"HTTP/1.1 {expected} ".encode())
        assert not backend.calls


@pytest.mark.parametrize("header", ["Host", "Origin"])
def test_duplicate_host_and_origin_are_rejected(header):
    with serving() as (server, backend, _):
        host = f"127.0.0.1:{server.server_address[1]}"
        value = host if header == "Host" else "http://" + host
        wire = f"GET /health HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {TOKEN}\r\n{header}: {value}\r\n{header}: {value}\r\n\r\n".encode()
        assert raw_request(server, wire).startswith(b"HTTP/1.1 403 ")
        assert not backend.calls


def test_header_total_and_request_line_bounds_are_sanitized():
    secret = "PRIVATE-SENTINEL-DO-NOT-LOG"
    with serving() as (server, backend, log):
        host = f"127.0.0.1:{server.server_address[1]}"
        header = f"GET /health HTTP/1.1\r\nHost: {host}\r\nX-Private: " + secret + "a" * http.MAX_HEADER_BYTES + "\r\n\r\n"
        assert raw_request(server, header.encode()).startswith(b"HTTP/1.1 431 ")
        line = "GET /" + secret + "a" * 5000 + f" HTTP/1.1\r\nHost: {host}\r\n\r\n"
        assert raw_request(server, line.encode()).startswith(b"HTTP/1.1 414 ")
        assert secret not in log.getvalue() and not backend.calls


def test_body_limit_and_incomplete_body():
    with serving() as (server, backend, _):
        assert request(server, "/runs", "POST", b" " * (http.MAX_BODY_BYTES + 1))[0] == 413
        wire = f"POST /runs HTTP/1.1\r\nHost: 127.0.0.1:{server.server_address[1]}\r\nAuthorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\nContent-Length: 20\r\n\r\n{{}}".encode()
        assert b'"body_incomplete"' in raw_request(server, wire)
        assert not backend.calls


def test_backend_failures_and_request_logs_never_contain_secrets_or_paths():
    sentinel = "LOCAL-PRIVATE-KEY-SENTINEL"
    with serving() as (server, backend, log):
        backend.failure = ValueError("D:/private/.env=" + sentinel)
        status, _, raw = request(server, "/runs/m10-private/review", "POST", {"rationale": sentinel})
        assert status == 500 and json.loads(raw) == {"error": "internal_error"}
        text = log.getvalue() + raw.decode()
        assert sentinel not in text and TOKEN not in text and "m10-private" not in text and ".env" not in text
        event = json.loads(log.getvalue())
        assert event == {"event": "http_request", "method": "POST", "route": "submit", "status": 500}


def test_safe_domain_error_status_preserved_and_untrusted_code_removed():
    with serving() as (server, backend, _):
        backend.failure = http.RequestError("stale_state", 409)
        status, _, raw = request(server)
        assert status == 409 and json.loads(raw)["error"] == "stale_state"
        backend.failure = http.RequestError("storage_limit", 507)
        status, _, raw = request(server)
        assert status == 507 and json.loads(raw)["error"] == "storage_limit"
        backend.failure = http.RequestError("secret:abcdef", 999)
        status, _, raw = request(server)
        assert status == 500 and json.loads(raw)["error"] == "internal_error"


def test_response_bound_and_nonfinite_backend_response():
    with serving() as (server, backend, _):
        backend.health = lambda: {"large": "x" * http.MAX_RESPONSE_BYTES}
        status, _, raw = request(server)
        assert status == 500 and json.loads(raw)["error"] == "response_too_large"
        backend.health = lambda: {"value": float("nan")}
        status, _, raw = request(server)
        assert status == 500 and json.loads(raw)["error"] == "response_invalid"


def test_get_body_and_unsupported_methods_are_rejected():
    with serving() as (server, backend, log):
        assert request(server, method="GET", body="{}")[0] == 400
        status, _, raw = request(server, "/private-value", method="DELETE")
        assert status == 501 and json.loads(raw)["error"] == "http_request_rejected"
        assert "private-value" not in log.getvalue() and not backend.calls


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "::1", "192.168.1.1", "127.0.0.2"])
def test_only_explicit_ipv4_loopback_bind_is_supported(host):
    with pytest.raises(ValueError, match="loopback_bind_required"):
        http.make_server(FakeService(), TOKEN, host=host, port=0)


@pytest.mark.parametrize("token", [None, "", "short", "x" * 32, "a" * 300, "s p a c e " * 8, "é" * 32])
def test_token_policy_redacts_rejected_values(token):
    with pytest.raises(ValueError) as error:
        http.validate_token(token)
    assert str(error.value) == "pilot_token_invalid"


def test_cli_missing_token_fails_closed_without_loading_service(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("META_AGENT_PILOT_TOKEN", raising=False)
    assert http.main(["serve", "--root", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == '{"error":"pilot_start_failed"}\n'


def test_body_timeout_does_not_hold_the_single_user_server(monkeypatch):
    monkeypatch.setattr(http, "REQUEST_TIMEOUT_SECONDS", 0.1)
    with serving() as (server, backend, _):
        with socket.create_connection(server.server_address, timeout=3) as connection:
            wire = f"POST /runs HTTP/1.1\r\nHost: 127.0.0.1:{server.server_address[1]}\r\nAuthorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\nContent-Length: 20\r\n\r\n{{}}".encode()
            connection.sendall(wire)
            response = connection.recv(65536)
            assert response.startswith(b"HTTP/1.1 408 ")
        assert not backend.calls
        assert request(server)[0] == 200


def test_test_network_guard_still_denies_egress_and_unowned_loopback():
    for address in (("example.com", 443), ("127.0.0.1", 1)):
        with pytest.raises(AssertionError, match="ephemeral loopback"):
            socket.create_connection(address)
    with pytest.raises(AssertionError, match="Network access is forbidden"):
        socket.getaddrinfo("example.com", 443)


@pytest.fixture(scope="module")
def cli_bundle(tmp_path_factory):
    from meta_data_extraction.agent.pilot_bundle import BUNDLE_FILES, build_bundle
    repository = Path(__file__).resolve().parents[2]
    source = tmp_path_factory.mktemp("ph") / "s"
    for name in BUNDLE_FILES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, target)
    return Path(build_bundle(source, "m10-http-bundle")["runtime_root"])


def test_production_cli_refuses_checkout_code_against_an_isolated_bundle(cli_bundle, monkeypatch, capsys):
    monkeypatch.setenv("META_AGENT_PILOT_TOKEN", TOKEN)
    def forbidden(*args, **kwargs):
        raise AssertionError("Wrong-origin code must not start a listener")
    monkeypatch.setattr(http, "make_server", forbidden)
    assert http.main(["serve", "--root", str(cli_bundle), "--port", "0"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == '{"error":"pilot_start_failed"}\n'


@pytest.mark.parametrize("contaminated", [False, True])
def test_relocated_cli_origin_check_with_egress_denied(cli_bundle, contaminated):
    # Real relocated imports, real bundle/service verification; replace only the
    # indefinite listener loop. Root integration exercises actual subprocess TCP.
    script = '''
import os, socket, sys, types
def denied(*args, **kwargs): raise AssertionError("Network forbidden")
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
socket.getaddrinfo = denied
from meta_data_extraction.agent import pilot_http
class Listener:
    server_address = ("127.0.0.1", 0)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def serve_forever(self, **kwargs): pass
pilot_http.make_server = lambda *args, **kwargs: Listener()
if sys.argv[2] == "True":
    foreign = types.ModuleType("meta_data_extraction.foreign")
    foreign.__file__ = sys.executable
    sys.modules[foreign.__name__] = foreign
raise SystemExit(pilot_http.main(["serve", "--root", sys.argv[1], "--port", "0"]))
'''
    env = os.environ.copy()
    env.update(PYTHONPATH=str(cli_bundle / "src"), PYTHONDONTWRITEBYTECODE="1", META_AGENT_PILOT_TOKEN=TOKEN)
    completed = subprocess.run([sys.executable, "-B", "-c", script, str(cli_bundle), str(contaminated)],
                               cwd=cli_bundle, env=env, capture_output=True, text=True, timeout=30)
    if contaminated:
        assert completed.returncode == 2 and completed.stdout == ""
        assert completed.stderr == '{"error":"pilot_start_failed"}\n'
    else:
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == {"event": "pilot_listening", "host": "127.0.0.1", "port": 0,
                                                "mode": "synthetic_offline", "live_requests": 0}
        assert completed.stderr == ""
    assert TOKEN not in completed.stdout + completed.stderr


@pytest.mark.parametrize("authenticated,body,status", [
    (False, b'{"run_id":"m10-rejected"}', 401),
    (True, b" " * (http.MAX_BODY_BYTES + 1), 413),
], ids=["unauthorized", "oversized"])
def test_early_rejection_allows_a_delayed_inflight_body_without_tcp_reset(authenticated, body, status):
    with serving() as (server, backend, _):
        connection = http_client.HTTPConnection(*server.server_address, timeout=3)
        connection.putrequest("POST", "/runs")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(len(body)))
        if authenticated:
            connection.putheader("Authorization", "Bearer " + TOKEN)
        connection.endheaders()
        # The server rejects on headers before this body arrives. A full socket
        # close used to reset the connection and discard its rejection response.
        time.sleep(0.04)
        connection.send(body)
        response = connection.getresponse()
        assert response.status == status
        assert json.loads(response.read())["error"] in {"authentication_required", "body_too_large"}
        connection.close()
        assert not backend.calls


def test_completed_response_cannot_race_its_structured_log():
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()

    class BlockingLog(io.StringIO):
        def write(self, value):
            entered.set()
            assert release.wait(2)
            return super().write(value)

    with serving(log=BlockingLog()) as (server, backend, log):
        backend.failure = ValueError("private input must remain redacted")
        responses = []

        def client():
            responses.append(request(server))
            completed.set()

        worker = threading.Thread(target=client, daemon=True)
        worker.start()
        try:
            assert entered.wait(1)
            assert not completed.wait(0.05)
        finally:
            release.set()
            worker.join(timeout=3)
        assert not worker.is_alive() and responses[0][0] == 500
        assert json.loads(log.getvalue()) == {"event": "http_request", "method": "GET", "route": "health", "status": 500}


def test_rejected_body_drain_has_a_total_deadline_and_recovers(monkeypatch):
    monkeypatch.setattr(http, "REJECT_DRAIN_SECONDS", 0.06)
    with serving() as (server, backend, _):
        with socket.create_connection(server.server_address, timeout=3) as connection:
            connection.sendall(f"POST /runs HTTP/1.1\r\nHost: 127.0.0.1:{server.server_address[1]}\r\nContent-Type: application/json\r\nContent-Length: 900000000\r\n\r\n".encode())
            assert connection.recv(65536).startswith(b"HTTP/1.1 401 ")
            # Keep the rejected sender open and silent. The server must still
            # accept the next legitimate request without its five-second timeout.
            started = time.monotonic()
            assert request(server)[0] == 200
            assert time.monotonic() - started < 1
        assert backend.calls == [("health",)]


def test_repeated_early_rejections_remain_observable_without_dispatch():
    with serving() as (server, backend, log):
        for _ in range(15):
            assert request(server, "/runs", "POST", {"run_id": "m10-rejected"}, auth=False)[0] == 401
            assert request(server, "/runs", "POST", b" " * (http.MAX_BODY_BYTES + 1))[0] == 413
        assert not backend.calls
        events = [json.loads(line) for line in log.getvalue().splitlines()]
        assert len(events) == 30 and [item["status"] for item in events] == [401, 413] * 15
