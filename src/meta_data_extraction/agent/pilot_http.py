"""Authenticated, loopback-only HTTP adapter for the single-user mock pilot.

This deliberately small server is not an Internet-facing deployment target.
The backend verifies an isolated allowlisted bundle on every operation.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Protocol, TextIO

MAX_BODY_BYTES = 131_072
MAX_RESPONSE_BYTES = 2_000_000
MAX_HEADER_BYTES = 16_384
MAX_JSON_DEPTH = 32
REQUEST_TIMEOUT_SECONDS = 5
# A rejected sender may already be transmitting a bounded request. Draining a
# small fixed amount after sending the response prevents a Windows TCP reset
# from discarding the error response, without accepting or parsing its body.
REJECT_DRAIN_BYTES = MAX_BODY_BYTES + MAX_HEADER_BYTES + 4096
REJECT_DRAIN_SECONDS = 0.25
_RUN_ID = re.compile(r"^m10-[a-z0-9][a-z0-9_-]{0,59}$")
_RUN_ROUTE = re.compile(r"^/runs/(m10-[a-z0-9][a-z0-9_-]{0,59})(?:/(review|export))?$")


class Service(Protocol):
    def health(self) -> dict: ...
    def start(self, run_id: str, *, max_steps: int | None = None) -> dict: ...
    def status(self, run_id: str) -> dict: ...
    def review(self, run_id: str) -> dict: ...
    def submit(self, run_id: str, response: dict) -> dict: ...
    def export(self, run_id: str, expected_state_id: str) -> dict: ...


class RequestError(ValueError):
    def __init__(self, code: str, status_code: int = 400):
        self.code, self.status_code = code, status_code
        super().__init__(code)


def validate_token(token: str | None) -> str:
    """Require a bounded URL-safe generated secret; never include it in errors."""
    if (not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token)
            or len(set(token)) < 12):
        raise ValueError("pilot_token_invalid")
    return token


def strict_object(raw: bytes) -> dict:
    if len(raw) > MAX_BODY_BYTES:
        raise RequestError("body_too_large", 413)
    try:
        text = raw.decode("utf-8")
        depth = 0
        in_string = escaped = False
        for char in text:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
                if depth > MAX_JSON_DEPTH:
                    raise RequestError("json_too_deep")
            elif char in "]}":
                depth -= 1

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise RequestError("duplicate_json_key")
                result[key] = value
            return result

        def constant(_value):
            raise RequestError("nonfinite_json")

        def floating(value):
            result = float(value)
            if not math.isfinite(result):
                raise RequestError("nonfinite_json")
            return result

        result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)
        if not isinstance(result, dict):
            raise RequestError("json_object_required")
        return result
    except RequestError:
        raise
    except (UnicodeError, ValueError, RecursionError, OverflowError):
        raise RequestError("invalid_json") from None


class _BoundedHeaders:
    def __init__(self, stream):
        self.stream, self.remaining = stream, MAX_HEADER_BYTES

    def readline(self, limit=-1):
        line = self.stream.readline(min(limit if limit >= 0 else self.remaining + 1, self.remaining + 1))
        self.remaining -= len(line)
        if self.remaining < 0:
            raise http.client.LineTooLong("header")
        return line


class PilotHTTPServer(HTTPServer):
    """Serial requests fit the explicitly single-user SQLite backend."""
    allow_reuse_address = False
    request_queue_size = 4

    def __init__(self, address, service: Service, token: str, log_stream: TextIO | None = None):
        if address[0] != "127.0.0.1":
            raise ValueError("loopback_bind_required")
        if type(address[1]) is not int or not 0 <= address[1] <= 65535:
            raise ValueError("port_invalid")
        self.token = validate_token(token)
        self.service, self.log_stream = service, log_stream if log_stream is not None else sys.stderr
        self.ui = Path(__file__).with_name("pilot_ui.html").read_bytes()
        if len(self.ui) > MAX_RESPONSE_BYTES:
            raise ValueError("ui_too_large")
        self.csp = _ui_policy(self.ui)
        super().__init__(address, PilotHandler)

    def handle_error(self, request, client_address):
        # Socket/parser failures must never print exception contents or a traceback.
        self.log_stream.write('{"event":"http_connection_error"}\n')
        self.log_stream.flush()


def _ui_policy(ui: bytes) -> str:
    def hashes(tag):
        values = re.findall(b"<" + tag + b">(.*?)</" + tag + b">", ui, flags=re.DOTALL)
        return " ".join("'sha256-" + base64.b64encode(hashlib.sha256(value).digest()).decode("ascii") + "'" for value in values)
    return ("default-src 'none'; script-src " + hashes(b"script") + "; style-src " + hashes(b"style")
            + "; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


class PilotHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        self.request.settimeout(REQUEST_TIMEOUT_SECONDS)
        super().setup()

    def version_string(self):
        return "LocalPilot"

    def log_message(self, format, *args):
        pass

    def log_error(self, format, *args):
        pass

    def parse_request(self):
        self.route_name = "rejected"
        self._body_consumed = False
        if len(self.raw_requestline) > 4096:
            self.requestline = ""
            self.request_version = "HTTP/1.1"
            self.command = ""
            self.send_error(414)
            return False
        original = self.rfile
        self.rfile = _BoundedHeaders(original)
        try:
            parsed = super().parse_request()
        finally:
            self.rfile = original
        self.close_connection = True
        if not parsed:
            return False
        try:
            if self.request_version not in {"HTTP/1.0", "HTTP/1.1"}:
                raise RequestError("http_version_rejected", 505)
            expected_host = f"127.0.0.1:{self.server.server_address[1]}"
            if self.headers.get_all("Host", []) != [expected_host]:
                raise RequestError("host_rejected", 403)
            origins = self.headers.get_all("Origin", [])
            if origins and origins != ["http://" + expected_host]:
                raise RequestError("origin_rejected", 403)
            if any(self.headers.get_all(name, []) for name in ("Transfer-Encoding", "Expect")):
                raise RequestError("framing_rejected")
            for name in ("Content-Length", "Content-Type", "Authorization"):
                if len(self.headers.get_all(name, [])) > 1:
                    raise RequestError("duplicate_header")
            if any(char in self.path for char in ("?", "#", "%", "\\")) or not self.path.startswith("/"):
                raise RequestError("path_rejected")
            return True
        except RequestError as error:
            self._json(error.status_code, {"error": error.code})
            return False

    def handle_expect_100(self):
        self.send_error(417)
        return False

    def send_error(self, code, message=None, explain=None):
        self.close_connection = True
        self._json(code, {"error": "http_request_rejected"})

    def _write(self, status: int, raw: bytes, content_type: str):
        self.close_connection = True
        if len(raw) > MAX_RESPONSE_BYTES:
            status, raw, content_type = 500, b'{"error":"response_too_large"}', "application/json; charset=utf-8"
        # Record the sanitized attempt before a client can observe the complete
        # response. No thread race should leave a completed response unlogged.
        command = getattr(self, "command", None)
        event = {"event": "http_request", "method": command if command in {"GET", "POST"} else "OTHER",
                 "route": getattr(self, "route_name", "rejected"), "status": status}
        self.server.log_stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        self.server.log_stream.flush()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", self.server.csp)
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        if getattr(self, "command", None) != "HEAD":
            self.wfile.write(raw)
        self.wfile.flush()
        if status >= 400:
            self._drain_rejected_input()

    def _drain_rejected_input(self):
        if getattr(self, "_body_consumed", False):
            return
        headers = getattr(self, "headers", None)
        lengths = headers.get_all("Content-Length", []) if headers is not None else []
        if len(lengths) == 1 and re.fullmatch(r"[0-9]{1,9}", lengths[0]):
            remaining = min(int(lengths[0]), REJECT_DRAIN_BYTES)
        elif headers is not None and not lengths and self.command in {"GET", "HEAD"}:
            return
        else:
            remaining = REJECT_DRAIN_BYTES
        if not remaining:
            return
        deadline = time.monotonic() + REJECT_DRAIN_SECONDS
        stream = getattr(self.rfile, "stream", self.rfile)
        try:
            # Half-close the response side, keeping a brief bounded receive
            # window for data already being sent. Never interpret these bytes.
            self.connection.shutdown(socket.SHUT_WR)
            while remaining:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                self.connection.settimeout(timeout)
                chunk = stream.read1(min(8192, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
        except (OSError, ValueError):
            pass

    def _json(self, status: int, value: dict):
        try:
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, RecursionError, OverflowError):
            status, raw = 500, b'{"error":"response_invalid"}'
        self._write(status, raw, "application/json; charset=utf-8")

    def _body(self):
        content_type = self.headers.get("Content-Type", "").lower().replace(" ", "")
        if content_type not in {"application/json", "application/json;charset=utf-8"}:
            raise RequestError("json_content_type_required", 415)
        length = self.headers.get("Content-Length")
        if length is None:
            raise RequestError("content_length_required", 411)
        if not re.fullmatch(r"[0-9]{1,9}", length):
            raise RequestError("content_length_invalid")
        size = int(length)
        if size > MAX_BODY_BYTES:
            raise RequestError("body_too_large", 413)
        raw = self.rfile.read(size)
        self._body_consumed = len(raw) == size
        if len(raw) != size:
            raise RequestError("body_incomplete")
        return strict_object(raw)

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def _handle(self):
        try:
            if self.command == "GET" and self.headers.get("Content-Length", "0") != "0":
                raise RequestError("get_body_rejected")
            if self.path == "/" and self.command == "GET":
                self.route_name = "ui"
                self._write(200, self.server.ui, "text/html; charset=utf-8")
                return
            supplied = self.headers.get("Authorization", "")
            expected = "Bearer " + self.server.token
            if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("ascii")):
                raise RequestError("authentication_required", 401)
            if self.path == "/health" and self.command == "GET":
                self.route_name = "health"
                result = self.server.service.health()
            elif self.path == "/runs" and self.command == "POST":
                self.route_name = "start"
                body = self._body()
                if (set(body) - {"run_id", "max_steps"} or not isinstance(body.get("run_id"), str)
                        or not _RUN_ID.fullmatch(body["run_id"])):
                    raise RequestError("start_request_invalid")
                steps = body.get("max_steps")
                if steps is not None and (type(steps) is not int or not 1 <= steps <= 12):
                    raise RequestError("max_steps_invalid")
                result = self.server.service.start(body["run_id"], max_steps=steps)
            else:
                match = _RUN_ROUTE.fullmatch(self.path)
                if not match:
                    raise RequestError("route_not_found", 404)
                run_id, action = match.groups()
                if self.command == "GET" and action is None:
                    self.route_name = "status"
                    result = self.server.service.status(run_id)
                elif self.command == "GET" and action == "review":
                    self.route_name = "review"
                    result = self.server.service.review(run_id)
                elif self.command == "POST" and action == "review":
                    self.route_name = "submit"
                    result = self.server.service.submit(run_id, self._body())
                elif self.command == "POST" and action == "export":
                    self.route_name = "export"
                    body = self._body()
                    if set(body) != {"expected_state_id"} or not isinstance(body["expected_state_id"], str):
                        raise RequestError("export_request_invalid")
                    result = self.server.service.export(run_id, body["expected_state_id"])
                else:
                    raise RequestError("method_not_allowed", 405)
            self._json(200, result)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (TimeoutError, socket.timeout):
            self._json(408, {"error": "request_timeout"})
        except Exception as error:
            code = getattr(error, "code", "internal_error")
            status = getattr(error, "status_code", 500)
            if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", code):
                code = "internal_error"
            if type(status) is not int or status not in {400, 401, 403, 404, 405, 408, 409, 411, 413, 415, 422, 429, 500, 503, 507}:
                status = 500
            self._json(status, {"error": code})


def make_server(service: Service, token: str, *, host: str = "127.0.0.1", port: int = 8765,
                log_stream: TextIO | None = None) -> PilotHTTPServer:
    return PilotHTTPServer((host, port), service, token, log_stream)


def _verify_loaded_package(root: Path) -> None:
    """Production must execute pinned bundle code, not a checkout beside it.

    The library adapter stays injectable for tests. This startup boundary also
    checks package search paths so later imports cannot fall back to another
    meta_data_extraction installation.
    """
    from .pilot_bundle import MARKER, verify_bundle
    verify_bundle(root)
    root = root.resolve(strict=True)
    package = root / "src/meta_data_extraction"
    pins = json.loads((root / MARKER).read_bytes())["files"]
    for name, module in tuple(sys.modules.items()):
        spec_name = getattr(getattr(module, "__spec__", None), "name", None)
        if not (name == "meta_data_extraction" or name.startswith("meta_data_extraction.")
                or spec_name == "meta_data_extraction.agent.pilot_http"):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise ValueError("package_origin_rejected")
        path = Path(filename).resolve(strict=True)
        if not path.is_relative_to(package):
            raise ValueError("package_origin_rejected")
        relative = path.relative_to(root).as_posix()
        if (relative not in pins or path.suffix != ".py"
                or hashlib.sha256(path.read_bytes()).hexdigest() != pins[relative]["sha256"]):
            raise ValueError("package_origin_rejected")
        for search in getattr(module, "__path__", ()):
            directory = Path(search).resolve(strict=True)
            if not directory.is_relative_to(package):
                raise ValueError("package_origin_rejected")
            init = (directory / "__init__.py").relative_to(root).as_posix()
            if init not in pins:
                raise ValueError("package_origin_rejected")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="Serve one verified isolated mock bundle on loopback.")
    serve.add_argument("--root", required=True, help="Verified isolated pilot bundle directory.")
    serve.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        token = validate_token(os.environ.get("META_AGENT_PILOT_TOKEN"))
        from .pilot_core import PilotService
        _verify_loaded_package(Path(args.root))
        service = PilotService(Path(args.root))
        with make_server(service, token, host=args.host, port=args.port) as server:
            print(json.dumps({"event": "pilot_listening", "host": "127.0.0.1", "port": server.server_address[1],
                              "mode": "synthetic_offline", "live_requests": 0}), flush=True)
            server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        return 0
    except Exception:
        print('{"error":"pilot_start_failed"}', file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
