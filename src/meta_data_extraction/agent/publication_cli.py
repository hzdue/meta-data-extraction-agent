"""Explicit local publication run, status, review and authenticated JSON service."""
from __future__ import annotations

import argparse
import json
import os

from .paths import workspace_root
from .pilot_core import PilotError
from .pilot_http import validate_token
from .publication_service import PublicationService, make_publication_server
from .publication_types import load_publication_spec
from .review_cli import load_response


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "review", "submit", "export", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--root", default=".")
        command.add_argument("--spec", required=True,
                             help="Explicit frozen specification under configs/agent or runs/agent.")
        if name in {"run", "serve"}:
            command.add_argument("--chatai-endpoint", help="Trusted configured endpoint; required for live dispatch.")
            command.add_argument("--load-dotenv", action="store_true",
                                 help="Explicitly load credentials from the workspace .env without printing them.")
        if name == "run":
            command.add_argument("--max-steps", type=int, choices=range(1, 13))
        if name == "submit":
            command.add_argument("--response", required=True,
                                 help="Explicit JSON in the parent run's review-responses/ or fixture directory.")
        if name == "export":
            command.add_argument("--expected-state-id", required=True, help="Current review state ID from review.")
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
            command.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        root = workspace_root(args.root)
        spec = load_publication_spec(root, args.spec)
        service = PublicationService(root, spec,
            configured_chatai_endpoint=getattr(args, "chatai_endpoint", None),
            load_dotenv_file=getattr(args, "load_dotenv", False))
        if args.command == "run":
            result = service.start(spec.run_id, max_steps=args.max_steps)
        elif args.command == "status":
            result = service.status(spec.run_id)
        elif args.command == "review":
            result = service.review(spec.run_id)
        elif args.command == "submit":
            response = load_response(root, spec.run_id, args.response)
            result = service.submit(spec.run_id, response.model_dump(mode="json"))
        elif args.command == "export":
            result = service.export(spec.run_id, args.expected_state_id)
        else:
            token = validate_token(os.environ.get("META_AGENT_PILOT_TOKEN"))
            with make_publication_server(service, token, host=args.host, port=args.port) as server:
                print(json.dumps({"status": "listening", "host": "127.0.0.1",
                    "port": server.server_port, "run_id": spec.run_id,
                    "mode": spec.mode, "interface": "authenticated_json_api"}), flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
            return 0
    except PilotError as error:
        parser.exit(2, f"Publication command rejected ({error.code}).\n")
    except Exception:
        parser.exit(2, "Publication command failed; check the scope, inputs and local state.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if args.command == "run" and result.get("status") not in {"finished", "pending_review", "checkpointed"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
