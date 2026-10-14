"""Prepare or serve the isolated M10 mock pilot using the existing environment.

No installation, .env loading, inference call or public listener. New bundles
never replace an existing directory. See docs/agent/PILOT_TRIAL.md for the trial.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess


def bundle_id(value: str) -> str:
    if not re.fullmatch(r"m10-[a-z0-9][a-z0-9_-]{0,39}", value):
        raise argparse.ArgumentTypeError("Use a short m10- name with lowercase letters, digits, dashes or underscores.")
    return value


def port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Port must be an integer from 1 through 65535.") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be an integer from 1 through 65535.")
    return port


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id", required=True, type=bundle_id)
    parser.add_argument("--port", type=port_number, default=8765)
    parser.add_argument("--prepare-only", action="store_true", help="Build/verify only; no token, clipboard write or listener.")
    parser.add_argument("--resume-bundle", action="store_true", help="Explicitly verify and reuse an existing bundle.")
    parser.add_argument("--token-mode", choices=("clipboard", "environment"), default="clipboard",
                        help="Generate a temporary clipboard token, or use META_AGENT_PILOT_TOKEN already in the environment.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    runtime = root / "runs/agent" / args.bundle_id / "runtime"
    if not python.is_file():
        parser.exit(2, "The repository .venv is missing. Follow docs/agent/PILOT.md for setup.\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "src"))
    if not args.prepare_only and args.token_mode == "environment" and not env.get("META_AGENT_PILOT_TOKEN"):
        parser.exit(2, "Environment mode needs a separate META_AGENT_PILOT_TOKEN; never use the ChatAI key.\n")
    command = [str(python), "-m", "meta_data_extraction.agent.pilot_bundle"]
    command += (["verify", "--root", str(runtime)] if args.resume_bundle else
                ["build", "--root", str(root), "--run-id", args.bundle_id])
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    if result.returncode:
        parser.exit(2, "Bundle preparation failed; existing artifacts were retained. Use --resume-bundle only for a verified existing bundle.\n")
    try:
        manifest = json.loads(result.stdout)
        if manifest.get("verified") is not True:
            raise ValueError("unverified")
        digest = manifest["manifest_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("invalid digest")
    except (ValueError, KeyError, TypeError, AttributeError):
        parser.exit(2, "Bundle verification returned an invalid result; no listener started.\n")
    print(f"Verified isolated bundle: {runtime}", flush=True)
    print(f"Manifest SHA256: {digest}", flush=True)
    if args.prepare_only:
        print("Prepared only: no token, clipboard write, listener, or model request.", flush=True)
        print(f"Later: .\\.venv\\Scripts\\python.exe scripts/agent/start_pilot.py --bundle-id {args.bundle_id} --resume-bundle", flush=True)
        return 0

    if args.token_mode == "clipboard":
        shell = (shutil.which("powershell.exe") or shutil.which("pwsh.exe")) if os.name == "nt" else None
        if shell is None:
            parser.exit(2, "Clipboard mode needs Windows PowerShell. Use --token-mode environment with a separate pilot token.\n")
        env["META_AGENT_PILOT_TOKEN"] = secrets.token_urlsafe(32)
        try:
            copied = subprocess.run(
                [shell, "-NoProfile", "-NonInteractive", "-Command", "Set-Clipboard -Value ([Console]::In.ReadToEnd())"],
                input=env["META_AGENT_PILOT_TOKEN"], capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if copied.returncode:
                raise OSError("clipboard unavailable")
        except (OSError, subprocess.SubprocessError):
            parser.exit(2, "Clipboard copy failed; no listener started. Use --token-mode environment with a separate pilot token.\n")
        print("Temporary pilot token copied to the clipboard. Paste into Access token; clear the clipboard when finished.", flush=True)
    env["PYTHONPATH"] = str(runtime / "src")
    print(f"Open http://127.0.0.1:{args.port} and follow docs/agent/PILOT_TRIAL.md. Stop with Ctrl+C.", flush=True)
    result = subprocess.run(
        [str(python), "-m", "meta_data_extraction.agent.pilot_http", "serve",
         "--root", str(runtime), "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=runtime, env=env,
    )
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0) from None
    except OSError:
        raise SystemExit("Local launch failed; no credential details are displayed.") from None
