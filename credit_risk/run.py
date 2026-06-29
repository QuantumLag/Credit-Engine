"""Start CreditLens API on port 8000.

The Lovable frontend (lens-credit-insight-main) is served separately via its own
dev server (npm run dev / bun dev inside that directory).
For a production bundle, build the frontend first, then this server will
automatically serve it from lens-credit-insight-main/dist.
"""

import subprocess
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "lens-credit-insight-main"


def _build_frontend() -> bool:
    """Build the Lovable frontend. Returns True on success."""
    if not FRONTEND.exists():
        print("⚠  Frontend directory not found — skipping build.")
        return False

    print("Building frontend (this may take a minute)…")
    # Prefer bun; fall back to npm
    for cmd in (["bun", "run", "build"], ["npm", "run", "build"]):
        result = subprocess.run(
            cmd,
            cwd=FRONTEND,
            shell=(sys.platform == "win32"),
            capture_output=False,
            check=False,
        )
        if result.returncode == 0:
            print(f"✓ Frontend built with {cmd[0]}")
            return True
        if cmd[0] == "bun":
            print("  bun not found or failed — trying npm…")

    print("❌ Frontend build failed.")
    return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CreditLens API server")
    parser.add_argument(
        "--build-frontend",
        action="store_true",
        default=False,
        help="Build the Lovable frontend before starting the API server",
    )
    args = parser.parse_args()

    if args.build_frontend:
        _build_frontend()

    print("Starting CreditLens API at http://localhost:8000")
    print("Frontend dev server: cd lens-credit-insight-main && bun dev")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
