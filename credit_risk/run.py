"""Start CreditLens: build frontend and serve on port 8000."""

import subprocess
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


def main() -> None:
    print("Building frontend...")
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=FRONTEND,
        shell=sys.platform == "win32",
        check=False,
    )
    if result.returncode != 0:
        print("Frontend build failed. Fix errors above and retry.")
        sys.exit(1)

    print("Starting server at http://localhost:8000")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
