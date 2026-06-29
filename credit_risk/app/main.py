from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.inference import state
from app.routers import applicants, dashboard, explain

# Load .env from the credit_risk directory
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# New Lovable frontend build output
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "lens-credit-insight-main" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load_all()
    app.state.data = state
    logger.info("✓ API ready at http://localhost:8000")
    yield


app = FastAPI(title="CreditLens API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(applicants.router)
app.include_router(explain.router)


# Serve the Lovable frontend as a static SPA (when built)
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            return {"detail": "Not found"}
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
