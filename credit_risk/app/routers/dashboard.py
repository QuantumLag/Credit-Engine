from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/overview")
async def dashboard_overview(request: Request):
    return request.app.state.data.get_overview()


@router.get("/feature-importance")
async def feature_importance(request: Request):
    return request.app.state.data.get_feature_importance()


@router.get("/thin-file")
async def thin_file_analysis(request: Request):
    return request.app.state.data.get_thin_file()
