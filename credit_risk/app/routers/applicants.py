from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/applicants", tags=["applicants"])


@router.get("/list")
async def list_applicants(request: Request):
    """Return { ids: [...] } — shape expected by the Lovable frontend."""
    return request.app.state.data.get_applicant_list()


@router.get("/search")
async def search_applicants(request: Request, q: str = ""):
    return request.app.state.data.search_applicants(q)


@router.get("/{applicant_id}")
async def get_applicant(request: Request, applicant_id: str):
    data = request.app.state.data
    if applicant_id not in data.sample_explanations:
        available = ", ".join(sorted(data.sample_ids))
        raise HTTPException(
            status_code=404,
            detail=(
                f"Only sample applicants are available in demo mode. "
                f"Available IDs: {available}"
            ),
        )
    return data.get_applicant_detail(applicant_id)
