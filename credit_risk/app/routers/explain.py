from fastapi import APIRouter, HTTPException, Request

from app.narrative import generate_credit_narrative
from app.schemas import NarrativeRequest

router = APIRouter(prefix="/api/explain", tags=["explain"])


@router.post("/narrative")
async def create_narrative(request: Request, body: NarrativeRequest):
    data = request.app.state.data
    applicant_id = body.applicant_id

    if force:
        data.narrative_cache.pop(applicant_id, None)

    if applicant_id in data.narrative_cache:
        return data.narrative_cache[applicant_id]

    if applicant_id not in data.sample_explanations:
        available = ", ".join(sorted(data.sample_ids))
        raise HTTPException(
            status_code=404,
            detail=(
                f"Only sample applicants are available in demo mode. "
                f"Available IDs: {available}"
            ),
        )

    explanation = data.sample_explanations[applicant_id]
    result = await generate_credit_narrative(explanation)
    data.narrative_cache[applicant_id] = result
    return result


@router.get("/counterfactuals/{applicant_id}")
async def get_counterfactuals(request: Request, applicant_id: str):
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
    explanation = data.sample_explanations[applicant_id]
    return explanation.get("counterfactuals") or []
