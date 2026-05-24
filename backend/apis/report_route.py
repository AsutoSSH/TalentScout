from fastapi import (
    APIRouter,
    HTTPException
)

from fastapi.responses import (
    StreamingResponse
)

from pydantic import BaseModel

import asyncio

from backend.utils.mongodb_helpers import (
    load_session
)

from backend.agents.final_report_generator import (
    generate_interview_report
)

# =========================================================
# Router
# =========================================================

router = APIRouter(

    prefix="/report",

    tags=["Report"]
)

# =========================================================
# Response Schemas
# =========================================================

class ReportMetadataResponse(BaseModel):

    session_id: str

    report_available: bool

    total_questions: int

    average_score: float

    completion_status: str


# =========================================================
# Report Metadata
# =========================================================

@router.get(

    "/{session_id}/metadata",

    response_model=ReportMetadataResponse
)

async def report_metadata(
    session_id: str
):

    state = load_session(
        session_id
    )

    if state is None:

        raise HTTPException(

            status_code=404,

            detail="Session not found"
        )

    evaluations = state.get(
        "evaluations",
        []
    )

    scores = [

        e.get("score", 0)

        for e in evaluations
    ]

    avg_score = (

        sum(scores) / len(scores)

        if scores else 0
    )

    return ReportMetadataResponse(

        session_id=session_id,

        report_available=len(
            evaluations
        ) > 0,

        total_questions=len(
            state.get(
                "question_history",
                []
            )
        ),

        average_score=round(
            avg_score,
            2
        ),

        completion_status=state.get(
            "phase",
            "unknown"
        )
    )


# =========================================================
# Download PDF Report
# =========================================================

@router.get(
    "/{session_id}"
)

async def download_report(
    session_id: str
):

    state = load_session(
        session_id
    )

    if state is None:

        raise HTTPException(

            status_code=404,

            detail="Session not found"
        )

    pdf_buffer = await asyncio.to_thread(generate_interview_report,state)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
            "attachment; filename=interview_report.pdf"
        }
    )