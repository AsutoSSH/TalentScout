from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Form
)

from pathlib import Path

import shutil
from uuid import uuid4
import os


from pydantic import BaseModel, model_validator
import json
from typing import List, Optional, Literal

from backend.core.orchestration import (
    initialize_interview_session
)

from backend.utils.mongodb_helpers import(
    load_session
)

router = APIRouter(
    prefix="/session",
    tags=["Session"]
)

UPLOAD_DIR = "uploaded_resumes"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ==================================================
# Pydantic Schemas
# ==================================================

class CreateSessionPayload(BaseModel):
    session_id: str | None = None
    rag_mode: str

    # Optional but highly recommended helper to parse JSON if sent as a single string
    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return json.loads(value)
        return value

def get_session_payload(
    session_id: str | None = Form(None),
    rag_mode: str = Form(...)
) -> CreateSessionPayload:
    """Extracts individual form fields and maps them into your Pydantic model."""
    return CreateSessionPayload(session_id=session_id, rag_mode=rag_mode)


class SessionCreateResponse(BaseModel):
    session_id: str
    skills: List[str]
    domains: List[str]
    rag_mode: str

class SessionSummaryResponse(BaseModel):
    session_id: str
    role: Optional[str]
    current_topic_index: int
    current_subtopic_index: int
    total_questions: int
    completed_questions: int
    rag_mode: str
    phase: str


class DeleteSessionResponse(BaseModel):
    deleted: bool


# =========================================================
# Create Session
# =========================================================

@router.post("/create", response_model=SessionCreateResponse)

async def create_session(

    payload: CreateSessionPayload = Depends(get_session_payload),
    resume: UploadFile = File(...),
):
    session_id = payload.session_id or str(uuid4())

    file_ext = Path(
        resume.filename
    ).suffix

    filename = (
        f"{str(uuid4())}{file_ext}"
    )

    save_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(save_path, "wb") as buffer:

        shutil.copyfileobj(
            resume.file,
            buffer
        )

    session_id, state = (

        await initialize_interview_session(
            pdf_path=save_path,
            session_id=session_id
        )
    )

    state["rag_mode"] = payload.rag_mode

    return SessionCreateResponse(

        session_id=session_id,

        skills=state.get(
            "extracted_skills",
            []
        ),

        domains=state.get(
            "extracted_domains",
            []
        ),

        rag_mode=payload.rag_mode
    )


# =========================================================
# Get Session
# =========================================================

@router.get("/{session_id}")

async def get_session(
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

    return SessionSummaryResponse(

        session_id=session_id,

        role=state.get(
            "job_role"
        ),

        current_topic_index=state.get(
            "current_topic_index",
            0
        ),

        current_subtopic_index=state.get(
            "current_subtopic_index",
            0
        ),

        total_questions=len(
            state.get(
                "question_history",
                []
            )
        ),

        completed_questions=len(
            state.get(
                "evaluations",
                []
            )
        ),

        rag_mode=state.get(
            "rag_mode",
            "normal"
        ),

        phase=state.get(
            "phase",
            "unknown"
        )
    )

# =========================================================
# Delete Session
# =========================================================

# @router.delete(

#     "/{session_id}",

#     response_model=DeleteSessionResponse
# )

# async def delete_session_endpoint(
#     session_id: str
# ):

#     success = delete_session(
#         session_id
#     )

#     return DeleteSessionResponse(
#         deleted=success
#     )