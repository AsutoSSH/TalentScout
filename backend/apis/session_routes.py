import json
import os
import shutil
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Form, UploadFile
from pydantic import BaseModel, model_validator

from backend.core.orchestration import initialize_interview_session
from backend.utils.mongodb_helpers import load_session

router = APIRouter(prefix="/session", tags=["Session"])
UPLOAD_DIR = "uploaded_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==================================================
# Pydantic Schemas & Helpers
# ==================================================

class CreateSessionPayload(BaseModel):
    session_id: Optional[str] = None
    rag_mode: str

    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        return json.loads(value) if isinstance(value, str) else value

def get_session_payload(
    session_id: Optional[str] = Form(None),
    rag_mode: str = Form(...)
) -> CreateSessionPayload:
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
# Endpoints
# =========================================================

@router.post("/create", response_model=SessionCreateResponse)
async def create_session(
    payload: CreateSessionPayload = Depends(get_session_payload),
    resume: UploadFile = File(...),
):
    session_id = payload.session_id or str(uuid4())
    file_ext = Path(resume.filename).suffix
    save_path = os.path.join(UPLOAD_DIR, f"{str(uuid4())}{file_ext}")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)

    session_id, state = await initialize_interview_session(pdf_path=save_path, session_id=session_id)
    state["rag_mode"] = payload.rag_mode

    return SessionCreateResponse(
        session_id=session_id,
        skills=state.get("extracted_skills", []),
        domains=state.get("extracted_domains", []),
        rag_mode=payload.rag_mode
    )

@router.get("/{session_id}", response_model=SessionSummaryResponse)
async def get_session(session_id: str):
    state = load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionSummaryResponse(
        session_id=session_id,
        role=state.get("job_role"),
        current_topic_index=state.get("current_topic_index", 0),
        current_subtopic_index=state.get("current_subtopic_index", 0),
        total_questions=len(state.get("question_history", [])),
        completed_questions=len(state.get("evaluations", [])),
        rag_mode=state.get("rag_mode", "normal"),
        phase=state.get("phase", "unknown")
    )