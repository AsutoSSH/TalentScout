from uuid import uuid4

from fastapi import (
    APIRouter,
    HTTPException
)

from pydantic import (
    BaseModel,
    Field
)

from typing import (
    List,
    Optional,
    Literal
)

from backend.core.orchestration import (

    submit_role,

    submit_answer
)

# =========================================================
# Router
# =========================================================

router = APIRouter(

    prefix="/interview",

    tags=["Interview"]
)

# =========================================================
# Payload Schemas
# =========================================================

class RoleSubmissionPayload(BaseModel):

    session_id: str

    role: str = Field(
        min_length=2,
        max_length=100
    )


class AnswerSubmissionPayload(BaseModel):

    session_id: str

    answer: str = Field(
        min_length=1
    )


# =========================================================
# Response Schemas
# =========================================================

class EvaluationSchema(BaseModel):

    question: str

    answer: str

    correctness: str

    score: float

    missing_points: List[str]

    ideal_answer: Optional[str]

    improvement_suggestion: str


class RoleSubmissionResponse(BaseModel):

    session_id: str

    question: str

    topic: str

    subtopic: str

    difficulty: int

    source_mode: Literal[
        "normal",
        "book_rag"
    ]


class AnswerSubmissionResponse(BaseModel):

    evaluation: EvaluationSchema

    next_question: str

    topic: str

    subtopic: str

    difficulty: int

    source_mode: Literal[
        "normal",
        "book_rag"
    ]


# =========================================================
# Submit Role
# =========================================================

@router.post(

    "/role",

    response_model=RoleSubmissionResponse
)

async def role_endpoint(

    payload: RoleSubmissionPayload
):
    
    session_id = payload.session_id or str(uuid4())
    try:

        state = await submit_role(

            session_id=session_id,

            role=payload.role
        )

        latest_q = state[
            "question_history"
        ][-1]

        return RoleSubmissionResponse(

            session_id=session_id,

            question=latest_q["question"],

            topic=latest_q["topic"],

            subtopic=latest_q["subtopic"],

            difficulty=latest_q["difficulty"],

            source_mode=latest_q[
                "source_mode"
            ]
        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )


# =========================================================
# Submit Answer
# =========================================================

@router.post(

    "/answer",

    response_model=AnswerSubmissionResponse
)

async def answer_endpoint(

    payload: AnswerSubmissionPayload
):
    
    session_id = payload.session_id or str(uuid4())
    try:

        result = await submit_answer(

            session_id=session_id,
            answer=payload.answer
        )

        latest_q = result[
            "question_doc"
        ]

        return AnswerSubmissionResponse(

            evaluation=EvaluationSchema(
                **result["evaluation"]
            ),

            next_question=result[
                "next_question"
            ],

            topic=latest_q["topic"],

            subtopic=latest_q["subtopic"],

            difficulty=latest_q["difficulty"],

            source_mode=latest_q[
                "source_mode"
            ]
        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )