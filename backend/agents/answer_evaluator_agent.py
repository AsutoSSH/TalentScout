from typing import List

from pydantic import BaseModel, Field

from backend.core.agent_base import AgentBase


# =========================================================
# Structured Evaluation Schema
# =========================================================

class AnswerEvaluation(BaseModel):

    candidate_response: str = Field(
        description=(
            "Exact candidate response being evaluated"
        )
    )

    correctness: str = Field(
        description=(
            "One of: Correct, "
            "Partially Correct, Incorrect"
        )
    )

    missing_points: List[str] = Field(
        description="Important missing concepts"
    )

    ideal_answer: str = Field(
        description="Ideal concise technical answer"
    )

    improvement_suggestion: str = Field(
        description="How candidate can improve"
    )

    improved_version_of_answer: str = Field(
        description=(
            "An improved version of the candidate's "
            "answer while preserving their intent"
        )
    )

    score: float = Field(
        ge=0,
        le=10,
        description="Evaluation score out of 10"
    )


# =========================================================
# Answer Evaluator Agent
# =========================================================

class AnswerEvaluatorAgent(AgentBase):

    async def execute(
        self,
        question: str,
        answer: str,
        job_role: str,
        skills: List[str],
    ) -> AnswerEvaluation:

        # ==========================================
        # Base Prompt
        # ==========================================

        chat_history = [

            {
                "role": "system",

                "content": """
                You are a strict but fair technical interviewer.

                Evaluate candidate answers objectively.

                Guidelines:
                - Be concise
                - Be technically accurate
                - Do NOT over-penalize short answers
                - If the candidate is directionally correct,
                  acknowledge it
                - Focus on conceptual understanding
                - Preserve candidate intent when improving
                  their answer
                """
            },

            {
                "role": "system",

                "content": f"""
                Candidate Role:
                {job_role}

                Candidate Skills:
                {", ".join(skills)}
                """
            }
        ]

        # ==========================================
        # Evaluation Request
        # ==========================================

        chat_history.append(

            {
                "role": "user",

                "content": f"""
                Interview Question:
                {question}

                Candidate Answer:
                {answer}

                Evaluate the answer.

                Requirements:
                - Store the exact candidate answer
                - Identify correctness
                - Mention missing concepts
                - Provide an ideal concise answer
                - Suggest improvements
                - Rewrite the candidate answer into a
                  stronger improved version while
                  preserving the candidate's intent
                - Give a score out of 10
                """
            }
        )

        # ==========================================
        # Structured Output
        # ==========================================

        result = await self.call_model(

            chat_history=chat_history,

            response_model=AnswerEvaluation,

            temperature=0.0,

            max_tokens=1200
        )

        return result.parsed