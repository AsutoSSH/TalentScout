from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Union

from backend.core.agent_base import AgentBase


class LLMQuestionOutput(BaseModel):
    question: str = Field(..., description="The interview question")
    topic: str = Field(..., description="The main topic of the question")
    subtopic: str = Field(..., description="The sub-topic of the question")
    difficulty: str = Field(..., description="e.g., easy, medium, hard")
    reasoning: str = Field(..., description="Why this question is being asked")

class InterviewQuestion(BaseModel):
    question: str
    topic: str
    subtopic: str
    difficulty: str
    reasoning: str
    source_mode: Optional[Literal["book_rag", "normal"]] = None
    retrieved_context: Optional[List[dict]] = None

class InterviewTopic(BaseModel):

    topic: str

    retrieval_query: str = Field(..., description="Query to be used to retrieve topic chunks.")

    subtopics: List[str]


class InterviewPlan(BaseModel):

    role: str

    topics: List[
        InterviewTopic
    ]

class InterviewPlannerAgent(AgentBase):

    async def execute(
        self,
        state: dict
    ) -> InterviewPlan:

        role = state.get(
            "job_role",
            "Unknown"
        )

        skills = state.get(
            "extracted_skills",
            []
        )

        domains = state.get(
            "extracted_domains",
            []
        )

        years = state.get(
            "years_of_experience",
            0
        )

        chat_history = [

            {
                "role": "system",

                "content": """
You are an expert technical interviewer.

Generate a realistic interview roadmap.

Rules:

- Generate diverse interview topics.
- Maintain natural interview flow.
- Topics should support follow-up questions.
- Progress from easier to harder concepts.
- Avoid repetitive semantic areas.
- Generate retrieval-friendly search queries for chunk retrieval of textbook content using vector search. Avoid using phrases like "interview questions" in the query. The chunks DO NOT CONTAIN LITERAL QUESTIONS.
- Cover both theory and practical systems.
- Prefer realistic industry interview progression.
"""
            },

            {
                "role": "user",

                "content": f"""
Role:
{role}

Years of Experience:
{years}

Skills:
{skills}

Domains:
{domains}

Generate a complete structured interview plan. Prefer questions aligned with given role.
"""
            }
        ]

        result = await self.call_model(

            chat_history=chat_history,

            response_model=InterviewPlan,
            temperature=0.5
        )

        return result.parsed


class InterviewQuestionAgent(AgentBase):

    async def execute(
        self,
        job_role: str,
        skills: List[str],
        domains: List[str],
        experience: str,
        previous_questions: List[str],
        current_topic: str,
        difficulty_level: Union[str, int],
        current_subtopic: str,
        mode: Literal[
        "normal",
        "book_rag"
        ] = "normal",
        retrieved_context: List[dict] = None,
        
    ) -> InterviewQuestion:

        chat_history = [

            {
                "role": "system",

                "content": """
                You are an expert technical interviewer.

                Rules:
                - Ask ONE question at a time
                - Prefer conceptual understanding
                - Avoid repeating topics
                - Use conversational continuity
                - Questions should feel natural
                - Prefer short questions: Answer to the questions should be within 2 lines
                Progression Rules:
                - If continuing the same topic,
                ask a deeper conceptual question.
                - Gradually increase difficulty.
                - Avoid repeating the same concept.
                - Behave like a real technical interviewer.
                """
            },

            {
                "role": "system",

                "content": f"""
                Candidate Role:
                {job_role}

                Skills:
                {skills}

                Domains:
                {domains}

                Experience:
                {experience}

                Current Interview Topic:
                {current_topic}

                Current Difficulty Level:
                {difficulty_level}

                Current Subtopic:
                {current_subtopic}
                Generate a question focused on the current subtopic while maintaining natural interview continuity.
                """
            }
        ]

        # ==========================================
        # RAG Mode Instructions
        # ==========================================

        if mode == "book_rag":

            chat_history.append(

                {
                    "role": "system",

                    "content": """
                    IMPORTANT:
                    Use the retrieved textbook material
                    as the PRIMARY grounding source
                    for generating the question.

                    Rules:
                    - Do NOT copy textbook lines
                    - Prefer conceptual understanding. Ask a mix between memory based and conceptual question.
                    - Ask interview-style questions
                    - Ask practical understanding questions
                    """
                }
            )

        # ==========================================
        # Retrieved Context
        # ==========================================



        if (
            mode == "book_rag"
            and retrieved_context
        ):

            context_text = "\n\n".join(
                [

                    f"""
                    CHUNK_ID: {i}

                    Source Book:
                    {chunk["source_book"]}

                    Page:
                    {chunk["page"]}

                    Content:
                    {chunk["text"]}
                    """

                    for i, chunk in enumerate(
                        retrieved_context
                    )
                ]
            )

            chat_history.append(

                {
                    "role": "system",

                    "content": f"""
                    Retrieved Reference Material:

                    {context_text}
                    """
                }
            )

        # ==========================================
        # Final Prompt
        # ==========================================

        if mode == "book_rag":

            final_prompt = f"""
            Previous Questions:
            {previous_questions}
            Generate the next interview question. STRICTLY Do not repeat any previous questions.

            Requirements:
            - Use retrieved textbook material heavily
            - Ask interview questions
            - Avoid direct textbook recall
            - Prefer practical understanding 
            - Prefer questions with very concise answers within 50 words
            """

        else:

            final_prompt = f"""
            Previous Questions:
            {previous_questions}
            Generate the next interview question. STRICTLY Do not repeat any previous questions.

            Requirements:
            - Prefer conceptual understanding. Ask a mix between memory based and conceptual questions.
            - Avoid repeating previous topics
            - Could be a follow-up question
            - Prefer questions with concise answers within 50 words
            """

        chat_history.append(

            {
                "role": "user",

                "content": final_prompt
            }
        )

        result = await self.call_model(
            chat_history=chat_history,
            response_model=LLMQuestionOutput,
            temperature=0.3
        )

        parsed = result.parsed
        print(f"DIRECT LLM: {parsed}")

        final_question = InterviewQuestion(
            question=parsed.question,
            topic=parsed.topic,
            subtopic=parsed.subtopic,
            difficulty=parsed.difficulty,
            reasoning=parsed.reasoning,
            source_mode=mode,                       
            retrieved_context=retrieved_context     
        )

        return final_question