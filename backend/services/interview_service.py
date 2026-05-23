from typing import TypedDict, List, Optional, Annotated
from operator import add

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)

# =========================================================
# Agents and Retriever
# =========================================================

from backend.agents.profile_extraction_agent import (
    ProfileExtractorAgent, extract_text_from_pdf
)

from backend.agents.interview_question_agent import (
    InterviewQuestionAgent
)

from backend.agents.answer_evaluator_agent import (
    AnswerEvaluatorAgent
)

# from backend.agents.final_report_agent import (
#     FinalReportAgent
# )

from backend.rag.retriever import retrieve_relevant_chunks

# =========================================================
# Initializing Agents
# =========================================================

profile_extractor = ProfileExtractorAgent(
    name="ProfileExtractor",
    verbose=True
)

question_agent = InterviewQuestionAgent(
    name="InterviewQuestionAgent",
    verbose=False
)

evaluation_agent = AnswerEvaluatorAgent(
    name="AnswerEvaluatorAgent",
    verbose=False
)

# report_agent = FinalReportAgent(
#     name="FinalReportAgent",
#     verbose=False
# )

# =========================================================
# LangGraph State
# =========================================================

class InterviewState(TypedDict):

    # Conversation history
    messages: Annotated[
        List[BaseMessage],
        add
    ]

    # Resume
    resume_pdf_path: Optional[str]
    resume_text: Optional[str]

    # Extracted profile
    extracted_skills: Optional[List[str]]
    extracted_domains: Optional[List[str]]
    years_of_experience: Optional[str]

    # Interview metadata
    job_role: Optional[str]
    matched_role: Optional[str]

    interview_plan: Optional[dict]

    current_plan_step: int

    # Question metadata
    last_question: Optional[str]
    question_history: List[dict]
    current_topic_index: Optional[int]

    current_subtopic_index: Optional[int]

    cached_retrieved_context: Optional[dict]
    query_history: List[str]

    # Evaluations
    evaluations: List[dict]

    # Workflow phase
    phase: str



# =========================================================
# Node 1 — Parse Resume
# =========================================================

async def parse_resume_node(
    state: InterviewState
):

    pdf_path = state.get(
        "resume_pdf_path"
    )

    if not pdf_path:

        return {

            "messages": [
                AIMessage(
                    content="Resume PDF path missing."
                )
            ],

            "phase":
                "completed"
        }

    # ==========================================
    # Extract text from PDF
    # ==========================================

    try:

        resume_text = extract_text_from_pdf(
            pdf_path
        )

    except Exception as e:

        return {

            "messages": [
                AIMessage(
                    content=f"Failed to parse resume PDF: {str(e)}"
                )
            ],

            "phase":
                "completed"
        }

    # ==========================================
    # Structured profile extraction
    # ==========================================

    profile = await profile_extractor.execute(
        resume_text
    )

    return {

        # Store raw text too
        "resume_text":
            resume_text,

        # Structured fields
        "extracted_skills":
            profile.extracted_skills,

        "extracted_domains":
            profile.extracted_domains,

        "years_of_experience":
            str(profile.years_of_experience),

        "phase":
            "collect_role"
    }


# =========================================================
# Node 2 — Collect Role
# =========================================================

def collect_job_role(
    state: InterviewState
):

    messages = state["messages"]

    # Save user role response
    if (
        messages and
        isinstance(messages[-1], HumanMessage)
    ):

        return {
            "job_role":
                messages[-1].content.strip(),

            "phase":
                "interview"
        }

    # Ask role
    return {

        "messages": [
            AIMessage(
                content=(
                    "Which role are you interviewing for?\n"
                    "Example: Backend Engineer, AI/ML Engineer"
                )
            )
        ],

        "phase":
            "collect_role"
    }

# ==============================
# Node 3 --- Question Asking Agent
# ================================

async def interview_question_node(state: InterviewState):

    # 1. Fetch history and current position in the plan
    previous_questions = [q["question"] for q in state.get("question_history", [])]
    plan = state["interview_plan"]
    t_idx = state["current_topic_index"]
    s_idx = state["current_subtopic_index"]

    if state["current_topic_index"] >= len(plan["topics"]):
        return {
            "phase": "completed"
        }
        
    topic_plan = plan["topics"][t_idx]
    current_topic = topic_plan["topic"]
    current_subtopic = topic_plan["subtopics"][s_idx]
    retrieval_query = topic_plan["retrieval_query"]
    
    mode = state.get("rag_mode", "normal")
    matched_role = state.get("matched_role", None)
    retrieved_context = None

    # 2. Handle context retrieval for book_rag mode
    if mode == "book_rag":
        if s_idx == 0 or state["cached_retrieved_context"] is None:
            print(f"Using query {retrieval_query}")
            retrieval_result = retrieve_relevant_chunks(
                matched_role=matched_role, query=retrieval_query
            )
            state["query_history"].append(retrieval_query)

            if retrieval_result["use_rag"]:
                retrieved_context = retrieval_result["retrieved_chunks"]
                state["cached_retrieved_context"] = retrieved_context
            else:
                mode = "normal"
        else:
            print("Using Cached Retrieval")
            retrieved_context = state["cached_retrieved_context"]

    # 3. Generate the interview question
    question = await question_agent.execute(
        job_role=state["job_role"],
        skills=state.get("extracted_skills", []),
        domains=state.get("extracted_domains", []),
        experience=state.get("years_of_experience", "Unknown"),
        previous_questions=previous_questions,
        mode=mode,
        retrieved_context=retrieved_context,
        current_topic=current_topic,
        current_subtopic=current_subtopic,
        difficulty_level=s_idx + 1,
    )

    # 4. Advance pointer index for the next node execution
    if s_idx < len(topic_plan["subtopics"]) - 1:
        state["current_subtopic_index"] += 1
    else:
        state["current_topic_index"] += 1
        state["current_subtopic_index"] = 0
        state["cached_retrieved_context"] = None  # Clear cache for the new topic

    # 5. Return updated state keys
    return {
        "messages": [AIMessage(content=question.question)],
        "last_question": question.question,
        "question_history": state.get("question_history", []) + [question.model_dump()],
        "current_topic_index": state["current_topic_index"],
        "current_subtopic_index": state["current_subtopic_index"],
        "cached_retrieved_context": state["cached_retrieved_context"],
        "phase": "evaluation"
    }

async def answer_evaluation_node(
    state: InterviewState
):

    if not state.get("last_question"):

        return {
            "phase": "interview"
        }

    if not state["messages"]:

        return {
            "phase": "interview"
        }

    # Ensure user answered
    if not isinstance(
        state["messages"][-1],
        HumanMessage
    ):

        return {
            "phase": "interview"
        }

    candidate_answer = (
        state["messages"][-1].content
    )

    # Blank answer
    if not candidate_answer:
        print("Candidate didnt answer")

        evaluation_doc = {

            "question":
                state["last_question"],

            "answer":
                "",

            "correctness":
                "Incorrect",

            "missing_points":
                [
                    "Candidate did not provide an answer."
                ],

            "ideal_answer":
                "",

            "improvement_suggestion":
                (
                    "Try to provide at least a brief response "
                    "explaining your understanding or approach."
                ),

            "score":
                0.0
        }

        return {

            "evaluations":
                state.get(
                    "evaluations",
                    []
                ) + [evaluation_doc],

            "phase":
                "interview"
        }


    # ==========================================
    # Structured evaluation
    # ==========================================

    evaluation = await evaluation_agent.execute(

        question=state["last_question"],

        answer=candidate_answer,

        job_role=state["job_role"],

        skills=state.get(
            "extracted_skills",
            []
        )
    )

    evaluation_doc = {

        "question":
            state["last_question"],

        "answer":
            candidate_answer,

        "correctness":
            evaluation.correctness,

        "missing_points":
            evaluation.missing_points,

        "ideal_answer":
            evaluation.ideal_answer,

        "improvement_suggestion":
            evaluation.improvement_suggestion,
        
        "improved_version_of_answer":
            evaluation.improved_version_of_answer,

        "score":
            evaluation.score
    }

    return {

        "evaluations":
            state.get(
                "evaluations",
                []
            ) + [evaluation_doc],

        "last_question":
            None,

        # Continue interview
        "phase":
            "interview"
    }



def router(
    state: InterviewState
):

    return state["phase"]

