import asyncio
import uuid
from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from backend.services.interview_service import (

    parse_resume_node,

    collect_job_role,

    interview_question_node,

    answer_evaluation_node
)
from backend.agents.interview_question_agent import InterviewPlannerAgent

from backend.rag.retriever import (
    retrieve_relevant_chunks
)
from backend.utils.mongodb_helpers import (
    save_session, load_session
)
from backend.rag.retriever import get_closest_role



# =========================================================
# Session Initialization
# =========================================================

async def initialize_interview_session(
    pdf_path: str,
    session_id: str = None
):

    session_id = session_id or str(
        uuid.uuid4()
    )

    # ==========================================
    # Initial State
    # ==========================================

    state = {

        "messages": [],

        "resume_pdf_path":
            pdf_path,

        "resume_text":
            None,

        "extracted_skills":
            None,

        "extracted_domains":
            None,

        "years_of_experience": None ,
        "job_role": None,
        "matched_role": None,

        "last_question":
            None,

        "question_history":
            [],

        "current_topic_index": 0,
        "current_subtopic_index": 0,
        "cached_retrieved_context": None,
        "query_history": [],

        "evaluations":
            [],
        "rag_mode": "normal",
    }

    # ==========================================
    # Parse Resume ONLY ONCE
    # ==========================================

    updates = await parse_resume_node(
        state
    )

    state.update(updates)

    # ==========================================
    # Ask Role
    # ==========================================

    updates = collect_job_role(
        state
    )

    state.update(updates)

    # ==========================================
    # Store Session
    # ==========================================

    save_session(
        session_id,
        state
    )

    return session_id, state


# =========================================================
# Submit Role
# =========================================================

async def submit_role(
    session_id: str,
    role: str
):

    state = load_session(
        session_id
    )

    # ==========================================
    # Add User Message
    # ==========================================

    state["messages"].append(

        HumanMessage(
            content=role
        )
    )

    # ==========================================
    # Process Role
    # ==========================================

    updates = collect_job_role(
        state
    )

    state.update(updates)

    matched_role = get_closest_role(state["job_role"])
    state["matched_role"] = matched_role
    
    if matched_role is not None:
        print(f"[Initialization] Valid RAG match found: '{matched_role}' for input '{state['job_role']}'")
        state["rag_mode"] = "book_rag"
    else:
        print(f"[Initialization] No book mapping for '{state['job_role']}'. Defaulting to standard LLM knowledge.")
        state["rag_mode"] = "normal"


    planner = InterviewPlannerAgent(name="InterviewPlanner", verbose=False)

    plan = await planner.execute(
        state
    )

    state["interview_plan"] = (
        plan.model_dump()
    )
    state["current_topic_index"] = 0
    state["current_subtopic_index"] = 0
    state["cached_retrieved_context"] = None

    updates = await interview_question_node(
        state
    )
    state.update(updates)

    # ==========================================
    # Save State
    # ==========================================

    save_session(
        session_id,
        state
    )

    return state


# =========================================================
# Submit Interview Answer
# =========================================================

async def submit_answer(
    session_id: str,
    answer: str
):

    state = load_session(
        session_id
    )

    # ==========================================
    # Add Candidate Response
    # ==========================================

    state["messages"].append(

        HumanMessage(
            content=answer
        )
    )

    # ==========================================
    # Evaluate Answer
    # ==========================================

    updates = await answer_evaluation_node(
        state
    )

    state.update(updates)

    latest_evaluation = (
        state["evaluations"][-1]
    )
    

    # ==========================================
    # Generate Next Question
    # ==========================================

    updates = await interview_question_node(
        state
    )

    state.update(updates)

    # ==========================================
    # Save Session
    # ==========================================

    save_session(
        session_id,
        state
    )

    latest_question_doc = (
        state["question_history"][-1]
    )

    return {

        "evaluation":
            latest_evaluation,
        "next_question":
            state["last_question"],
        "question_doc":
            latest_question_doc
    }


# =========================================================
# CLI TESTING
# =========================================================

async def main():

    print(
        "\n===== INITIALIZING SESSION =====\n"
    )

    session_id, state = (
        await initialize_interview_session(
            "Resume_1_1.pdf"
        )
    )

    print(
        f"Session ID: {session_id}"
    )

    print(
        "\n===== PROFILE =====\n"
    )

    print(
        "Skills:",
        state["extracted_skills"]
    )

    print(
        "\nDomains:",
        state["extracted_domains"]
    )

    # ==========================================
    # Role Question
    # ==========================================

    print(
        "\n===== ROLE QUESTION =====\n"
    )

    print(
        state["messages"][-1].content
    )

    role = input(
        "\nEnter Role: "
    )

    state = await submit_role(
        session_id,
        role
    )

    # ==========================================
    # First Interview Question
    # ==========================================

    print(
        "\n===== FIRST QUESTION =====\n"
    )

    print(
        state["last_question"]
    )

    # ==========================================
    # Interactive Interview Loop
    # ==========================================

    while True:

        latest_state = load_session(session_id)
        if latest_state.get("phase") == "completed":
            print("\n===== INTERVIEW COMPLETED =====\n")
            break

        answer = input(
            "\nYour Answer: "
        )

        if answer.lower() == "exit":
            break

        result = await submit_answer(

            session_id=session_id,

            answer=answer
        )

        print(
            "\n===== EVALUATION =====\n"
        )

        print(
            result["evaluation"]
        )

        print(
            "\n===== NEXT QUESTION =====\n"
        )

        print(
            result["next_question"]
        )

    print(
        "\n===== SESSION ENDED =====\n"
    )


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    asyncio.run(main())