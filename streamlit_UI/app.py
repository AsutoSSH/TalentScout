from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_PORT = os.getenv("PORT")
API_BASE = f"http://{BASE_URL}:{API_PORT}"
st.set_page_config(page_title="TalentScout Interview System", layout="wide")
st.markdown("""
    <style>
        /* Make chat input more prominent */
        div[data-testid="stChatInput"] {
            border: 2px solid #ff4b4b;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        div[data-testid="stChatInput"]:focus-within {
            border-color: #ff4b4b;
            box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.2);
            transform: scale(1.01);
        }
        
        /* Add a pulsing effect to draw attention */
        @keyframes subtle-pulse {
            0% { border-color: #ff4b4b; opacity: 1; }
            50% { border-color: #ff8c8c; opacity: 0.8; }
            100% { border-color: #ff4b4b; opacity: 1; }
        }
        
        div[data-testid="stChatInput"] {
            animation: subtle-pulse 2s infinite;
        }
        
        div[data-testid="stChatInput"]:focus-within {
            animation: none;
        }
    </style>
""", unsafe_allow_html=True)


def download_report(session_id: str):
    """Download the interview report PDF"""
    try:
        response = requests.get(f"{API_BASE}/report/{session_id}")
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Failed to download report: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error downloading report: {str(e)}")
        return None


# =========================================================
# Session State & Initialization
# =========================================================
for key, default in [
    ("session_id", None), ("current_question", None),
    ("interview_started", False), ("history", []),
    ("role_set", False), ("evaluation", None),
    ("waiting_for_next", False), ("focus_counter", 0),
    ("report_ready", False), ("show_report_button", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("AI Technical Interview System")

# =========================================================
# Sidebar - Control Panel & Setup
# =========================================================
with st.sidebar:
    st.header("Interview Control Panel")
    
    if not st.session_state.interview_started:
        uploaded_resume = st.file_uploader("Upload Resume PDF", type=["pdf"])
        rag_mode = st.selectbox("Interview Mode", options=["book_rag", "normal"])
        
        if st.button("Initialize Interview", use_container_width=True):
            if uploaded_resume is None:
                st.error("Please upload a resume.")
            else:
                with st.spinner("Parsing resume..."):
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_resume.read())
                        tmp_path = tmp.name

                    with open(tmp_path, "rb") as f:
                        files = {"resume": (uploaded_resume.name, f, "application/pdf")}
                        response = requests.post(f"{API_BASE}/session/create", files=files, data={"rag_mode": rag_mode})

                    if response.status_code != 200:
                        st.error(response.text)
                    else:
                        result = response.json()
                        st.session_state.session_id = result["session_id"]
                        st.session_state.interview_started = True
                        
                        # Store profile info in session state
                        st.session_state.skills = result.get('skills', [])
                        st.session_state.domains = result.get('domains', [])
                        
                        st.success("System initialized successfully.")
                        st.rerun()
    
    else:  # Interview has started
        st.success("Connected to Session")
        st.caption(f"ID: `{st.session_state.session_id}`")
        
        # Display stored profile information persistently
        if hasattr(st.session_state, 'skills') and st.session_state.skills:
            st.write("### Identified Profiles")
            st.caption(f"**Skills:** {', '.join(st.session_state.skills)}")
            st.caption(f"**Domains:** {', '.join(st.session_state.domains)}")
        
        # Show download button if report is ready (interview completed)
        if st.session_state.get('report_ready', False):
            report_pdf = download_report(st.session_state.session_id)
            if report_pdf:
                st.download_button(
                    label="📥 Download Final Report",
                    data=report_pdf,
                    file_name="interview_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        # Exit Interview button functionality
        if st.button("Exit Interview", type="primary", use_container_width=True):
            # Check if interview has any history (has questions answered)
            if len(st.session_state.history) > 0:
                # Offer to download report before exiting
                report_pdf = download_report(st.session_state.session_id)
                if report_pdf:
                    st.session_state.pending_report = report_pdf
                    st.session_state.show_report_button = True
                    
                    # Show success message
                    st.success("Interview session ended. Download your report below before closing.")
                    
                    # Create download button in the main area
                    st.download_button(
                        label="📊 Download Interview Report",
                        data=report_pdf,
                        file_name=f"interview_report_{st.session_state.session_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="exit_download_button"
                    )
                    
                    # Add a note
                    st.info("Click the download button above to save your report, then refresh the page to start a new interview.")
                    
                    # Reset session state but keep report ready
                    st.session_state.interview_started = False
                    st.session_state.current_question = None
                    st.session_state.role_set = False

                else:
                    st.warning("Could not generate report. Exiting anyway.")
                    st.session_state.interview_started = False
                    st.session_state.current_question = None
                    st.session_state.history = []
                    st.session_state.role_set = False
            else:
                # No interview history, just exit
                st.session_state.interview_started = False
                st.session_state.current_question = None
                st.session_state.history = []
                st.session_state.role_set = False
                st.rerun()

# =========================================================
# Chat Workspace Interface
# =========================================================
if st.session_state.interview_started:
    
    # Display all historical messages
    for entry in st.session_state.history:
        with st.chat_message("assistant"):
            st.write(entry["question"])
        with st.chat_message("user"):
            st.write(entry["answer"])
    
    # Show evaluation if it exists (from last answer)
    if st.session_state.evaluation:
        with st.expander("Evaluation", expanded=False):
            st.markdown(f"### Score: {st.session_state.evaluation['score']}/10")
            st.markdown(f"**Correctness:** {st.session_state.evaluation['correctness']}")
            if st.session_state.evaluation.get("missing_points"):
                st.markdown("### Missing Points")
                for p in st.session_state.evaluation["missing_points"]:
                    st.markdown(f"- {p}")
            st.markdown("### Suggestion")
            st.write(st.session_state.evaluation["improvement_suggestion"])
    
    # Role selection phase
    if not st.session_state.role_set:
        with st.chat_message("assistant"):
            st.write("Hello! I have analyzed your resume profile. Please let me know what targeted role you are interviewing for today to begin.")
        
        role = st.chat_input("e.g., Backend Engineer, Data Scientist...")
        if role:
            with st.spinner("Configuring customized interview matrix..."):
                payload = {"session_id": st.session_state.session_id, "role": role}
                response = requests.post(f"{API_BASE}/interview/role", json=payload)
                if response.status_code != 200:
                    st.error(response.text)
                else:
                    result = response.json()
                    st.session_state.current_question = result["question"]
                    st.session_state.role_set = True
                    st.session_state.evaluation = None
                    st.rerun()
    
    # Show current question and handle answer (just like testing script)
    elif st.session_state.current_question:
        with st.chat_message("assistant"):
            st.write(st.session_state.current_question)
        
        user_answer = st.chat_input("Type your technical explanation here...")
        if user_answer:
            if not user_answer.strip():
                st.warning("Input blank answers cannot be parsed.")
            else:
                with st.spinner("Analyzing answer context..."):
                    # This does both evaluation AND generates next question
                    payload = {"session_id": st.session_state.session_id, "answer": user_answer}
                    response = requests.post(f"{API_BASE}/interview/answer", json=payload)
                
                if response.status_code != 200:
                    st.error(response.text)
                else:
                    result = response.json()
                    
                    # Add current Q&A to history
                    st.session_state.history.append({
                        "question": st.session_state.current_question,
                        "answer": user_answer,
                    })
                    
                    # Store evaluation for display
                    st.session_state.evaluation = result["evaluation"]
                    
                    # Check if interview is complete (no next question)
                    if result["next_question"] is None:
                        st.session_state.current_question = None
                        st.session_state.report_ready = True
                        st.balloons()
                        st.success("🎓 Interview completed! Thank you for participating.")

                        report_pdf = download_report(st.session_state.session_id)
                        if report_pdf:
                            st.download_button(
                                label="📊 Download Your Interview Report",
                                data=report_pdf,
                                file_name=f"interview_report_{st.session_state.session_id}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="completion_download_button"
                            )
                            st.info("Click the button above to download your interview report.")

                    else:
                        # Set the next question (just like testing script)
                        st.session_state.current_question = result["next_question"]
                    st.session_state.focus_counter += 1
                    st.rerun()

else:
    with st.chat_message("assistant"):
        st.write("Welcome. To get started with the evaluation workspace, please upload your verification profile (Resume PDF) via the sidebar control menu.")

        if st.session_state.get('show_report_button', False) and st.session_state.get('pending_report'):
            st.success("✅ Interview session has ended. Click below to download your report:")
            st.download_button(
                label="📥 Download Interview Report",
                data=st.session_state.pending_report,
                file_name=f"interview_report_{st.session_state.session_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            # Clear after showing
            # st.session_state.show_report_button = False
            # st.session_state.pending_report = None
            # st.session_state.history = []  # Clear history after report download